"""Regression tests for the MLA decode AOT compile driver.

The driver and compile API diverged when both were introduced on 2025-05-29 in
commit 01864fa8e2347421bc5c314de8b584777ea991ec. This driver previously had no
unit-test coverage, allowing the mismatched arguments to remain unnoticed.
"""

import ast
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

_DRIVER = pathlib.Path(__file__).parents[1] / "aiter/aot/asm_mla_decode_fwd.py"
_COMPILE_IMPL = (
    pathlib.Path(__file__).parents[1]
    / "csrc/cpp_itfs/mla/asm_mla_decode_fwd.py"
)


def load_driver(compile_mock=None):
    compile_mock = compile_mock or Mock()

    compile_module = types.ModuleType("csrc.cpp_itfs.mla.asm_mla_decode_fwd")
    compile_module.compile = compile_mock
    worker_module = types.ModuleType("aiter.utility.worker_utils")
    worker_module.configure_worker_subprocesses = Mock()
    worker_module.get_worker_count = Mock(return_value=1)

    modules = {
        "csrc": types.ModuleType("csrc"),
        "csrc.cpp_itfs": types.ModuleType("csrc.cpp_itfs"),
        "csrc.cpp_itfs.mla": types.ModuleType("csrc.cpp_itfs.mla"),
        "csrc.cpp_itfs.mla.asm_mla_decode_fwd": compile_module,
        "aiter": types.ModuleType("aiter"),
        "aiter.utility": types.ModuleType("aiter.utility"),
        "aiter.utility.worker_utils": worker_module,
    }
    spec = importlib.util.spec_from_file_location("test_asm_mla_decode_aot_driver", _DRIVER)
    driver = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(driver)
    return driver, compile_mock


class AsmMlaDecodeAotTest(unittest.TestCase):
    def test_config_fields_match_compile_api(self):
        driver, _ = load_driver()
        tree = ast.parse(_COMPILE_IMPL.read_text())
        compile_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "compile"
        )
        compile_parameters = tuple(
            argument.arg for argument in compile_function.args.args[:6]
        )

        self.assertEqual(driver.MLAConfig._fields, compile_parameters)

    def test_process_config_matches_compile_api(self):
        driver, compile_mock = load_driver()
        config = driver.MLAConfig(
            gqa_ratio=16,
            page_size=1,
            q_dtype="__hip_bfloat16",
            kv_dtype="__hip_bfloat16",
            num_kv_splits=7,
            v_head_dim=512,
        )

        driver.process_config(config)

        compile_mock.assert_called_once_with(
            16,
            1,
            "__hip_bfloat16",
            "__hip_bfloat16",
            7,
            512,
        )

    def test_main_builds_all_gqa16_split_variants(self):
        driver, _ = load_driver()
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.map.return_value = iter([None] * 16)

        with patch.object(driver.concurrent.futures, "ProcessPoolExecutor", return_value=executor):
            driver.main()

        process_config, configs = executor.map.call_args.args
        self.assertIs(process_config, driver.process_config)
        self.assertEqual([config.num_kv_splits for config in configs], list(range(1, 17)))
        self.assertEqual({config.gqa_ratio for config in configs}, {16})
        self.assertEqual({config.q_dtype for config in configs}, {"__hip_bfloat16"})
        self.assertEqual({config.kv_dtype for config in configs}, {"__hip_bfloat16"})

    def test_main_surfaces_worker_compile_errors(self):
        driver, _ = load_driver()
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)

        def failed_result():
            raise RuntimeError("compile failed")
            yield

        executor.map.return_value = failed_result()
        with (
            patch.object(
                driver.concurrent.futures,
                "ProcessPoolExecutor",
                return_value=executor,
            ),
            self.assertRaisesRegex(RuntimeError, "compile failed"),
        ):
            driver.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
