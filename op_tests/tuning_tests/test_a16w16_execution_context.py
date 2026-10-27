# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import aiter.tuned_gemm as tuned
from aiter.execution_context import (
    execution_phase,
    execution_role,
    get_execution_phase,
    get_execution_role,
)


class TestExecutionContext(unittest.TestCase):
    def test_nested_context_and_exception_restore_defaults(self):
        self.assertIsNone(get_execution_role())
        self.assertIsNone(get_execution_phase())
        with execution_role("target"), execution_phase("prefill"):
            with self.assertRaises(RuntimeError), execution_role(
                "draft"
            ), execution_phase("decode"):
                self.assertEqual(get_execution_role(), "draft")
                self.assertEqual(get_execution_phase(), "decode")
                raise RuntimeError("restore context")
            self.assertEqual(get_execution_role(), "target")
            self.assertEqual(get_execution_phase(), "prefill")

            def other_runner():
                with execution_role("draft"), execution_phase("verify"):
                    return get_execution_role(), get_execution_phase()

            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(
                    pool.submit(other_runner).result(), ("draft", "verify")
                )
            self.assertEqual(get_execution_role(), "target")
            self.assertEqual(get_execution_phase(), "prefill")
        self.assertIsNone(get_execution_role())
        self.assertIsNone(get_execution_phase())

    def test_dispatch_cache_separates_role_and_phase(self):
        tuned.get_GEMM_A16W16_config.cache_clear()
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)
        args = {
            "M": 96,
            "N": 4096,
            "K": 1536,
            "bias": False,
            "dtype": "torch.bfloat16",
            "otype": "torch.bfloat16",
        }
        ordinary = {"libtype": "triton", "solidx": 0}
        with mock.patch.object(
            tuned, "get_gfx", return_value="gfx950"
        ), mock.patch.object(tuned, "get_cu_num", return_value=256), mock.patch.object(
            tuned, "get_GEMM_A16W16_tuned_config", return_value=ordinary
        ):
            for _ in range(2):
                for role in (None, "target", "draft"):
                    for phase in (None, "decode", "verify", "prefill"):
                        with self.subTest(role=role, phase=phase):
                            config = tuned.get_GEMM_A16W16_config(
                                **args, execution_role=role, execution_phase=phase
                            )
                            if (role, phase) == ("target", "prefill"):
                                self.assertEqual(
                                    config["kernelName"],
                                    "_ZN5aiter24bf16gemm_bf16_tn_256x256E",
                                )
                                self.assertEqual(config["splitK"], 1)
                            else:
                                self.assertEqual(config, ordinary)

    def test_phase_override_is_limited_to_supported_family(self):
        args = {
            "N": 4096,
            "K": 1536,
            "bias": False,
            "dtype": "torch.bfloat16",
            "otype": "torch.bfloat16",
            "scaleAB": False,
            "bpreshuffle": False,
            "execution_role": "target",
            "execution_phase": "prefill",
            "gfx": "gfx950",
            "cu_num": 256,
        }
        self.assertIsNotNone(tuned._get_execution_phase_invariant_config(**args))
        for changes in (
            {"N": 4097},
            {"K": 1537},
            {"bias": True},
            {"dtype": "torch.float16"},
            {"otype": "torch.float32"},
            {"scaleAB": True},
            {"bpreshuffle": True},
            {"gfx": "gfx942"},
            {"cu_num": 304},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(
                    tuned._get_execution_phase_invariant_config(**(args | changes))
                )


if __name__ == "__main__":
    unittest.main()
