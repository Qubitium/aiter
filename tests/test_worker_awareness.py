import importlib.util
import os
import pathlib
import unittest
from unittest.mock import patch

_HELPER = pathlib.Path(__file__).parents[1] / "aiter/utility/worker_utils.py"
_SPEC = importlib.util.spec_from_file_location("aiter_worker_utils", _HELPER)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
get_worker_count = _MODULE.get_worker_count
configure_worker_subprocesses = _MODULE.configure_worker_subprocesses


class WorkerAwarenessTest(unittest.TestCase):
    def test_legacy_formula_reproduces_four_worker_case(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(os, "cpu_count", return_value=4):
            legacy_count = int(os.environ.get("MAX_JOBS", os.cpu_count() or 16))
            self.assertEqual(legacy_count, 4)

    def test_four_cpu_reproduction_leaves_one_cpu_free(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _MODULE, "_available_memory_bytes", return_value=10 * 1024**3
        ), patch.object(_MODULE, "_cgroup_worker_budget", return_value=None), patch.object(
            os, "cpu_count", return_value=4
        ):
            self.assertEqual(get_worker_count(), 3)
            self.assertEqual(os.environ["MAX_JOBS"], "3")

    def test_explicit_max_jobs_is_not_capped(self):
        with patch.dict(os.environ, {"MAX_JOBS": "99"}, clear=True), patch.object(
            os, "cpu_count", return_value=4
        ):
            self.assertEqual(get_worker_count(), 99)

    def test_explicit_lower_max_jobs_is_honored(self):
        with patch.dict(os.environ, {"MAX_JOBS": "1"}, clear=True), patch.object(
            os, "cpu_count", return_value=4
        ):
            self.assertEqual(get_worker_count(), 1)

    def test_nonpositive_max_jobs_is_clamped_and_exported(self):
        for raw_value in ("0", "-7"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"MAX_JOBS": raw_value}, clear=True
            ):
                self.assertEqual(get_worker_count(), 1)
                self.assertEqual(os.environ["MAX_JOBS"], "1")

    def test_zero_capacity_probes_still_return_one_worker(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _MODULE, "_available_memory_bytes", return_value=0
        ), patch.object(_MODULE, "_cgroup_worker_budget", return_value=0), patch.object(
            os, "cpu_count", return_value=None
        ):
            self.assertEqual(get_worker_count(), 1)
            self.assertEqual(os.environ["MAX_JOBS"], "1")

    def test_available_memory_caps_default_workers(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _MODULE,
            "_available_memory_bytes",
            return_value=4 * _MODULE._MEMORY_PER_WORKER_BYTES,
        ), patch.object(_MODULE, "_cgroup_worker_budget", return_value=None), patch.object(
            os, "cpu_count", return_value=64
        ):
            self.assertEqual(get_worker_count(), 4)
            self.assertEqual(os.environ["MAX_JOBS"], "4")

    def test_cgroup_task_capacity_caps_default_workers(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _MODULE, "_available_memory_bytes", return_value=128 * 1024**3
        ), patch.object(_MODULE, "_cgroup_worker_budget", return_value=7), patch.object(
            os, "cpu_count", return_value=64
        ):
            self.assertEqual(get_worker_count(), 7)
            self.assertEqual(os.environ["MAX_JOBS"], "7")

    def test_worker_descendants_default_to_one_job(self):
        with patch.dict(os.environ, {"MAX_JOBS": "23"}, clear=True):
            configure_worker_subprocesses()
            self.assertEqual(os.environ["MAX_JOBS"], "1")
            self.assertEqual(os.environ["CMAKE_BUILD_PARALLEL_LEVEL"], "1")
            self.assertEqual(os.environ["MAKEFLAGS"], "-j1")
            self.assertEqual(os.environ["NINJAFLAGS"], "-j1")

    def test_zero_subprocess_jobs_is_clamped_to_one(self):
        with patch.dict(
            os.environ, {"AITER_SUBPROCESS_MAX_JOBS": "0"}, clear=True
        ):
            configure_worker_subprocesses()
            self.assertEqual(os.environ["MAX_JOBS"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
