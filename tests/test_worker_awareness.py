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


class WorkerAwarenessTest(unittest.TestCase):
    def test_legacy_formula_reproduces_four_worker_case(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(os, "cpu_count", return_value=4):
            legacy_count = int(os.environ.get("MAX_JOBS", os.cpu_count() or 16))
            self.assertEqual(legacy_count, 4)

    def test_four_cpu_reproduction_leaves_one_cpu_free(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(_MODULE, "_quota_cpu_count", return_value=None), patch.object(os, "cpu_count", return_value=4), patch.object(
            os, "sched_getaffinity", return_value={0, 1, 2, 3}
        ):
            self.assertEqual(get_worker_count(), 3)

    def test_max_jobs_cannot_exceed_cpu_budget(self):
        with patch.dict(os.environ, {"MAX_JOBS": "99"}), patch.object(_MODULE, "_quota_cpu_count", return_value=None), patch.object(
            os, "cpu_count", return_value=4
        ), patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}):
            self.assertEqual(get_worker_count(), 3)

    def test_explicit_lower_max_jobs_is_honored(self):
        with patch.dict(os.environ, {"MAX_JOBS": "1"}), patch.object(_MODULE, "_quota_cpu_count", return_value=None), patch.object(
            os, "cpu_count", return_value=4
        ), patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}):
            self.assertEqual(get_worker_count(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
