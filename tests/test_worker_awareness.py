import inspect
import os
import unittest
from unittest.mock import patch

import aiter_worker_limits as worker_limits

configure_worker_subprocesses = worker_limits.configure_worker_subprocesses
get_automatic_worker_budgets = worker_limits.get_automatic_worker_budgets
get_cpu_worker_budget = worker_limits.get_cpu_worker_budget
get_worker_count = worker_limits.get_worker_count
get_worker_count_for = worker_limits.get_worker_count_for
get_worker_count_per_parent = worker_limits.get_worker_count_per_parent


class WorkerAwarenessTest(unittest.TestCase):
    def test_worker_count_accepts_no_per_caller_default(self):
        self.assertEqual(tuple(inspect.signature(get_worker_count).parameters), ())

    def test_cpu_budget_uses_at_most_eighty_percent(self):
        self.assertEqual(get_cpu_worker_budget(cpu_count=24), 19)
        self.assertEqual(get_cpu_worker_budget(cpu_count=4), 3)

    def test_cpu_budget_uses_process_available_cpus(self):
        with patch.object(worker_limits, "_process_cpu_count", return_value=24):
            self.assertEqual(get_cpu_worker_budget(), 19)

    def test_process_cpu_count_falls_back_to_affinity(self):
        with patch.object(
            os, "process_cpu_count", return_value=None, create=True
        ), patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}):
            self.assertEqual(worker_limits._process_cpu_count(), 4)

    def test_cpu_budget_always_returns_at_least_one(self):
        for logical_cpus in (0, 1):
            with self.subTest(logical_cpus=logical_cpus):
                self.assertEqual(get_cpu_worker_budget(cpu_count=logical_cpus), 1)
        with patch.object(worker_limits, "_process_cpu_count", return_value=1):
            self.assertEqual(get_cpu_worker_budget(), 1)

    def test_automatic_budgets_contain_only_cpu_and_memory(self):
        with patch.object(
            worker_limits, "get_cpu_worker_budget", return_value=6
        ), patch.object(
            worker_limits,
            "_available_memory_bytes",
            return_value=3 * worker_limits.EST_WORKER_RSS_BYTES,
        ):
            self.assertEqual(get_automatic_worker_budgets(), (6, 3))

    def test_four_cpu_worker_count_uses_eighty_percent(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            worker_limits, "_available_memory_bytes", return_value=10 * 1024**3
        ), patch.object(worker_limits, "_process_cpu_count", return_value=4):
            self.assertEqual(get_worker_count(), 3)
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "3")

    def test_explicit_aiter_max_jobs_is_not_capped(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "99"}, clear=True):
            self.assertEqual(get_worker_count(), 99)

    def test_framework_max_jobs_is_ignored(self):
        with patch.dict(os.environ, {"MAX_JOBS": "99"}, clear=True), patch.object(
            worker_limits, "_available_memory_bytes", return_value=10 * 1024**3
        ), patch.object(worker_limits, "_process_cpu_count", return_value=4):
            self.assertEqual(get_worker_count(), 3)
            self.assertEqual(os.environ["MAX_JOBS"], "99")
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "3")

    def test_explicit_lower_aiter_max_jobs_is_honored(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "1"}, clear=True):
            self.assertEqual(get_worker_count(), 1)

    def test_nonpositive_aiter_max_jobs_is_clamped_and_exported(self):
        for raw_value in ("0", "-7"):
            with self.subTest(raw_value=raw_value), patch.dict(
                os.environ, {"AITER_MAX_JOBS": raw_value}, clear=True
            ):
                self.assertEqual(get_worker_count(), 1)
                self.assertEqual(os.environ["AITER_MAX_JOBS"], "1")

    def test_zero_memory_capacity_still_returns_one_worker(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            worker_limits, "_available_memory_bytes", return_value=0
        ), patch.object(worker_limits, "_process_cpu_count", return_value=1):
            self.assertEqual(get_worker_count(), 1)
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "1")

    def test_available_memory_caps_default_workers(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            worker_limits,
            "_available_memory_bytes",
            return_value=4 * worker_limits.EST_WORKER_RSS_BYTES,
        ), patch.object(worker_limits, "_process_cpu_count", return_value=64):
            self.assertEqual(get_worker_count(), 4)
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "4")

    def test_worker_descendants_are_forced_to_one_job(self):
        with patch.dict(
            os.environ,
            {
                "AITER_MAX_JOBS": "23",
                "MAX_JOBS": "64",
            },
            clear=True,
        ):
            configure_worker_subprocesses()
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "1")
            self.assertEqual(os.environ["MAX_JOBS"], "64")
            self.assertEqual(os.environ["CMAKE_BUILD_PARALLEL_LEVEL"], "1")
            self.assertEqual(os.environ["MAKEFLAGS"], "-j1")
            self.assertEqual(os.environ["NINJAFLAGS"], "-j1")

    def test_nested_worker_share_never_returns_zero(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "1"}, clear=True):
            self.assertEqual(get_worker_count_per_parent(5), 1)
            self.assertEqual(get_worker_count_per_parent(0), 1)

    def test_nested_worker_share_divides_the_global_budget(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "19"}, clear=True):
            self.assertEqual(get_worker_count_per_parent(5), 3)

    def test_work_capped_worker_count_never_returns_zero(self):
        with patch.dict(os.environ, {"AITER_MAX_JOBS": "19"}, clear=True):
            self.assertEqual(get_worker_count_for(0), 1)
            self.assertEqual(get_worker_count_for(3), 3)

    def test_one_job_reaches_all_descendant_controls(self):
        with patch.dict(
            os.environ,
            {"PREBUILD_THREAD_NUM": "19"},
            clear=True,
        ):
            configure_worker_subprocesses()
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "1")
            self.assertEqual(os.environ["CMAKE_BUILD_PARALLEL_LEVEL"], "1")
            self.assertEqual(os.environ["MAKEFLAGS"], "-j1")
            self.assertEqual(os.environ["NINJAFLAGS"], "-j1")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertNotIn("PREBUILD_THREAD_NUM", os.environ)


if __name__ == "__main__":
    unittest.main(verbosity=2)
