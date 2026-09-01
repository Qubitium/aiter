"""Regression tests for the sampling-kernel AOT compile driver."""

import unittest
from unittest.mock import Mock, patch

from aiter.aot import sampling as driver


class SamplingAotTest(unittest.TestCase):
    def test_all_kernel_families_are_submitted_before_results_are_consumed(self):
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        submitted = []

        def submit(_worker, configs):
            configs = list(configs)
            submitted.append(configs)

            def results():
                self.assertEqual(len(submitted), 3)
                yield from [None] * len(configs)

            return results()

        executor.map.side_effect = submit
        with (
            patch.object(
                driver.concurrent.futures,
                "ProcessPoolExecutor",
                return_value=executor,
            ) as process_pool,
            patch.object(driver, "get_worker_count_for", return_value=20) as workers,
        ):
            driver.main()

        self.assertEqual([len(configs) for configs in submitted], [4, 8, 8])
        workers.assert_called_once_with(20)
        process_pool.assert_called_once_with(
            max_workers=20,
            initializer=driver.configure_worker_subprocesses,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
