# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests for lightweight tuning scans."""

import os
import unittest
from unittest import mock


class _FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, _other):
        return 0.009


class TestEventTimingPath(unittest.TestCase):
    def test_explicit_event_timing_skips_memory_and_profiler_setup(self):
        from aiter import test_common

        calls = []

        def operation():
            calls.append(None)
            return len(calls)

        test_common._CUDA_EVENT_PAIRS.clear()
        self.addCleanup(test_common._CUDA_EVENT_PAIRS.clear)
        with (
            mock.patch.object(
                test_common.torch.cuda, "Event", side_effect=lambda **_kw: _FakeEvent()
            ) as event,
            mock.patch.object(test_common.torch.cuda, "current_device", return_value=0),
            mock.patch.object(test_common.torch.cuda, "synchronize") as synchronize,
            mock.patch.object(
                test_common,
                "device_memory_profiling",
                side_effect=AssertionError("memory profiling entered"),
            ) as memory_profiling,
            mock.patch.object(
                test_common.tpf,
                "profile",
                side_effect=AssertionError("PyTorch profiler entered"),
            ) as profiler,
            mock.patch.object(
                test_common.copy,
                "deepcopy",
                side_effect=AssertionError("argument rotation entered"),
            ) as deepcopy,
        ):
            result, latency = test_common.run_perftest(
                operation,
                num_warmup=2,
                num_iters=3,
                use_cuda_event=True,
            )
            second_result, second_latency = test_common.run_perftest(
                operation,
                num_warmup=2,
                num_iters=3,
                use_cuda_event=True,
            )

        self.assertEqual(result, 5)
        self.assertEqual(second_result, 10)
        self.assertEqual(len(calls), 10)
        self.assertAlmostEqual(latency, 3.0)
        self.assertAlmostEqual(second_latency, 3.0)
        self.assertEqual(event.call_count, 2)
        synchronize.assert_not_called()
        memory_profiling.assert_not_called()
        profiler.assert_not_called()
        deepcopy.assert_not_called()


class TestMultiprocessFastScan(unittest.TestCase):
    def test_worker_forwards_event_timing(self):
        from aiter import test_common
        from aiter.utility import mp_tuner

        with (
            mock.patch.object(mp_tuner.torch, "device", return_value="cuda:0"),
            mock.patch.object(mp_tuner.torch.cuda, "set_device"),
            mock.patch.object(mp_tuner.torch.cuda, "synchronize"),
            mock.patch.object(
                test_common, "run_perftest", return_value=("result", 7.25)
            ) as run_perftest,
        ):
            result = mp_tuner.worker(
                0,
                "candidate",
                object(),
                [],
                {},
                use_cuda_event=True,
            )

        self.assertEqual(result, ("candidate", 7.25, 0.0))
        self.assertTrue(run_perftest.call_args.kwargs["use_cuda_event"])

    def test_validation_can_be_disabled_without_reference(self):
        from aiter.utility import mp_tuner

        task = (
            "candidate",
            None,
            (),
            object(),
            (),
            {},
            None,
            (),
            {},
            None,
        )
        with (
            mock.patch.object(mp_tuner.torch, "device", return_value="cuda:0"),
            mock.patch.object(mp_tuner.torch.cuda, "set_device"),
            mock.patch.object(
                mp_tuner, "worker", return_value=("candidate", 1.0, 0.0)
            ) as worker,
        ):
            result = mp_tuner.work_group(
                {os.getpid(): 0},
                fast_mode=0,
                err_ratio=0.05,
                in_data=(1, ()),
                tasks=task,
                use_cuda_event=True,
                validate_results=False,
            )

        self.assertEqual(result, [("candidate", 1.0, 0.0)])
        self.assertIsNone(worker.call_args.args[5])
        self.assertTrue(worker.call_args.args[-2])
        self.assertTrue(worker.call_args.args[-1])


if __name__ == "__main__":
    unittest.main()
