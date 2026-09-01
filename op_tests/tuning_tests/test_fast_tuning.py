# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests for lightweight tuning scans."""

import os
import unittest
from types import SimpleNamespace
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
            mock.patch.object(test_common.logger, "debug") as debug_log,
            mock.patch.object(test_common.logger, "info") as info_log,
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
        self.assertEqual(debug_log.call_count, 2)
        info_log.assert_not_called()


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


class TestMxfp4AccuracyFunnel(unittest.TestCase):
    @staticmethod
    def _candidate(name):
        return {"kernelName1": f"g1_{name}", "kernelName2": f"g2_{name}", "us": 0}

    def test_normalized_timing_ties_prefer_accuracy(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        fastest = self._candidate("fastest")
        fastest.update(us=100.0, err1=0.03)
        accurate_tie = self._candidate("accurate_tie")
        accurate_tie.update(us=100.99, err1=0.01)
        outside_tie = self._candidate("outside_tie")
        outside_tie.update(us=101.0, err1=0.001)

        ranked = gemm_moe_tune._rank_mxfp4_candidates(
            [outside_tie, fastest, accurate_tie]
        )

        self.assertEqual(
            [candidate["kernelName1"] for candidate in ranked],
            ["g1_accurate_tie", "g1_fastest", "g1_outside_tie"],
        )

    def test_final_global_gate_prefers_accuracy_without_chaining(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        fastest = self._candidate("fastest")
        fastest.update(us=100.0, err1=0.03)
        accurate_tie = self._candidate("accurate_tie")
        accurate_tie.update(us=100.9, err1=0.01)
        chained_only = self._candidate("chained_only")
        chained_only.update(us=101.8, err1=0.001)

        selected = gemm_moe_tune._select_mxfp4_finalist(
            [chained_only, fastest, accurate_tie]
        )

        self.assertIs(selected, accurate_tie)

    def test_final_global_gate_includes_exactly_one_percent(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        fastest = self._candidate("fastest")
        fastest.update(us=100.0, err1=0.03)
        accurate_boundary = self._candidate("accurate_boundary")
        accurate_boundary.update(us=101.0, err1=0.01)

        selected = gemm_moe_tune._select_mxfp4_finalist(
            [fastest, accurate_boundary]
        )

        self.assertIs(selected, accurate_boundary)

    def test_inaccurate_candidate_is_rejected_before_timing(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(
            gemm_moe_tune.Mxfp4FlydslTuner
        )
        row = {
            "expert": 8,
            "model_dim": 256,
            "inter_dim": 256,
            "token": 1,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        args = SimpleNamespace(fast_scan=True, errRatio=0.1, warmup=1, iters=1)
        with (
            mock.patch.object(tuner, "_port_e2e", return_value="output"),
            mock.patch.object(
                gemm_moe_tune, "cosine_diff_compare", return_value=0.2
            ),
            mock.patch(
                "aiter.test_common.run_perftest",
                side_effect=AssertionError("inaccurate candidate was timed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cosine err_ratio"):
                tuner._run_candidate(
                    row,
                    self._candidate("bad"),
                    args,
                    data="fixture",
                    reference="reference",
                )

    def test_finalists_are_revalidated_and_failures_are_discarded(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(
            gemm_moe_tune.Mxfp4FlydslTuner
        )
        fast = self._candidate("fast_bad")
        accurate = self._candidate("accurate")
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=2,
            fast_scan_iters=8,
            fast_scan_final_iters=100,
            fast_scan_finalists=2,
            timeout=0,
            errRatio=0.1,
            warmup=2,
            iters=101,
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        calls = []

        def run_candidate(_row, candidate, _args, **kwargs):
            calls.append((candidate["kernelName1"], kwargs))
            if kwargs.get("num_iters") == 8:
                candidate["us"] = (
                    1.0 if "fast_bad" in candidate["kernelName1"] else 2.0
                )
                return candidate["us"]
            if "fast_bad" in candidate["kernelName1"]:
                raise RuntimeError("finalist accuracy regression")
            candidate["us"] = 2.0
            return candidate["us"]

        with (
            mock.patch.object(tuner, "_candidate_rows", return_value=[fast, accurate]),
            mock.patch.object(
                tuner, "_precompile_candidates", return_value=(2, 0)
            ) as precompile,
            mock.patch.object(tuner, "_prepare_case", return_value="fixture"),
            mock.patch.object(tuner, "_torch_ref", return_value="reference"),
            mock.patch.object(tuner, "_run_candidate", side_effect=run_candidate),
            mock.patch("builtins.print"),
        ):
            best = tuner._tune_one_shape(row, args)

        self.assertEqual(best["kernelName1"], "g1_accurate")
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(kwargs["reference"] == "reference" for _, kwargs in calls))
        self.assertEqual(
            [call[1]["num_warmup"] for call in calls], [2, 2, 0, 0]
        )
        self.assertEqual(
            [call[1]["num_iters"] for call in calls], [8, 8, 100, 100]
        )
        precompile.assert_called_once()

    def test_warmup_outputs_are_accuracy_checks_outside_timing(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(
            gemm_moe_tune.Mxfp4FlydslTuner
        )
        args = SimpleNamespace(fast_scan=True, errRatio=0.1, warmup=2, iters=8)
        row = {
            "expert": 8,
            "model_dim": 256,
            "inter_dim": 256,
            "token": 1,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }

        def run_perftest(operation, **kwargs):
            self.assertEqual(kwargs["num_warmup"], 0)
            self.assertEqual(kwargs["num_iters"], 8)
            self.assertFalse(kwargs.get("use_cuda_event", False))
            self.assertEqual(kwargs["profile_warm_iters"], 0)
            self.assertEqual(kwargs["num_rotate_args"], 1)
            return operation(), 7.0

        with (
            mock.patch.object(
                tuner, "_port_e2e", side_effect=["warm-1", "warm-2", "timed"]
            ) as port,
            mock.patch.object(
                gemm_moe_tune,
                "cosine_diff_compare",
                side_effect=[0.01, 0.02, 0.03],
            ) as compare,
            mock.patch("aiter.test_common.run_perftest", side_effect=run_perftest),
        ):
            candidate = self._candidate("accurate")
            tuner._run_candidate(
                row,
                candidate,
                args,
                data="fixture",
                reference="reference",
                num_warmup=2,
                num_iters=8,
            )

        self.assertEqual(port.call_count, 3)
        self.assertEqual(compare.call_count, 3)
        self.assertEqual(candidate["err1"], 0.03)

    def test_accuracy_failure_aborts_the_current_shape(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(
            gemm_moe_tune.Mxfp4FlydslTuner
        )
        candidates = [self._candidate("bad"), self._candidate("never-run")]
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=2,
            fast_scan_iters=8,
            fast_scan_final_iters=100,
            fast_scan_finalists=2,
            timeout=0,
            errRatio=0.1,
            warmup=2,
            iters=101,
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        with (
            mock.patch.object(tuner, "_candidate_rows", return_value=candidates),
            mock.patch.object(tuner, "_precompile_candidates", return_value=(2, 0)),
            mock.patch.object(tuner, "_prepare_case", return_value="fixture"),
            mock.patch.object(tuner, "_torch_ref", return_value="reference"),
            mock.patch.object(
                tuner,
                "_run_candidate",
                side_effect=gemm_moe_tune.Mxfp4AccuracyError("warmup failed"),
            ) as run_candidate,
            mock.patch("builtins.print"),
        ):
            result = tuner._tune_one_shape(row, args)

        self.assertEqual(run_candidate.call_count, 1)
        self.assertEqual(result["us"], tuner.INVALID_TIME)
        self.assertTrue(result["kernelName1"].startswith("FAILED accuracy:"))


class TestParallelMxfp4Precompile(unittest.TestCase):
    def test_uses_forkserver_pool_and_reports_failed_cache_jobs(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(
            gemm_moe_tune.Mxfp4FlydslTuner
        )
        row = {"token": 1, "model_dim": 256}
        candidates = [{"kernelName1": "g1", "kernelName2": "g2"}]
        jobs = [
            {"kernel_name": "g1"},
            {"kernel_name": "g2"},
            {"kernel_name": "g3"},
        ]
        results = [
            {
                "kernel_name": "batch 1",
                "compile_time": 1.0,
                "results": [
                    {"kernel_name": "g1", "compile_time": 0.5},
                    {"kernel_name": "g3", "compile_time": None},
                ],
            },
            {
                "kernel_name": "batch 2",
                "compile_time": 0.5,
                "results": [{"kernel_name": "g2", "compile_time": 0.5}],
            },
        ]
        with (
            mock.patch(
                "aiter.aot.flydsl.mxfp4_moe.jobs_from_rows", return_value=jobs
            ) as jobs_from_rows,
            mock.patch(
                "aiter.aot.flydsl.common.run_jobs_parallel", return_value=results
            ) as run_jobs_parallel,
            mock.patch("aiter_worker_limits.get_worker_count_for", return_value=2),
            mock.patch("builtins.print"),
        ):
            total, failed = tuner._precompile_candidates(row, candidates)

        self.assertEqual((total, failed), (3, 1))
        jobs_from_rows.assert_called_once_with(
            [{"token": 1, "model_dim": 256, **candidates[0]}],
            include_bias_variants=False,
        )
        self.assertEqual(
            run_jobs_parallel.call_args.args[1],
            [{"jobs": [jobs[0], jobs[2]]}, {"jobs": [jobs[1]]}],
        )
        self.assertEqual(
            run_jobs_parallel.call_args.kwargs["start_method"], "forkserver"
        )

    def test_precompile_failure_preserves_runtime_jit_fallback(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(
            gemm_moe_tune.Mxfp4FlydslTuner
        )
        with (
            mock.patch(
                "aiter.aot.flydsl.mxfp4_moe.jobs_from_rows",
                side_effect=RuntimeError("compiler unavailable"),
            ),
            mock.patch("builtins.print") as output,
        ):
            result = tuner._precompile_candidates({}, [{}])

        self.assertEqual(result, (0, 0))
        self.assertIn("falling back to runtime JIT", output.call_args.args[0])

    def test_flydsl_aot_pool_uses_central_worker_budget(self):
        from aiter.aot.flydsl import common

        with mock.patch.object(
            common, "get_worker_count_for", return_value=7
        ) as worker_count:
            self.assertEqual(common.get_max_workers(23), 7)
        worker_count.assert_called_once_with(23)


if __name__ == "__main__":
    unittest.main()
