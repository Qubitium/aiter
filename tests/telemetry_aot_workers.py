"""Run a bounded cold AOT build and report worker RSS and process fanout."""

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import psutil


DURATION_SECONDS = int(os.environ.get("AITER_TELEMETRY_SECONDS", "60"))
SAMPLE_SECONDS = 0.02
MEMORY_PER_WORKER_BYTES = 2 * 1024**3
CGROUP_TASKS_PER_WORKER = 11
CGROUP_TASK_RESERVE = 16
LOG_PATH = pathlib.Path("/tmp/aiter-aot-telemetry.log")
RESULT_PATH = pathlib.Path("/tmp/aiter-aot-telemetry.json")


def safe_cmdline(proc: psutil.Process) -> list[str]:
    try:
        return proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return []


def safe_name(proc: psutil.Process) -> str:
    try:
        return proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def safe_ppid(proc: psutil.Process) -> int:
    try:
        return proc.ppid()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return -1


def safe_rss(proc: psutil.Process) -> int:
    try:
        return proc.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return 0


def safe_threads(proc: psutil.Process) -> int:
    try:
        return proc.num_threads()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return 0


def children(proc: psutil.Process, recursive: bool = True) -> list[psutil.Process]:
    try:
        return proc.children(recursive=recursive)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def cgroup_pids() -> int:
    try:
        return int(pathlib.Path("/sys/fs/cgroup/pids.current").read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return 0


def cgroup_pids_max() -> int | None:
    try:
        raw = pathlib.Path("/sys/fs/cgroup/pids.max").read_text().strip()
        return None if raw == "max" else int(raw)
    except (FileNotFoundError, OSError, ValueError):
        return None


def main() -> None:
    env = os.environ.copy()
    env.pop("MAX_JOBS", None)
    env.setdefault("AITER_SUBPROCESS_MAX_JOBS", "1")

    available = psutil.virtual_memory().available
    cpu_budget = max(1, (os.cpu_count() or 1) - 1)
    memory_budget = max(1, available // MEMORY_PER_WORKER_BYTES)
    baseline_pids = cgroup_pids()
    pids_max = cgroup_pids_max()
    task_budget = (
        max(1, (pids_max - baseline_pids - CGROUP_TASK_RESERVE) // CGROUP_TASKS_PER_WORKER)
        if pids_max is not None
        else None
    )
    budgets = [cpu_budget, memory_budget]
    if task_budget is not None:
        budgets.append(task_budget)
    predicted_workers = min(budgets)

    command = [sys.executable, "-m", "aiter.aot.pa"]
    metrics = {
        "duration_seconds": DURATION_SECONDS,
        "available_memory_bytes": available,
        "memory_per_worker_assumption_bytes": MEMORY_PER_WORKER_BYTES,
        "memory_derived_workers": memory_budget,
        "cpu_derived_workers": cpu_budget,
        "cgroup_tasks_per_worker_assumption": CGROUP_TASKS_PER_WORKER,
        "cgroup_task_reserve": CGROUP_TASK_RESERVE,
        "cgroup_pids_max": pids_max,
        "task_derived_workers": task_budget,
        "predicted_workers": predicted_workers,
        "predicted_worker_rss_bytes": predicted_workers * MEMORY_PER_WORKER_BYTES,
        "predicted_peak_cgroup_pids": baseline_pids
        + predicted_workers * CGROUP_TASKS_PER_WORKER,
        "baseline_cgroup_pids": baseline_pids,
        "peak_cgroup_pids": baseline_pids,
        "peak_tree_processes": 0,
        "peak_tree_threads": 0,
        "peak_workers": 0,
        "peak_concurrent_hipconfig": 0,
        "unique_hipconfig_processes": 0,
        "peak_tree_rss_bytes": 0,
        "peak_worker_rss_bytes": 0,
        "peak_worker_subtree_rss_bytes": 0,
        "peak_descendants_per_worker": 0,
    }
    hipconfig_pids: set[int] = set()

    with LOG_PATH.open("w") as log:
        build = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        root = psutil.Process(build.pid)
        deadline = time.monotonic() + DURATION_SECONDS
        try:
            while time.monotonic() < deadline and build.poll() is None:
                tree = [root, *children(root)]
                metrics["peak_tree_processes"] = max(
                    metrics["peak_tree_processes"], len(tree)
                )
                metrics["peak_tree_rss_bytes"] = max(
                    metrics["peak_tree_rss_bytes"], sum(safe_rss(p) for p in tree)
                )
                metrics["peak_tree_threads"] = max(
                    metrics["peak_tree_threads"], sum(safe_threads(p) for p in tree)
                )
                metrics["peak_cgroup_pids"] = max(
                    metrics["peak_cgroup_pids"], cgroup_pids()
                )

                forkservers = [
                    p
                    for p in tree
                    if safe_ppid(p) == root.pid
                    and "multiprocessing.forkserver" in " ".join(safe_cmdline(p))
                ]
                workers: list[psutil.Process] = []
                for server in forkservers:
                    workers.extend(children(server, recursive=False))
                workers = [
                    p
                    for p in {p.pid: p for p in workers}.values()
                    if "/aiter/aot/pa.py" in " ".join(safe_cmdline(p))
                ]
                metrics["peak_workers"] = max(metrics["peak_workers"], len(workers))

                for worker in workers:
                    descendants = children(worker)
                    metrics["peak_worker_rss_bytes"] = max(
                        metrics["peak_worker_rss_bytes"], safe_rss(worker)
                    )
                    metrics["peak_worker_subtree_rss_bytes"] = max(
                        metrics["peak_worker_subtree_rss_bytes"],
                        safe_rss(worker) + sum(safe_rss(p) for p in descendants),
                    )
                    metrics["peak_descendants_per_worker"] = max(
                        metrics["peak_descendants_per_worker"], len(descendants)
                    )

                hipconfigs = [
                    p
                    for p in tree
                    if safe_name(p) == "hipconfig"
                    or any(pathlib.Path(arg).name == "hipconfig" for arg in safe_cmdline(p))
                ]
                hipconfig_pids.update(p.pid for p in hipconfigs)
                metrics["peak_concurrent_hipconfig"] = max(
                    metrics["peak_concurrent_hipconfig"], len(hipconfigs)
                )
                time.sleep(SAMPLE_SECONDS)
        finally:
            if build.poll() is None:
                os.killpg(build.pid, signal.SIGTERM)
                try:
                    build.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(build.pid, signal.SIGKILL)
                    build.wait()

    metrics["exit_code"] = build.returncode
    metrics["unique_hipconfig_processes"] = len(hipconfig_pids)
    metrics["observed_processes_per_worker"] = round(
        metrics["peak_tree_processes"] / max(1, metrics["peak_workers"]), 2
    )
    metrics["observed_peak_rss_per_worker_bytes"] = (
        metrics["peak_tree_rss_bytes"] // max(1, metrics["peak_workers"])
    )
    RESULT_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
