"""Bounded worker-count helpers for AOT compilation."""

import os

from aiter_worker_limits import get_cpu_worker_budget

# Approximate observed peak RSS per worker, rounded to 1.5 GB.
_MEMORY_PER_WORKER_BYTES = 1_500_000_000
_CGROUP_TASKS_PER_WORKER = 12
_CGROUP_TASK_RESERVE = 16
_SUBPROCESS_JOB_ENV = {
    "CMAKE_BUILD_PARALLEL_LEVEL": "1",
    "MAKEFLAGS": "-j1",
    "NINJAFLAGS": "-j1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _available_memory_bytes() -> int:
    """Return currently available host memory as exposed by /proc."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return _MEMORY_PER_WORKER_BYTES


def _cgroup_worker_budget() -> int | None:
    """Return worker capacity from the cgroup task limit and measured fanout."""
    try:
        with open("/sys/fs/cgroup/pids.max") as max_file:
            raw_max = max_file.read().strip()
        if raw_max == "max":
            return None
        with open("/sys/fs/cgroup/pids.current") as current_file:
            current = int(current_file.read().strip())
        available = max(0, int(raw_max) - current - _CGROUP_TASK_RESERVE)
        return max(1, available // _CGROUP_TASKS_PER_WORKER)
    except (FileNotFoundError, OSError, ValueError):
        return None


def configure_worker_subprocesses() -> None:
    """Force compiler descendants of an AOT worker to use one build job."""
    subprocess_jobs = os.environ.get("AITER_SUBPROCESS_MAX_JOBS", "1")
    try:
        subprocess_jobs = str(max(1, int(subprocess_jobs)))
    except ValueError as exc:
        raise ValueError(
            "AITER_SUBPROCESS_MAX_JOBS must be an integer, "
            f"got {subprocess_jobs!r}"
        ) from exc
    os.environ["MAX_JOBS"] = subprocess_jobs
    for name, value in _SUBPROCESS_JOB_ENV.items():
        os.environ[name] = value


def get_worker_count() -> int:
    """Return and export a memory-, CPU-, and task-bounded worker count."""
    raw = os.environ.get("MAX_JOBS")
    if raw is not None:
        try:
            workers = max(1, int(raw))
        except ValueError as exc:
            raise ValueError(f"MAX_JOBS must be an integer, got {raw!r}") from exc
        os.environ["MAX_JOBS"] = str(workers)
        return workers

    cpu_budget = get_cpu_worker_budget()
    memory_budget = max(1, _available_memory_bytes() // _MEMORY_PER_WORKER_BYTES)
    budgets = [cpu_budget, memory_budget]
    task_budget = _cgroup_worker_budget()
    if task_budget is not None:
        budgets.append(task_budget)
    # Automatic workers are bounded by 80% of CPUs, available memory divided
    # by the observed per-worker RSS, and the cgroup task capacity when set.
    workers = max(1, min(budgets))
    os.environ["MAX_JOBS"] = str(workers)
    return workers
