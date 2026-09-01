"""Dependency-free worker limits shared by setup and AITER runtime code."""

import os

CPU_CORE_COUNT_UTILIZATION = 0.80
# Approximate peak RSS observed per AOT worker, rounded up to 1.5 GB.
EST_WORKER_RSS_BYTES = 1_500_000_000
CGROUP_TASKS_PER_WORKER = 12
CGROUP_TASK_RESERVE = 16
_SUBPROCESS_JOB_ENV = {
    "CMAKE_BUILD_PARALLEL_LEVEL": "1",
    "MAKEFLAGS": "-j1",
    "NINJAFLAGS": "-j1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def get_cpu_worker_budget(cpu_count: int | None = None) -> int:
    """Return at most 80% of logical CPUs, with one worker as the floor."""
    logical_cpus = (os.cpu_count() if cpu_count is None else cpu_count) or 1
    return max(1, int(logical_cpus * CPU_CORE_COUNT_UTILIZATION))


def _available_memory_bytes() -> int:
    """Return currently available host memory as exposed by /proc."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return EST_WORKER_RSS_BYTES


def _cgroup_worker_budget() -> int | None:
    """Return worker capacity from the cgroup task limit and measured fanout."""
    try:
        with open("/sys/fs/cgroup/pids.max") as max_file:
            raw_max = max_file.read().strip()
        if raw_max == "max":
            return None
        with open("/sys/fs/cgroup/pids.current") as current_file:
            current = int(current_file.read().strip())
        available = max(0, int(raw_max) - current - CGROUP_TASK_RESERVE)
        return max(1, available // CGROUP_TASKS_PER_WORKER)
    except (FileNotFoundError, OSError, ValueError):
        return None


def get_automatic_worker_budgets() -> tuple[int, int, int | None]:
    """Return the CPU, memory, and optional cgroup task worker budgets."""
    return (
        get_cpu_worker_budget(),
        max(1, _available_memory_bytes() // EST_WORKER_RSS_BYTES),
        _cgroup_worker_budget(),
    )


def get_worker_count() -> int:
    """Return and export the single AITER CPU worker budget.

    An explicit ``MAX_JOBS`` is a user override and is therefore only normalized
    to the one-worker floor. Automatic sizing takes the minimum of the CPU,
    available-memory, and cgroup task budgets.
    """
    raw = os.environ.get("MAX_JOBS")
    if raw is not None:
        try:
            workers = max(1, int(raw))
        except ValueError as exc:
            raise ValueError(f"MAX_JOBS must be an integer, got {raw!r}") from exc
        os.environ["MAX_JOBS"] = str(workers)
        return workers

    cpu_budget, memory_budget, task_budget = get_automatic_worker_budgets()
    budgets = [cpu_budget, memory_budget]
    if task_budget is not None:
        budgets.append(task_budget)
    workers = max(1, min(budgets))
    os.environ["MAX_JOBS"] = str(workers)
    return workers


def get_worker_count_per_parent(parent_count: int) -> int:
    """Return each parent's nested-worker share, never fewer than one."""
    parents = max(1, int(parent_count))
    return max(1, get_worker_count() // parents)


def get_worker_count_for(work_count: int) -> int:
    """Cap the global worker budget to available work, with a floor of one."""
    return min(get_worker_count(), max(1, int(work_count)))


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
