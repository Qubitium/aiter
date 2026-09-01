"""Dependency-free worker limits shared by setup and AITER runtime code."""

import os

CPU_CORE_COUNT_UTILIZATION = 0.80
# Approximate peak RSS observed per AOT worker, rounded up to 1.5 GB.
EST_WORKER_RSS_BYTES = 1_500_000_000
CGROUP_TASKS_PER_WORKER = 12
CGROUP_TASK_RESERVE = 16
_CGROUP_PIDS_MAX_PATH = "/sys/fs/cgroup/pids.max"
_CGROUP_PIDS_CURRENT_PATH = "/sys/fs/cgroup/pids.current"
_WORKER_ENV = "AITER_MAX_JOBS"


def _process_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity when possible."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def get_cpu_worker_budget(cpu_count: int | None = None) -> int:
    """Return at most 80% of logical CPUs, with one worker as the floor."""
    logical_cpus = (_process_cpu_count() if cpu_count is None else cpu_count) or 1
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
    paths = (_CGROUP_PIDS_MAX_PATH, _CGROUP_PIDS_CURRENT_PATH)
    if not all(os.path.exists(path) for path in paths):
        return None

    try:
        with open(_CGROUP_PIDS_MAX_PATH) as max_file:
            raw_max = max_file.read().strip()
        if raw_max == "max":
            return None
        with open(_CGROUP_PIDS_CURRENT_PATH) as current_file:
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

    An explicit ``AITER_MAX_JOBS`` is an AITER-local override and is therefore
    only normalized to the one-worker floor. Automatic sizing takes the minimum
    of the CPU, available-memory, and cgroup task budgets.
    """
    raw = os.environ.get(_WORKER_ENV)
    if raw is not None:
        try:
            workers = max(1, int(raw))
        except ValueError as exc:
            raise ValueError(
                f"{_WORKER_ENV} must be an integer, got {raw!r}"
            ) from exc
        os.environ[_WORKER_ENV] = str(workers)
        return workers

    cpu_budget, memory_budget, task_budget = get_automatic_worker_budgets()
    budgets = [cpu_budget, memory_budget]
    if task_budget is not None:
        budgets.append(task_budget)
    workers = max(1, min(budgets))
    os.environ[_WORKER_ENV] = str(workers)
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
    os.environ[_WORKER_ENV] = subprocess_jobs
    # A process-pool child is already one of the top-level workers. Do not
    # divide its explicit nested budget by the parent process count again.
    os.environ.pop("PREBUILD_THREAD_NUM", None)
    os.environ.update(
        {
            "CMAKE_BUILD_PARALLEL_LEVEL": subprocess_jobs,
            "MAKEFLAGS": f"-j{subprocess_jobs}",
            "NINJAFLAGS": f"-j{subprocess_jobs}",
            "OMP_NUM_THREADS": subprocess_jobs,
            "OPENBLAS_NUM_THREADS": subprocess_jobs,
            "MKL_NUM_THREADS": subprocess_jobs,
            "NUMEXPR_NUM_THREADS": subprocess_jobs,
        }
    )
