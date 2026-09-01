"""Dependency-free worker limits shared by setup and AITER runtime code."""

import os

CPU_CORE_COUNT_UTILIZATION = 0.80
# Approximate peak RSS observed per AOT worker, rounded up to 1.5 GB.
EST_WORKER_RSS_BYTES = 1_500_000_000
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


def get_automatic_worker_budgets() -> tuple[int, int]:
    """Return the CPU and memory worker budgets."""
    return (
        get_cpu_worker_budget(),
        max(1, _available_memory_bytes() // EST_WORKER_RSS_BYTES),
    )


def get_worker_count() -> int:
    """Return and export the single AITER CPU worker budget.

    An explicit ``AITER_MAX_JOBS`` is an AITER-local override and is therefore
    only normalized to the one-worker floor. Automatic sizing takes the minimum
    of the CPU and available-memory budgets.
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

    workers = max(1, min(get_automatic_worker_budgets()))
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
    """Force compiler descendants of a process-pool worker to one job."""
    os.environ[_WORKER_ENV] = "1"
    # A process-pool child is already one of the top-level workers. Do not
    # divide its explicit nested budget by the parent process count again.
    os.environ.pop("PREBUILD_THREAD_NUM", None)
    os.environ.update(
        {
            "CMAKE_BUILD_PARALLEL_LEVEL": "1",
            "MAKEFLAGS": "-j1",
            "NINJAFLAGS": "-j1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
