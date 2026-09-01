"""Bounded worker-count helpers for AOT compilation."""

import os

_MEMORY_PER_WORKER_BYTES = 250 * 1024 * 1024
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


def get_worker_count(default: int | None = None) -> int:
    """Return and export a memory- and CPU-bounded top-level worker count."""
    raw = os.environ.get("MAX_JOBS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError as exc:
            raise ValueError(f"MAX_JOBS must be an integer, got {raw!r}") from exc

    cpu_budget = max(1, (os.cpu_count() or 1) - 1)
    memory_budget = max(1, _available_memory_bytes() // _MEMORY_PER_WORKER_BYTES)
    workers = min(cpu_budget, memory_budget)
    os.environ["MAX_JOBS"] = str(workers)
    return workers
