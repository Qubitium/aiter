"""Bounded worker-count helpers for AOT compilation."""

import os


def _quota_cpu_count() -> int | None:
    """Return the cgroup v2 CPU quota, when one is configured."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota == "max":
            return None
        return max(1, int(int(quota) / int(period)))
    except (FileNotFoundError, OSError, ValueError, ZeroDivisionError):
        return None


def get_worker_count(default: int | None = None) -> int:
    """Return a worker count leaving one CPU free, respecting cgroup quota."""
    reported = max(1, os.cpu_count() or 1)
    budget = max(1, reported - 1)
    try:
        budget = min(budget, max(1, len(os.sched_getaffinity(0)) - 1))
    except (AttributeError, OSError):
        pass
    quota = _quota_cpu_count()
    if quota is not None:
        budget = min(budget, max(1, quota - 1))
    requested = default if default is not None else budget
    raw = os.environ.get("MAX_JOBS")
    if raw is not None:
        try:
            requested = int(raw)
        except ValueError as exc:
            raise ValueError(f"MAX_JOBS must be an integer, got {raw!r}") from exc
    if raw is not None:
        return max(1, requested)
    return min(budget, max(1, requested))
