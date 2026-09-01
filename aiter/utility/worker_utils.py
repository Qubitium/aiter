"""Bounded worker-count helpers for AOT compilation."""

import os


def get_worker_count(default: int | None = None) -> int:
    """Return a safe worker count while leaving one CPU for system work."""
    reported = max(1, os.cpu_count() or 1)
    budget = max(1, reported - 1)
    try:
        budget = min(budget, max(1, len(os.sched_getaffinity(0)) - 1))
    except (AttributeError, OSError):
        pass
    requested = default if default is not None else budget
    raw = os.environ.get("MAX_JOBS")
    if raw is not None:
        try:
            requested = int(raw)
        except ValueError as exc:
            raise ValueError(f"MAX_JOBS must be an integer, got {raw!r}") from exc
    return min(budget, max(1, requested))
