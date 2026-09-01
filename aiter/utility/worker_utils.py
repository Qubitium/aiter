"""Compatibility imports for the centralized AITER worker policy."""

from aiter_worker_limits import configure_worker_subprocesses, get_worker_count

__all__ = ["configure_worker_subprocesses", "get_worker_count"]
