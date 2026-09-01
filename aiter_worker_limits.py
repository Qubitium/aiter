"""Dependency-free worker limits shared by setup and AITER runtime code."""

import os

CPU_CORE_COUNT_UTILIZATION = 0.80


def get_cpu_worker_budget(cpu_count: int | None = None) -> int:
    """Return at most 80% of logical CPUs, with one worker as the floor."""
    logical_cpus = (os.cpu_count() if cpu_count is None else cpu_count) or 1
    return max(1, int(logical_cpus * CPU_CORE_COUNT_UTILIZATION))
