"""Worker limits shared by setup and AITER runtime code."""

import os
import posixpath

CPU_CORE_COUNT_UTILIZATION = 0.80
# Approximate peak RSS observed per AOT worker, rounded up to 1.5 GB.
EST_WORKER_RSS_BYTES = 1_500_000_000
_WORKER_ENV = "AITER_MAX_JOBS"
_PROC_SELF_CGROUP_PATH = "/proc/self/cgroup"
_PROC_SELF_MOUNTINFO_PATH = "/proc/self/mountinfo"


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


def _host_available_memory_bytes() -> int:
    """Return currently available host memory as reported by psutil."""
    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available))
    except Exception:  # noqa: BLE001
        return EST_WORKER_RSS_BYTES


def _decode_mountinfo_path(path: str) -> str:
    """Decode the escapes used for mount paths in /proc/self/mountinfo."""
    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        path = path.replace(encoded, decoded)
    return path


def _resolve_cgroup_directory(
    mount_root: str, mount_point: str, membership_path: str
) -> str | None:
    """Map a cgroup membership path into its mounted filesystem path."""
    mount_root = posixpath.normpath(mount_root)
    membership_path = posixpath.normpath(membership_path)
    if membership_path == mount_root:
        relative = ""
    elif mount_root == "/":
        relative = membership_path.lstrip("/")
    elif membership_path.startswith(f"{mount_root.rstrip('/')}/"):
        relative = membership_path[len(mount_root) :].lstrip("/")
    else:
        return None
    return os.path.join(mount_point, *relative.split("/")) if relative else mount_point


def _cgroup_memory_directories() -> list[tuple[str, str]]:
    """Return current-to-root memory cgroup directories for v2 or v1."""
    try:
        with open(_PROC_SELF_CGROUP_PATH) as cgroup_file:
            cgroup_lines = cgroup_file.readlines()
        with open(_PROC_SELF_MOUNTINFO_PATH) as mountinfo_file:
            mountinfo_lines = mountinfo_file.readlines()
    except (FileNotFoundError, OSError):
        return []

    unified_path = None
    memory_path = None
    for line in cgroup_lines:
        try:
            hierarchy, controllers, path = line.rstrip("\n").split(":", 2)
        except ValueError:
            continue
        if hierarchy == "0" and not controllers:
            unified_path = path
        if "memory" in controllers.split(","):
            memory_path = path

    directories = []
    seen_directories = set()
    for line in mountinfo_lines:
        try:
            mount_fields, filesystem_fields = line.rstrip("\n").split(" - ", 1)
            mount_fields = mount_fields.split()
            filesystem_fields = filesystem_fields.split()
            mount_root = _decode_mountinfo_path(mount_fields[3])
            mount_point = _decode_mountinfo_path(mount_fields[4])
            filesystem_type = filesystem_fields[0]
            super_options = filesystem_fields[2].split(",")
        except (IndexError, ValueError):
            continue

        version = None
        membership_path = None
        if filesystem_type == "cgroup2" and unified_path is not None:
            version = "v2"
            membership_path = unified_path
        elif (
            filesystem_type == "cgroup"
            and memory_path is not None
            and "memory" in super_options
        ):
            version = "v1"
            membership_path = memory_path
        if version is None:
            continue

        current = _resolve_cgroup_directory(
            mount_root, mount_point, membership_path
        )
        if current is None:
            continue

        while True:
            directory = (version, current)
            if directory not in seen_directories:
                directories.append(directory)
                seen_directories.add(directory)
            if current == mount_point:
                break
            parent = os.path.dirname(current)
            if parent == current or os.path.commonpath((mount_point, parent)) != mount_point:
                break
            current = parent
    return directories


def _cgroup_memory_remaining_bytes() -> int | None:
    """Return the tightest finite remaining-memory bound across cgroup ancestors."""
    remaining = None
    for version, directory in _cgroup_memory_directories():
        if version == "v2":
            limit_path = os.path.join(directory, "memory.max")
            usage_path = os.path.join(directory, "memory.current")
        else:
            limit_path = os.path.join(directory, "memory.limit_in_bytes")
            usage_path = os.path.join(directory, "memory.usage_in_bytes")

        try:
            with open(limit_path) as limit_file:
                raw_limit = limit_file.read().strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
        except (FileNotFoundError, OSError, ValueError):
            continue

        try:
            with open(usage_path) as usage_file:
                usage = int(usage_file.read().strip())
        except (FileNotFoundError, OSError, ValueError):
            usage = 0

        candidate = max(0, limit - usage)
        remaining = candidate if remaining is None else min(remaining, candidate)
    return remaining


def _available_memory_bytes() -> int:
    """Return memory available under both host and cgroup constraints."""
    host_available = _host_available_memory_bytes()
    cgroup_remaining = _cgroup_memory_remaining_bytes()
    return (
        host_available
        if cgroup_remaining is None
        else min(host_available, cgroup_remaining)
    )


def get_automatic_worker_budgets() -> tuple[int, int]:
    """Return the CPU and memory worker budgets."""
    return (
        get_cpu_worker_budget(),
        max(1, _available_memory_bytes() // EST_WORKER_RSS_BYTES),
    )


def get_worker_count() -> int:
    """Return and export the single AITER CPU worker budget.

    An explicit ``AITER_MAX_JOBS`` is an unsafe expert override: it is honored
    after normalization to the one-worker floor and bypasses automatic CPU and
    memory caps. Automatic sizing takes the minimum of the CPU and effective
    available-memory budgets.
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


def get_worker_count_for(work_count: int) -> int:
    """Cap the global worker budget to available work, with a floor of one."""
    return min(get_worker_count(), max(1, int(work_count)))


def configure_worker_subprocesses() -> None:
    """Force compiler descendants of a process-pool worker to one job."""
    os.environ[_WORKER_ENV] = "1"
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
