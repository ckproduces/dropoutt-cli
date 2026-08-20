"""What machine this is, and how much of it a scan is allowed to use.

Every sizing decision in the package reads from here rather than calling
``os.cpu_count()`` at the point of use. That matters for three reasons, and each
one was a real wrong answer before this module existed.

**A core count is not a permission.** ``os.cpu_count()`` reports the host's
processors, not the ones this process may run on. Inside a container with
``--cpus=2`` on a 96-core node it says 96, so the scan forked twelve workers onto
two cores' worth of quota and ran slower than serial while the cgroup throttled
it. CPU affinity and the cgroup quota are both consulted, and the smallest wins.

**Memory decides how wide the scan can go, not cores.** The streaming pass holds
a bounded sample per shard, so peak memory is roughly linear in worker count. On
a 96-core, 8 GB container the right answer is not 12 workers. :func:`plan` sizes
the pool against both and says which limit bound it.

**A GPU is only worth using where there is a matmul to move.** Nothing in the
structural scan is GPU work: it is JSON parsing, hashing and dictionary
bookkeeping, all of it pure Python and none of it vectorisable. The one place a
device helps is the atlas — pooling a hundred thousand documents into embeddings
and multiplying them against the map — so that is the only place
:func:`accelerator` is consulted. Reporting a GPU and then not using it for the
scan pass is the honest description of what a GPU can do for this workload.

Detection is cheap and never imports a heavy module to answer a question. torch
is *probed* rather than imported: ``find_spec`` tells us it exists, and it is
only imported when there is a matrix to put on a device.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

#: Bytes of resident memory one scan worker is assumed to want, on top of the
#: interpreter it starts with. Measured against the bounded per-shard sample
#: (see :data:`dropoutt.parallel.SAMPLE_TEXT_CHARS`) plus the benchmark indices
#: and the language model each worker loads. Deliberately generous: the cost of
#: over-estimating is one fewer worker, and the cost of under-estimating is the
#: machine swapping.
BYTES_PER_WORKER = 700 << 20

#: Never plan a pool wider than this however large the machine is. Past this the
#: pass is bound by page-cache misses and the parent's merge, so the extra
#: processes cost memory and return nothing. Raised from a flat 12 once worker
#: memory was bounded; a 128-core node with the RAM to match now uses 32.
MAX_WORKERS = 32

#: Share of *available* memory a scan will plan against. The rest is left for
#: the page cache, which is doing the reading this scan is bound by.
MEMORY_HEADROOM = 0.6


@dataclass(frozen=True)
class Machine:
    """What this process can actually use, as opposed to what is installed."""

    #: Processors the OS reports on the host.
    logical_cores: int = 1
    #: Distinct physical cores, when the platform will say. Equal to
    #: ``logical_cores`` when SMT cannot be detected.
    physical_cores: int = 1
    #: Processors this process is permitted to run on, after CPU affinity and
    #: any cgroup quota. This is the number that should size a pool.
    usable_cores: int = 1
    #: Bytes of RAM installed, or the cgroup limit when one is set. 0 = unknown.
    total_memory: int = 0
    #: Bytes free enough to allocate right now. 0 = unknown.
    available_memory: int = 0
    #: Names of the accelerators found, in the order the driver lists them.
    gpus: tuple[str, ...] = ()
    #: ``cuda``, ``rocm``, ``mps`` or ``cpu``. What the atlas would target.
    accelerator: str = "cpu"
    #: Why ``usable_cores`` is below ``logical_cores``, when it is.
    limited_by: str = ""

    @property
    def has_gpu(self) -> bool:
        return self.accelerator != "cpu"

    @property
    def smt(self) -> bool:
        return self.logical_cores > self.physical_cores

    def describe(self) -> str:
        """One line, for a report footer or a `--help` line."""
        parts = [f"{self.usable_cores} usable core{'' if self.usable_cores == 1 else 's'}"]
        if self.logical_cores != self.usable_cores:
            parts[0] += f" of {self.logical_cores}"
        if self.limited_by:
            parts[0] += f" ({self.limited_by})"
        if self.total_memory:
            parts.append(f"{self.total_memory / (1 << 30):.0f} GiB RAM")
        if self.gpus:
            head = self.gpus[0]
            extra = f" ×{len(self.gpus)}" if len(self.gpus) > 1 else ""
            parts.append(f"{head}{extra} [{self.accelerator}]")
        else:
            parts.append("no GPU")
        return " · ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "usable_cores": self.usable_cores,
            "total_memory": self.total_memory,
            "available_memory": self.available_memory,
            "gpus": list(self.gpus),
            "accelerator": self.accelerator,
            "limited_by": self.limited_by,
        }


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------


def _affinity_cores() -> int | None:
    """Processors this process is scheduled on, where the OS exposes it."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return len(getter(0))
    except OSError:  # pragma: no cover - permitted to fail, treated as unknown
        return None


def _cgroup_cores() -> tuple[int | None, bool]:
    """CPU quota from cgroup v2 then v1, in whole cores (rounded up).

    Returns ``(cores, found)`` so "no quota set" is distinguishable from "this
    is not a container". A container with ``--cpus=1.5`` gets 2, because a pool
    of one would leave a third of its allowance unused.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as fh:
            quota, _, period = fh.read().strip().partition(" ")
        if quota != "max":
            allowed = -(-int(quota) // max(int(period or 100000), 1))
            return max(1, allowed), True
        return None, True
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="ascii") as fh:
            v1_quota = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="ascii") as fh:
            v1_period = int(fh.read().strip())
        if v1_quota > 0 and v1_period > 0:
            return max(1, -(-v1_quota // v1_period)), True
    except (OSError, ValueError):
        pass
    return None, False


def _physical_cores(logical: int) -> int:
    """Cores that are not hyperthread siblings, when the platform will say.

    The scan pass is memory-bound rather than execution-unit-bound, so two
    threads on one core finish barely faster than one. Where the count cannot be
    had, the logical count is returned and the caller loses nothing but a
    slightly wider pool.
    """
    if sys.platform == "darwin":
        value = _sysctl_int("hw.physicalcpu")
        return value or logical
    if sys.platform.startswith("linux"):
        siblings: set[str] = set()
        try:
            base = "/sys/devices/system/cpu"
            for name in os.listdir(base):
                if not name.startswith("cpu") or not name[3:].isdigit():
                    continue
                path = os.path.join(base, name, "topology", "thread_siblings_list")
                try:
                    with open(path, encoding="ascii") as fh:
                        siblings.add(fh.read().strip())
                except OSError:
                    continue
        except OSError:
            return logical
        return len(siblings) or logical
    return logical


def _sysctl(*names: str) -> list[str]:
    """One ``sysctl -n`` call for several keys, in the order asked for.

    Batched because each spawn costs about ten milliseconds and this module is
    on the path of ``dropoutt --help``. A key the kernel does not know produces
    no line, so the result is padded rather than misaligned.
    """
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", *names],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return [""] * len(names)
    lines = out.stdout.splitlines()
    if len(lines) != len(names):
        return [""] * len(names)
    return [line.strip() for line in lines]


def _sysctl_int(name: str) -> int:
    try:
        return int(_sysctl(name)[0])
    except ValueError:
        return 0


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


def _as_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _darwin_inactive_pages() -> int:
    """Inactive and speculative pages, which ``sysctl`` does not expose.

    ``vm_stat`` prints them; there is no ``vm.page_inactive_count`` key. Zero on
    failure, which only costs the plan a worker.
    """
    try:
        out = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    pages = 0
    for line in out.stdout.splitlines():
        key, _, rest = line.partition(":")
        if key.strip() in ("Pages inactive", "Pages speculative"):
            pages += _as_int(rest.strip().rstrip("."))
    return pages


def _cgroup_memory() -> int:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, encoding="ascii") as fh:
                raw = fh.read().strip()
            if raw == "max":
                continue
            value = int(raw)
            # cgroup v1 spells "no limit" as a number close to 2**63, which is
            # not a memory size and must not be reported as one.
            if 0 < value < (1 << 60):
                return value
        except (OSError, ValueError):
            continue
    return 0


def _memory() -> tuple[int, int]:
    """``(total, available)`` in bytes. Zeros mean the platform did not say."""
    total = available = 0
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    if key == "MemTotal":
                        total = int(rest.split()[0]) * 1024
                    elif key == "MemAvailable":
                        available = int(rest.split()[0]) * 1024
                    if total and available:
                        break
        except (OSError, ValueError, IndexError):
            pass
        limit = _cgroup_memory()
        if limit:
            total = min(total, limit) if total else limit
            available = min(available, limit) if available else limit
    elif sys.platform == "darwin":
        # Free pages alone are not what is available. macOS keeps almost all of
        # RAM in the inactive, speculative and purgeable lists — file cache and
        # reclaimable allocations — and hands them back on demand. Reading only
        # `vm.page_free_count` reported 2 GiB free on a 48 GiB machine, which
        # planned a one-worker scan on fourteen cores.
        raw = _sysctl(
            "hw.memsize", "hw.pagesize",
            "vm.page_free_count", "vm.page_purgeable_count",
        )
        total = _as_int(raw[0])
        page = _as_int(raw[1]) or 4096
        reclaimable = _as_int(raw[2]) + _as_int(raw[3]) + _darwin_inactive_pages()
        available = reclaimable * page
    elif sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
        except Exception:
            pass
    # A machine that will not say how much memory is free is treated as having
    # half of what it has installed, which is the assumption that costs one
    # worker rather than an out-of-memory kill.
    if total and not available:
        available = total // 2
    return total, available


# --------------------------------------------------------------------------
# Accelerators
# --------------------------------------------------------------------------


def _nvidia_gpus() -> tuple[str, ...]:
    """Ask the driver, not a Python package.

    ``nvidia-smi`` is present wherever a CUDA GPU is usable and answers in
    milliseconds. Importing torch to ask the same question costs seconds and a
    gigabyte, on a code path that runs before the scan has read a byte.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if out.returncode != 0:
        return ()
    return tuple(line.strip() for line in out.stdout.splitlines() if line.strip())


def _rocm_gpus() -> tuple[str, ...]:
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if out.returncode != 0:
        return ()
    names = []
    for line in out.stdout.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 1 and parts[1]:
            names.append(parts[1])
    return tuple(names)


def _apple_gpu() -> tuple[str, ...]:
    """Apple silicon has a GPU by construction; Intel Macs are not worth it."""
    if sys.platform != "darwin":
        return ()
    if platform.machine() not in ("arm64", "aarch64"):
        return ()
    brand = _sysctl_string("machdep.cpu.brand_string") or "Apple silicon"
    return (f"{brand} GPU",)


def _sysctl_string(name: str) -> str:
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _detect_gpus() -> tuple[tuple[str, ...], str]:
    gpus = _nvidia_gpus()
    if gpus:
        return gpus, "cuda"
    gpus = _rocm_gpus()
    if gpus:
        return gpus, "rocm"
    gpus = _apple_gpu()
    if gpus:
        return gpus, "mps"
    return (), "cpu"


# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def probe() -> Machine:
    """Everything above, once per process.

    Cached because half of it shells out, and because a scan asks four different
    questions of the same machine. ``DROPOUTT_NO_GPU=1`` skips accelerator
    detection entirely, for the case where ``nvidia-smi`` exists but hangs on a
    node whose driver is wedged.
    """
    logical = os.cpu_count() or 1
    physical = _physical_cores(logical)

    usable = logical
    limited_by = ""
    affinity = _affinity_cores()
    if affinity is not None and affinity < usable:
        usable, limited_by = affinity, "cpu affinity"
    quota, _found = _cgroup_cores()
    if quota is not None and quota < usable:
        usable, limited_by = quota, "cgroup quota"

    total, available = _memory()

    gpus: tuple[str, ...] = ()
    accel = "cpu"
    if not _truthy(os.environ.get("DROPOUTT_NO_GPU")):
        gpus, accel = _detect_gpus()

    return Machine(
        logical_cores=logical,
        physical_cores=min(physical, logical),
        usable_cores=max(1, usable),
        total_memory=total,
        available_memory=available,
        gpus=gpus,
        accelerator=accel,
        limited_by=limited_by,
    )


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Sizing:
    """How wide a scan may run on this machine, and what decided it."""

    workers: int = 1
    #: Bytes the whole scan may hold in samples, across parent and workers.
    memory_budget: int = 1 << 30
    #: ``cores``, ``memory``, ``requested`` or ``ceiling``.
    bound_by: str = "cores"
    machine: Machine = field(default_factory=Machine)

    def describe(self) -> str:
        return (
            f"{self.workers} worker{'' if self.workers == 1 else 's'} "
            f"(bound by {self.bound_by}) on {self.machine.describe()}"
        )


def plan(requested: int | None = None) -> Sizing:
    """Decide the pool width and the memory the samples may use.

    An explicit ``--workers`` is obeyed exactly, because a user who names a
    number is usually working around something this function cannot see — a
    shared node, a scheduler allocation, a reproduction. Everything else is
    derived: physical cores less one for the parent, capped by what memory will
    hold and by :data:`MAX_WORKERS`.
    """
    machine = probe()

    env = os.environ.get("DROPOUTT_WORKERS")
    if requested is None and env:
        try:
            requested = int(env)
        except ValueError:
            requested = None

    budget = _memory_budget(machine)
    if requested is not None:
        return Sizing(max(1, requested), budget, "requested", machine)

    # Hyperthread siblings share an execution unit with work that is already
    # waiting on memory, so the pool is planned against physical cores. One is
    # left for the parent, which is merging shards while the workers run.
    cores = min(machine.usable_cores, machine.physical_cores or machine.usable_cores)
    by_cores = max(1, cores - 1)

    by_memory = by_cores
    if machine.available_memory:
        by_memory = max(1, int(machine.available_memory * MEMORY_HEADROOM) // BYTES_PER_WORKER)

    workers = min(by_cores, by_memory, MAX_WORKERS)
    if workers == MAX_WORKERS < min(by_cores, by_memory):
        bound = "ceiling"
    elif by_memory < by_cores:
        bound = "memory"
    else:
        bound = "cores"
    return Sizing(max(1, workers), budget, bound, machine)


def _memory_budget(machine: Machine) -> int:
    """Bytes the corpus samples may occupy, parent and workers together.

    Floored at 512 MiB so a machine that will not report its memory still gets
    a usable atlas sample, and capped at 4 GiB because past that the sample is
    larger than the statistics need — see
    :data:`dropoutt.runner.ATLAS_SAMPLE_TARGET`.
    """
    if not machine.available_memory:
        return 1 << 30
    return max(512 << 20, min(4 << 30, int(machine.available_memory * MEMORY_HEADROOM * 0.5)))


def accelerator() -> str:
    """The device the atlas should target, honouring ``DROPOUTT_DEVICE``.

    Returns ``cpu`` unless a GPU was found *and* torch is importable, because a
    device with no framework that can address it is not a device this package
    can use. torch is probed with ``find_spec`` and never imported here.
    """
    forced = (os.environ.get("DROPOUTT_DEVICE") or "").strip().lower()
    if forced:
        return forced
    machine = probe()
    if not machine.has_gpu:
        return "cpu"
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return "cpu"
    return machine.accelerator


def blas_threads() -> int:
    """Threads the numeric libraries should use for the atlas matmuls.

    Set for the parent only. Workers inherit ``1`` from :func:`limit_worker_threads`,
    because a pool of twelve processes each starting an eight-thread BLAS pool
    oversubscribes the machine by a factor of eight and runs slower than one.
    """
    machine = probe()
    return max(1, min(machine.usable_cores, machine.physical_cores or machine.usable_cores, 16))


#: The variables every numeric library in this dependency tree reads to decide
#: how many threads to start.
_THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
)


@contextmanager
def single_threaded_children():
    """Pin the numeric libraries to one thread in processes started inside this.

    Wrapped around pool creation in the *parent*, not called in the child, and
    that placement is the whole point. These libraries read the variables once,
    when they are loaded — so a child that sets them in its own entry point has
    already imported numpy and is too late, and a forked child inherited an
    initialised OpenMP runtime and was never going to read them at all. Setting
    them before the pool exists is the only moment that works for spawn, and is
    harmless for fork.

    A pool of twelve workers each starting an eight-thread OpenMP runtime
    oversubscribes the machine eightfold. The scan pass is Python, JSON parsing
    and dictionary bookkeeping — there is no arithmetic behind those threads for
    them to be doing.

    The parent's own settings are restored on exit, because the atlas pass runs
    after the pool closes and *is* one large matrix multiply that wants every
    core on the machine.
    """
    saved = {name: os.environ.get(name) for name in _THREAD_VARS}
    for name in _THREAD_VARS:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
