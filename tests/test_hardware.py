"""Sizing a scan against the machine it is actually allowed to use.

Every assertion here is about a wrong answer that `os.cpu_count()` gives: it
reports the host, not the quota, not the affinity mask, and not the memory.
"""

from __future__ import annotations

import os

import pytest

from dropoutt import hardware


@pytest.fixture(autouse=True)
def _fresh_probe():
    hardware.probe.cache_clear()
    yield
    hardware.probe.cache_clear()


def test_the_probe_describes_a_real_machine():
    machine = hardware.probe()
    assert machine.logical_cores >= 1
    assert 1 <= machine.usable_cores <= machine.logical_cores
    assert machine.physical_cores <= machine.logical_cores
    assert machine.accelerator in ("cpu", "cuda", "rocm", "mps")
    assert machine.describe()


def test_a_plan_never_asks_for_more_workers_than_the_machine_allows():
    sizing = hardware.plan()
    assert 1 <= sizing.workers <= hardware.MAX_WORKERS
    assert sizing.workers <= max(1, hardware.probe().usable_cores)
    assert sizing.bound_by in ("cores", "memory", "ceiling", "requested")
    assert sizing.memory_budget > 0


def test_an_explicit_worker_count_is_obeyed_exactly():
    """A user who names a number is working around something we cannot see."""
    assert hardware.plan(3).workers == 3
    assert hardware.plan(3).bound_by == "requested"
    assert hardware.plan(1).workers == 1


def test_a_small_memory_machine_gets_a_narrower_pool(monkeypatch):
    """Peak memory is roughly linear in worker count, so memory can bind first.

    Ninety-six cores and eight gigabytes is a real container shape, and the
    right answer there is not twelve workers.
    """
    big = hardware.Machine(logical_cores=96, physical_cores=96, usable_cores=96,
                           total_memory=8 << 30, available_memory=4 << 30)
    monkeypatch.setattr(hardware, "probe", lambda: big)
    sizing = hardware.plan()
    assert sizing.bound_by == "memory"
    assert sizing.workers < 95


def test_cores_bind_when_memory_is_plentiful(monkeypatch):
    roomy = hardware.Machine(logical_cores=8, physical_cores=4, usable_cores=8,
                             total_memory=64 << 30, available_memory=48 << 30)
    monkeypatch.setattr(hardware, "probe", lambda: roomy)
    sizing = hardware.plan()
    # Physical cores, less one for the parent doing the merging.
    assert sizing.workers == 3
    assert sizing.bound_by == "cores"


def test_the_environment_can_override_the_plan(monkeypatch):
    monkeypatch.setenv("DROPOUTT_WORKERS", "2")
    assert hardware.plan().workers == 2
    monkeypatch.setenv("DROPOUTT_WORKERS", "not-a-number")
    assert hardware.plan().workers >= 1


def test_gpu_detection_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("DROPOUTT_NO_GPU", "1")
    assert hardware.probe().accelerator == "cpu"
    assert hardware.probe().gpus == ()


def test_the_accelerator_is_cpu_without_a_framework_that_can_address_it(monkeypatch):
    """A device with no torch is not a device this package can use."""
    monkeypatch.delenv("DROPOUTT_DEVICE", raising=False)
    monkeypatch.setattr(
        hardware, "probe",
        lambda: hardware.Machine(gpus=("Fake GPU",), accelerator="cuda"),
    )
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert hardware.accelerator() == "cpu"


def test_children_are_pinned_to_one_thread_and_the_parent_is_restored():
    """The pin has to be set before a pool exists, and lifted after it closes.

    These libraries read the variables when they load, so setting them inside a
    worker is too late — and leaving them set would run the atlas matrix
    multiply, which happens in the parent after the pool closes, on one core.
    """
    before = {name: os.environ.get(name) for name in hardware._THREAD_VARS}
    with hardware.single_threaded_children():
        assert all(os.environ[name] == "1" for name in hardware._THREAD_VARS)
    assert {name: os.environ.get(name) for name in hardware._THREAD_VARS} == before
