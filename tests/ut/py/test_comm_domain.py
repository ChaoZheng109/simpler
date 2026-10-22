# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import copy
import struct
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import simpler.worker as worker_mod
from simpler.orchestrator import _callback_run
from simpler.task_interface import CommBufferSpec, CommDomainHandle, GlobalCommDomainHandle, TaskArgs
from simpler.worker import Worker


@pytest.fixture(params=[3, 4])
def domain_worker(request):
    child = Worker(
        level=3,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=[0, 1],
        num_sub_workers=0,
        comm_profile="sim",
        global_device_ranks=[10, 20],
    )
    worker = child
    members = [1, 0]
    try:
        if request.param == 4:
            worker = Worker(level=4, num_sub_workers=0)
            node = worker.add_worker(child)
            members = [(node, 1), (node, 0)]
        worker.init()
        yield worker, members
    finally:
        worker.close()


def _request(members, **changes) -> dict[str, Any]:
    config = dict(
        name="tp",
        members=members,
        window_size=4096,
        buffers=[CommBufferSpec("payload", "uint8", 32, 64)],
        retain_after_run=True,
    )
    config.update(changes)
    return config


@pytest.mark.parametrize("sizes", [(4,), (64, 16), (80, 64), (4096, 4096, 16, 16, 16)])
def test_unaligned_buffer_sizes_are_rejected_before_allocation(domain_worker, monkeypatch, sizes):
    worker, members = domain_worker
    specs = [CommBufferSpec(f"buffer_{i}", "uint8", size, size) for i, size in enumerate(sizes)]

    def unexpected_allocation(**_kwargs):
        pytest.fail("invalid buffer layout reached the allocation backend")

    monkeypatch.setattr(worker, "_allocate_domain", unexpected_allocation)
    monkeypatch.setattr(worker, "_allocate_global_domain", unexpected_allocation)

    def build(orch, _args, _cfg):
        for allocate in (orch.allocate_domain, orch.get_or_allocate_domain):
            with pytest.raises(ValueError, match="buffer_.*nbytes.*64"):
                allocate(**_request(members, window_size=16384, buffers=specs))
        assert worker.live_domains == {}
        assert tuple(spec.nbytes for spec in specs) == sizes

    worker.run(build)


def test_unaligned_window_size_is_rejected_before_allocation(domain_worker, monkeypatch):
    worker, members = domain_worker

    def unexpected_allocation(**_kwargs):
        pytest.fail("unaligned rank stride reached the allocation backend")

    monkeypatch.setattr(worker, "_allocate_domain", unexpected_allocation)
    monkeypatch.setattr(worker, "_allocate_global_domain", unexpected_allocation)

    def build(orch, _args, _cfg):
        for allocate in (orch.allocate_domain, orch.get_or_allocate_domain):
            with pytest.raises(ValueError, match="window_size.*64"):
                allocate(**_request(members, window_size=65))
        assert worker.live_domains == {}

    worker.run(build)


def test_caller_padded_buffers_keep_exact_offsets_and_logical_sizes(domain_worker):
    worker, members = domain_worker
    specs = [
        CommBufferSpec("signal", "int32", 1, 64),
        CommBufferSpec("payload", "uint8", 80, 128),
        CommBufferSpec("tail", "int32", 1, 64),
    ]

    def build(orch, _args, _cfg):
        with pytest.raises(ValueError, match="buffers exceed window_size"):
            orch.allocate_domain(**_request(members, window_size=192, buffers=specs))
        assert worker.live_domains == {}
        with orch.allocate_domain(**_request(members, window_size=256, buffers=specs)) as domain:
            for name, offset, nbytes in (("signal", 0, 64), ("payload", 64, 128), ("tail", 192, 64)):
                assert domain.buffer_range(name) == (offset, nbytes)
                if worker.level == 3:
                    for rank in range(len(members)):
                        context = domain.local_context(rank)
                        assert context.buffers[name].base == context.local_window_base + offset
                        assert context.buffers[name].base % 64 == 0
                        assert context.actual_window_size == 256
            orch.copy_to_domain(domain, 0, b"x" * 80, buffer="payload")
            assert orch.copy_from_domain(domain, 0, 80, buffer="payload") == b"x" * 80
            assert domain._config is not None
            assert [(spec.count, spec.nbytes) for spec in domain._config.specs()] == [(1, 64), (80, 128), (1, 64)]

    worker.run(build)


def test_reuse_snapshot_data_and_second_run_release(domain_worker):
    worker, members = domain_worker
    request = _request(members)
    saved = []

    def first(orch, _args, _cfg):
        handle = orch.get_or_allocate_domain(**request)
        assert type(handle) is CommDomainHandle is GlobalCommDomainHandle
        assert orch.get_or_allocate_domain(**request) is handle
        assert orch.get_domain("tp") is handle
        assert handle.member(0).local_worker_id == 1
        assert [handle.member(rank).global_device_rank for rank in range(2)] == [20, 10]
        orch.copy_to_domain(handle, 0, b"retained", buffer="payload", offset=3)
        saved.append((handle, orch))

    worker.run(first)
    handle, orch = saved[0]
    allocation_count = worker._next_alloc_id
    assert not handle.released and not handle.freed
    assert worker.live_domains == {"tp": handle}
    for name in ("tp", "missing"):
        with pytest.raises(RuntimeError, match="callback"):
            orch.get_or_allocate_domain(**dict(request, name=name))
    # Mutating a caller-owned spec cannot rewrite the captured configuration.
    request["buffers"][0].dtype = "int8"

    def second(orch, _args, _cfg):
        with pytest.raises(ValueError, match="buffers"):
            orch.get_or_allocate_domain(**request)
        same = orch.get_or_allocate_domain(**_request(members))
        assert same is handle and worker._next_alloc_id == allocation_count
        assert orch.copy_from_domain(same, 0, 8, buffer="payload", offset=3) == b"retained"
        orch.release_domain(same)
        assert same.released and not same.freed
        with pytest.raises(KeyError):
            orch.get_domain("tp")
        with pytest.raises(RuntimeError, match="released"):
            same.member(0)

    worker.run(second)
    assert handle.freed and worker.live_domains == {}
    handle.release()


def test_configuration_conflicts_have_no_device_effect(domain_worker, monkeypatch):
    worker, members = domain_worker
    request = _request(members)
    mutations = [dict(members=list(reversed(members))), dict(window_size=8192), dict(retain_after_run=False)]
    for field, value in (
        ("name", "other"),
        ("dtype", "int8"),
        ("count", 8),
        ("nbytes", 128),
        ("load_from_host", True),
        ("store_to_host", True),
    ):
        specs = copy.deepcopy(request["buffers"])
        setattr(specs[0], field, value)
        mutations.append(dict(buffers=specs))

    def build(orch, _args, _cfg):
        handle = orch.allocate_domain(**request)
        allocated = worker._next_alloc_id
        with pytest.raises(ValueError, match="already live"):
            orch.allocate_domain(**request)
        for mutation in mutations:
            with pytest.raises(ValueError, match="configuration differs"):
                orch.get_or_allocate_domain(**dict(request, **mutation))
        child = worker if worker.level == 3 else worker._next_level_workers[0]
        with monkeypatch.context() as patch:
            patch.setitem(child._config, "global_device_ranks", [30, 40])
            with pytest.raises(ValueError, match="topology"):
                orch.get_or_allocate_domain(**request)
        assert worker._next_alloc_id == allocated
        assert orch.get_domain("tp") is handle
        handle.release()

    worker.run(build)


def test_release_rebuild_does_not_remove_replacement(domain_worker):
    worker, members = domain_worker
    saved = []

    def build(orch, _args, _cfg):
        old = orch.allocate_domain(**_request(members))
        old.release()
        new = orch.allocate_domain(**_request(members))
        assert old.domain_id != new.domain_id
        assert orch.get_domain("tp") is new
        saved.extend((old, new))

    worker.run(build)
    old, new = saved
    assert old.freed and not new.freed
    assert worker.live_domains == {"tp": new}
    worker.run(lambda orch, _args, _cfg: orch.release_domain(orch.get_domain("tp")))
    assert new.freed


def test_context_exit_and_worker_close(domain_worker):
    worker, members = domain_worker
    saved = []

    def build(orch, _args, _cfg):
        handle = orch.allocate_domain(**_request(members))
        with pytest.raises(LookupError):
            with orch.get_or_allocate_domain(**_request(members)) as same:
                assert same is handle
                raise LookupError("body failure")
        assert handle.released and not handle.freed
        saved.append(handle)
        saved.append(orch.allocate_domain(**_request(members)))

    worker.run(build)
    assert saved[0].freed and not saved[1].freed
    worker.close()
    assert saved[1].released and saved[1].freed


def test_missing_lookup(domain_worker):
    worker, members = domain_worker

    def build(orch, _args, _cfg):
        with pytest.raises(KeyError):
            orch.get_domain("missing")
        orch.allocate_domain(**_request(members)).release()

    worker.run(build)


def test_post_allocation_failure_does_not_publish_an_incomplete_handle(monkeypatch):
    worker = Worker(level=3, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_ids=[0], num_sub_workers=0)
    try:
        worker.init()

        def fail_metadata(*_args):
            raise RuntimeError("metadata assembly failed")

        def build(orch, _args, _cfg):
            with monkeypatch.context() as patch:
                patch.setattr(worker_mod, "GlobalDomainMember", fail_metadata)
                with pytest.raises(RuntimeError, match="metadata assembly failed"):
                    orch.get_or_allocate_domain(**_request([0]))
            with pytest.raises(KeyError):
                orch.get_domain("tp")

        worker.run(build)
        assert worker.live_domains == {}
        worker.run(lambda orch, _args, _cfg: orch.get_or_allocate_domain(**_request([0])).release())
    finally:
        worker.close()


def test_failed_creation_stays_cleanup_owned_without_becoming_public(monkeypatch):
    worker = Worker(level=3, device_ids=[0], num_sub_workers=0)
    worker._worker = SimpleNamespace()
    worker._lifecycle = worker_mod._Lifecycle.READY
    resources = worker_mod._RunResources()
    worker._building_run_resources = resources
    released = []
    failed = []
    first = True

    def in_memory_shms(_count, action, **kwargs):
        owner = SimpleNamespace(create=lambda size: SimpleNamespace(buf=bytearray(size)))
        result = action(owner)
        if kwargs.get("after_success"):
            kwargs["after_success"]()
        return result

    def dispatch(**kwargs):
        nonlocal first
        if kwargs["op"] == "release":
            released.append(kwargs["allocation_id"])
            return
        request = kwargs["request_shms"][0].buf
        count = worker_mod._DOMAIN_REQ_HEADER.unpack_from(request)[4]
        reply = kwargs["reply_shms"][0].buf
        worker_mod._DOMAIN_REPLY_HEADER.pack_into(reply, 0, 1, 0xC0, 0x1000, count + 1 if first else count)
        if first:
            first = False
            failed.append(worker._live_domains["tp"])
        else:
            struct.pack_into("<Q", reply, worker_mod._DOMAIN_REPLY_HEADER.size, 0x1000)

    monkeypatch.setattr(worker_mod, "_run_with_owned_shared_memory", in_memory_shms)
    monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)
    monkeypatch.setattr(worker, "_dispatch_control_domain", dispatch)
    request = dict(
        name="tp", members=[0], window_size=4096, buffers=[CommBufferSpec("x", "uint8", 4, 64)], retain_after_run=True
    )
    try:
        with _callback_run(1, worker):
            with pytest.raises(RuntimeError, match="reply buffer_count"):
                worker._comm_domain(**request, reuse=False)
            old = failed[0]
            assert worker.live_domains == {}
            with pytest.raises(KeyError):
                worker._get_domain("tp")
            assert old.released and not old.freed
            assert old in resources.pending_release_domains
            new = worker._comm_domain(**request, reuse=True)
            assert new is not old and new._config is not None
            assert new.member(0).global_device_rank == 0
            assert new.allocation_id != old.allocation_id
            worker._execute_pending_domain_releases(resources)
            assert old.freed and released == [old.allocation_id]
            assert not new.released and not new.freed
            assert worker._get_domain("tp") is new
    finally:
        # Fake native objects contain no device allocations or operating-system mappings.
        worker._worker = None
        worker._building_run_resources = None
        worker._live_domains.clear()
        worker.close()


def test_nonretained_and_failed_retained_runs_are_closed(domain_worker):
    worker, members = domain_worker
    saved = []
    worker.run(
        lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members, retain_after_run=False)))
    )
    assert saved[0].released and saved[0].freed

    def failing(orch, _args, _cfg):
        saved.append(orch.get_or_allocate_domain(**_request(members)))
        raise LookupError("orchestration failed")

    with pytest.raises(LookupError, match="orchestration failed"):
        worker.run(failing)
    assert saved[1].released and saved[1].freed
    assert worker.live_domains == {}


def test_legacy_entrypoints_share_reuse_and_release(domain_worker):
    worker, members = domain_worker

    def noop(_orch, _args, _cfg):
        pass

    task = worker.register(noop) if worker.level >= 4 else None

    def build(orch, _args, _cfg):
        request = _request(members)
        if worker.level == 3:
            with pytest.raises(RuntimeError, match="L4"):
                orch.allocate_global_domain(**request)
            request["workers"] = request.pop("members")
            domain = orch.allocate_domain(**request)
        else:
            assert task is not None
            orch.submit_next_level(task, TaskArgs(), worker=members[0][0])
            with pytest.raises(TypeError, match="L3"):
                orch.allocate_domain(**dict(request, workers=request["members"], members=None))
            domain = orch.allocate_global_domain(**request)
            orch.copy_to_global_domain(domain, 0, b"legacy", buffer="payload")
            assert orch.copy_from_domain(domain, 0, 6, buffer="payload") == b"legacy"
        assert orch.get_or_allocate_domain(**_request(members)) is domain
        with pytest.raises(ValueError, match="already live"):
            orch.allocate_domain(**_request(members))
        orch.release_domain(domain)

    worker.run(build)


@pytest.mark.parametrize("release_api", ["handle", "orchestrator"])
def test_retained_domain_release_after_run(domain_worker, release_api, monkeypatch):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    release_calls = []
    backend_name = "_release_domain_claimed" if worker.level == 3 else "_release_global_domain_claimed"
    backend = getattr(worker, backend_name)

    def release_backend(handle):
        release_calls.append(handle)
        backend(handle)

    monkeypatch.setattr(worker, backend_name, release_backend)
    orch = worker._orch
    assert orch is not None
    release = domain.release if release_api == "handle" else lambda: orch.release_domain(domain)
    release()
    assert domain.released and domain.freed and worker.live_domains == {}
    domain.release()
    orch.release_domain(domain)
    orch.release_global_domain(domain)
    assert release_calls == [domain]


def test_retained_domain_serial_access_and_run_registration(domain_worker):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    orch = worker._orch
    assert orch is not None
    assert orch.get_domain("tp") is domain
    assert domain.member(0).local_worker_id == 1
    assert domain.buffer_range("payload") == (0, 64)
    if worker.level == 3:
        assert domain.local_context(0) is domain[1]
    orch.copy_to_domain(domain, 0, b"serial", buffer="payload")
    assert orch.copy_from_domain(domain, 0, 6, buffer="payload") == b"serial"
    if worker.level >= 4:
        orch.copy_to_global_domain(domain, 0, b"legacy", buffer="payload")
        assert orch.copy_from_global_domain(domain, 0, 6, buffer="payload") == b"legacy"
    assert not domain.released and not domain.freed

    def lookup(orch, _args, _cfg):
        current = orch.get_domain("tp")
        assert current is domain
        assert current.buffer_range("payload") == (0, 64)
        resources = worker._building_run_resources
        assert resources is not None and resources.requires_ordered_cleanup
        table = resources.live_domains if worker.level == 3 else resources.live_global_domains
        assert table.get("tp") is current

    worker.run(lookup)
    with domain as current:
        assert current is domain
    assert domain.freed


def test_accepted_callback_can_release_while_close_drains(domain_worker):
    worker, members = domain_worker
    saved = []
    request = _request(members, buffers=[CommBufferSpec(name, "uint8", 4, 64) for name in ("payload", "tail")])

    def allocate(orch, _args, _cfg):
        domain = orch.allocate_domain(**request)
        orch.copy_to_domain(domain, 0, b"abcdefgh", offset=60)
        saved.append(domain)

    worker.run(allocate)
    entered, resume = threading.Event(), threading.Event()
    errors = []

    def build(orch, _args, _cfg):
        entered.set()
        assert resume.wait(10)
        domain = orch.get_or_allocate_domain(**request)
        assert orch.get_domain("tp") is domain
        assert domain.member(0).local_worker_id == 1
        with pytest.raises(RuntimeError, match="READY"):
            with worker._operation_lease("fresh operation"):
                pass
        orch.copy_to_domain(domain, 0, b"copy", offset=62)
        assert orch.copy_from_domain(domain, 0, 8, offset=60) == b"abcopygh"
        domain.release()
        assert domain.released and not domain.freed

    def run():
        try:
            worker.run(build)
        except BaseException as exc:
            errors.append(exc)

    def resume_after_close_claim():
        try:
            with worker._hierarchical_start_cv:
                assert worker._hierarchical_start_cv.wait_for(
                    lambda: worker._lifecycle is worker_mod._Lifecycle.CLOSED, timeout=10
                )
            with pytest.raises(RuntimeError, match="READY"):
                worker.create_buffer(1)
            assert worker._orch is not None
            with _callback_run(0, object()), pytest.raises(RuntimeError, match="callback"):
                worker._orch.get_or_allocate_domain(**request)
        except BaseException as exc:
            errors.append(exc)
        finally:
            resume.set()

    runner = threading.Thread(target=run)
    observer = threading.Thread(target=resume_after_close_claim)
    runner.start()
    assert entered.wait(10)
    observer.start()
    try:
        # Native teardown belongs to the init-owner thread; close waits for the accepted callback.
        worker.close()
    finally:
        resume.set()
        runner.join(10)
        observer.join(10)
    assert not runner.is_alive() and not observer.is_alive()
    assert not errors
    assert saved[0].freed
