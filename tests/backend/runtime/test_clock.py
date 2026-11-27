# file: tests/backend/runtime/test_clock.py ; version: 1
"""Host Clock (application-execution checkpoint CP-11; plan step 3, 2.02)."""
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.application.applicationRuntime import ApplicationRuntime
from backend.context import CodeEntryIdentity
from backend.orchestration import JobState
from backend.packs.runtime import PackResolver
from backend.registration import RegistrationScope
from backend.runtime.clock import Clock, ManualClock, SystemClock, requireAware

TIMEOUT = 5
START = datetime(2026, 11, 27, 18, 0, tzinfo=timezone(timedelta(hours=1)))


def _run(target):
    """Runs target in a thread so a lost notification shows up as a hang, not a deadlock."""
    results = []
    thread = threading.Thread(target=lambda: results.append(target()), daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    assert not thread.is_alive(), "wait was not woken"
    return results[0]


def test_system_clock_is_aware_and_monotonic():
    clock = SystemClock()
    now = clock.now()
    assert now.tzinfo is not None and now.utcoffset() is not None
    first = clock.monotonic()
    assert clock.monotonic() >= first


def test_naive_datetimes_are_rejected():
    naive = datetime(2026, 11, 27, 18, 0)
    with pytest.raises(ValueError):
        ManualClock(naive)
    clock = ManualClock(START)
    with pytest.raises(ValueError):
        clock.setWall(naive)
    with pytest.raises(ValueError):
        requireAware(naive)
    with pytest.raises(TypeError):
        requireAware("2026-11-27T18:00:00+01:00")


def test_manual_clock_advances_wall_and_monotonic_together():
    clock = ManualClock(START, monotonicStart=100.0)
    clock.advance(90)
    assert clock.now() == START + timedelta(seconds=90)
    assert clock.monotonic() == 190.0
    assert clock.generation == 1
    clock.setWall(START)
    assert clock.now() == START and clock.monotonic() == 190.0 and clock.generation == 2
    with pytest.raises(ValueError):
        clock.advance(-1)


def test_lost_notification_regression_advance_between_check_and_wait():
    """An advance that lands after the predicate check but before the wait is not lost."""
    clock = ManualClock(START)
    condition = threading.Condition(threading.RLock())
    checks = []

    def predicate():
        checks.append(clock.monotonic())
        if len(checks) == 1:
            # Same thread, between this check and the wait: a naive
            # implementation would notify nobody and then wait forever.
            clock.advance(10)
            return False
        return clock.monotonic() >= 10

    def wait():
        with condition:
            return clock.waitFor(condition, predicate, deadline=None)

    assert _run(wait) is True
    assert checks == [0.0, 10.0]


def test_cross_thread_advance_wakes_waiter_and_deadline_is_monotonic():
    clock = ManualClock(START)
    condition = threading.Condition(threading.RLock())
    waiting = threading.Event()

    def predicate():
        waiting.set()
        return False

    def wait():
        with condition:
            return clock.waitFor(condition, predicate, deadline=30.0)

    results = []
    thread = threading.Thread(target=lambda: results.append(wait()), daemon=True)
    thread.start()
    assert waiting.wait(TIMEOUT)
    clock.advance(10)
    clock.setWall(START + timedelta(days=1))  # wall jump alone does not reach the deadline
    clock.advance(25)
    thread.join(TIMEOUT)
    assert not thread.is_alive()
    assert results == [False]


def test_other_notifier_wakes_manual_wait():
    clock = ManualClock(START)
    condition = threading.Condition(threading.RLock())
    flag = []
    waiting = threading.Event()

    def predicate():
        waiting.set()
        return bool(flag)

    results = []

    def wait():
        with condition:
            results.append(clock.waitFor(condition, predicate))

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    assert waiting.wait(TIMEOUT)
    with condition:
        flag.append(True)
        condition.notify_all()
    thread.join(TIMEOUT)
    assert not thread.is_alive() and results == [True]


def test_system_clock_wait_respects_deadline():
    clock = SystemClock()
    condition = threading.Condition()
    with condition:
        assert clock.waitFor(condition, lambda: False, deadline=clock.monotonic() + 0.01) is False
        assert clock.waitFor(condition, lambda: True, deadline=clock.monotonic()) is True


def test_runtime_rejects_non_clock():
    with pytest.raises(TypeError):
        ApplicationRuntime(appPackId="test.app", packResolver=PackResolver(roots=()), clock=object())


def test_ctx_clock_reads_host_clock_and_expires_with_context():
    clock = ManualClock(START)
    runtime = ApplicationRuntime(appPackId="test.app", packResolver=PackResolver(roots=()), clock=clock)
    assert isinstance(runtime.clock, Clock) and runtime.clock is clock
    runtime.start()
    identity = CodeEntryIdentity(
        applicationId=runtime.applicationRun.application.applicationId,
        applicationRunId=runtime.applicationRun.applicationRunId,
        packId="test.pack",
        packVersion="1.0.0",
        codeEntryId="entry",
        codeEntryInstanceId="clock-entry",
        sourceSha256="source-sha",
        implementationFormat="python-source@1",
        implementationId="implementation-sha",
    )
    runtime.registerCodeEntry(identity, Path.cwd())
    scope = RegistrationScope()
    held = {}

    def handler(ctx, _payload):
        held["ctx"] = ctx
        return {"now": ctx.clock.now(), "monotonic": ctx.clock.monotonic()}

    runtime.capabilities.register(scope, ownerId=identity.codeEntryInstanceId, capabilityId="test.clock.read@1", handler=handler)
    scope.publish()
    try:
        clock.advance(5)
        job = runtime.runJob("test.clock.read@1")
        assert job.state is JobState.SUCCEEDED
        assert job.result == {"now": START + timedelta(seconds=5), "monotonic": 5.0}
        assert not hasattr(held["ctx"].clock, "advance")
        with pytest.raises(RuntimeError):
            held["ctx"].clock.now()
        with pytest.raises(RuntimeError):
            held["ctx"].clock.monotonic()
    finally:
        scope.withdraw()
        runtime.unregisterCodeEntry(identity.codeEntryInstanceId)
        runtime.close()


def test_runtime_host_owns_one_clock(tmp_path):
    from backend.runtime import RuntimeHost
    from backend.save import ApplicationStore

    clock = ManualClock(START)
    host = RuntimeHost(applicationStore=ApplicationStore(tmp_path / "saves"), packResolver=PackResolver(roots=()), clock=clock)
    assert host.clock is clock
    assert isinstance(RuntimeHost(applicationStore=ApplicationStore(tmp_path / "s2"), packResolver=PackResolver(roots=())).clock, SystemClock)
    with pytest.raises(TypeError):
        RuntimeHost(applicationStore=ApplicationStore(tmp_path / "s3"), packResolver=PackResolver(roots=()), clock="clock")
