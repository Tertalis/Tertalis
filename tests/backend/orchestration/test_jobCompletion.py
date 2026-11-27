# file: tests/backend/orchestration/test_jobCompletion.py ; version: 1
"""
Job completion building blocks (application-execution checkpoint CP-10;
plan step 3, 2.01): wait, done callbacks with terminal publication vs
dispatch eligibility, deferred dispatch inside execution boundaries, and the
atomic start/cancel handoff. Races use barriers and events, never sleeps.
"""
import logging
import threading
from pathlib import Path

import pytest

from backend.application.applicationRuntime import ApplicationRuntime
from backend.context import CodeEntryIdentity
from backend.orchestration import Job, JobState
from backend.orchestration.runtime import JobCleanupFailed, jobDispatchBoundary
from backend.packs.runtime import PackResolver
from backend.registration import RegistrationScope
from backend.workspace import EphemeralWorkspace

TIMEOUT = 5


def _thread(target, *args):
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def _join(*threads):
    for thread in threads:
        thread.join(TIMEOUT)
        assert not thread.is_alive(), "worker thread did not finish"


# ---------------------------------------------------------------------------
# Standalone Jobs
# ---------------------------------------------------------------------------


def test_wait_times_out_then_returns_on_terminal_publication():
    job = Job.new()
    assert job.wait(0) is False
    assert job.tryStart() is True
    assert job.wait(0.01) is False
    waiting = threading.Event()
    results = []

    def waiter():
        waiting.set()
        results.append(job.wait(TIMEOUT))

    worker = _thread(waiter)
    assert waiting.wait(TIMEOUT)
    job.succeed("done")
    _join(worker)
    assert results == [True] and job.wait(0) is True


def test_standalone_callbacks_before_and_after_terminal_run_exactly_once():
    job = Job.new()
    calls = []
    job.addDoneCallback(lambda j: calls.append(("before", j)))
    assert calls == []
    job.tryStart()
    job.fail(ValueError("x"))
    assert calls == [("before", job)]
    job.addDoneCallback(lambda j: calls.append(("after", j)))
    assert calls == [("before", job), ("after", job)]


def test_callback_must_be_callable():
    with pytest.raises(TypeError):
        Job.new().addDoneCallback(object())


def test_exactly_once_under_concurrent_registration_and_finish():
    for _round in range(50):
        job = Job.new()
        job.tryStart()
        registrants = 8
        barrier = threading.Barrier(registrants + 1)
        counts = [0] * registrants
        lock = threading.Lock()

        def register(index):
            def callback(_job, index=index):
                with lock:
                    counts[index] += 1
            barrier.wait(TIMEOUT)
            job.addDoneCallback(callback)

        def finish():
            barrier.wait(TIMEOUT)
            job.succeed()

        workers = [_thread(register, index) for index in range(registrants)] + [_thread(finish)]
        _join(*workers)
        assert counts == [1] * registrants


def test_pending_cancellation_dispatches_on_the_cancelling_thread_after_lock_release():
    job = Job.new()
    seen = []

    # The probe thread acquires the Job lock: proof that dispatch happens after release.
    def releasingCallback(cancelled):
        seen.append(threading.get_ident())
        acquired = []

        def probe():
            ok = cancelled._lane.acquire(timeout=TIMEOUT)
            acquired.append(ok)
            if ok:
                cancelled._lane.release()

        _join(_thread(probe))
        assert acquired == [True]

    job.addDoneCallback(releasingCallback)
    cancelling = []

    def cancel():
        cancelling.append(threading.get_ident())
        assert job.requestCancellation() is True

    _join(_thread(cancel))
    assert job.state is JobState.CANCELLED
    assert seen == cancelling


def test_failing_callback_is_isolated_and_reported(caplog):
    job = Job.new()
    calls = []

    def bad(_job):
        raise RuntimeError("callback failure")

    job.addDoneCallback(bad)
    job.addDoneCallback(lambda j: calls.append(j.state))
    job.tryStart()
    with caplog.at_level(logging.ERROR, logger="tertalis.orchestration.jobs"):
        job.succeed("value")
    assert calls == [JobState.SUCCEEDED]
    assert job.state is JobState.SUCCEEDED and job.result == "value" and job.error is None
    assert any("callback" in record.getMessage() for record in caplog.records)


def test_failure_reporter_receives_callback_failure():
    job = Job.new()
    reported = []
    assert job.beginExecution(failureReporter=lambda j, cb, err: reported.append((j, err))) is True
    error = KeyError("bad")
    job.addDoneCallback(lambda _j: (_ for _ in ()).throw(error))
    job.tryStart()
    job.succeed()
    assert reported == []
    job.endExecution()
    assert reported == [(job, error)]


def test_try_start_against_concurrent_cancellation_is_atomic():
    for _round in range(200):
        job = Job.new()
        barrier = threading.Barrier(2)
        started = []

        def start():
            barrier.wait(TIMEOUT)
            started.append(job.tryStart())

        def cancel():
            barrier.wait(TIMEOUT)
            job.requestCancellation()

        _join(_thread(start), _thread(cancel))
        if started == [True]:
            assert job.state is JobState.RUNNING and job.cancellationRequested
        else:
            assert started == [False] and job.state is JobState.CANCELLED
        assert job.tryStart() is False if job.state is JobState.CANCELLED else True


def test_try_start_rejects_running_and_terminal_non_cancelled_jobs():
    job = Job.new()
    assert job.tryStart() is True
    with pytest.raises(RuntimeError):
        job.tryStart()
    job.succeed()
    with pytest.raises(RuntimeError):
        job.tryStart()


def test_owned_job_is_eligible_only_at_end_of_execution():
    job = Job.new()
    calls = []
    assert job.beginExecution() is True
    with pytest.raises(RuntimeError):
        job.beginExecution()
    job.addDoneCallback(lambda j: calls.append("early"))
    job.tryStart()
    job.succeed()
    assert job.wait(0) is True and calls == []
    job.addDoneCallback(lambda j: calls.append("between"))
    assert calls == []
    job.endExecution()
    assert calls == ["early", "between"]
    job.addDoneCallback(lambda j: calls.append("late"))
    assert calls == ["early", "between", "late"]
    with pytest.raises(RuntimeError):
        job.endExecution()


def test_begin_execution_on_terminal_job_takes_no_ownership():
    job = Job.new()
    job.requestCancellation()
    assert job.beginExecution() is False
    calls = []
    job.addDoneCallback(calls.append)
    assert calls == [job]


def test_boundary_defers_dispatch_to_outermost_exit():
    job = Job.new()
    calls = []
    job.addDoneCallback(lambda j: calls.append("done"))
    with jobDispatchBoundary():
        with jobDispatchBoundary():
            job.requestCancellation()
        assert calls == []
    assert calls == ["done"]


def test_boundary_flushes_on_exceptional_exit():
    job = Job.new()
    calls = []
    job.addDoneCallback(lambda j: calls.append("done"))
    with pytest.raises(LookupError):
        with jobDispatchBoundary():
            job.requestCancellation()
            raise LookupError("boom")
    assert calls == ["done"]


# ---------------------------------------------------------------------------
# runJob as the execution owner
# ---------------------------------------------------------------------------


@pytest.fixture
def harness():
    runtime = ApplicationRuntime(appPackId="test.app", packResolver=PackResolver(roots=()))
    runtime.start()
    identity = CodeEntryIdentity(
        applicationId=runtime.applicationRun.application.applicationId,
        applicationRunId=runtime.applicationRun.applicationRunId,
        packId="test.pack",
        packVersion="1.0.0",
        codeEntryId="entry",
        codeEntryInstanceId="completion-entry",
        sourceSha256="source-sha",
        implementationFormat="python-source@1",
        implementationId="implementation-sha",
    )
    runtime.registerCodeEntry(identity, Path.cwd())
    scopes = []

    def register(capabilityId, handler):
        scope = RegistrationScope()
        runtime.capabilities.register(scope, ownerId=identity.codeEntryInstanceId, capabilityId=capabilityId, handler=handler)
        scope.publish()
        scopes.append(scope)

    try:
        yield runtime, register
    finally:
        for scope in scopes:
            scope.withdraw()
        runtime.unregisterCodeEntry(identity.codeEntryInstanceId)
        runtime.close()


def _probeLocks(runtime, job):
    """Acquires the runtime lane and the Job lock from another thread."""
    acquired = []

    def probe():
        for lock in (runtime._lane, job._lane):
            ok = lock.acquire(timeout=TIMEOUT)
            acquired.append(ok)
            if ok:
                lock.release()

    _join(_thread(probe))
    return acquired


def test_run_job_dispatches_after_leaving_runtime_lane(harness):
    """CP-10.06: another thread acquires the runtime lane while the callback runs."""
    runtime, register = harness
    register("test.done.ok@1", lambda ctx, payload: "ok")
    job = Job.new()
    observed = []
    job.addDoneCallback(lambda j: observed.append((j.state, _probeLocks(runtime, j))))
    returned = runtime.runJob("test.done.ok@1", job=job)
    assert returned is job
    assert observed == [(JobState.SUCCEEDED, [True, True])]


def test_registration_between_terminal_publication_and_eligibility_queues(harness):
    """A callback registered after terminal publication waits for the runJob exit."""
    runtime, register = harness
    register("test.done.between@1", lambda ctx, payload: "ok")
    job = Job.new()
    reachedEnd = threading.Event()
    release = threading.Event()
    original = Job.endExecution
    calls = []

    def gatedEnd(self):
        reachedEnd.set()
        assert release.wait(TIMEOUT)
        return original(self)

    def registrant():
        assert reachedEnd.wait(TIMEOUT)
        assert job.wait(0) is True
        job.addDoneCallback(lambda j: calls.append(threading.get_ident()))
        assert calls == []
        release.set()

    results = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Job, "endExecution", gatedEnd)
        runner = _thread(lambda: results.append((threading.get_ident(), runtime.runJob("test.done.between@1", job=job))))
        other = _thread(registrant)
        _join(other, runner)
    runnerIdent, returned = results[0]
    assert returned is job and job.state is JobState.SUCCEEDED
    assert calls == [runnerIdent]


def test_dispatch_on_exceptional_run_job_exit(harness):
    """Callbacks dispatch exactly once when runJob raises after terminal publication."""
    runtime, register = harness

    def staging(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/done/cleanup", 1)
        transaction.commit()
        return "ok"

    register("test.done.cleanup@1", staging)
    job = Job.new()
    calls = []
    job.addDoneCallback(lambda j: calls.append((j.state, _probeLocks(runtime, j))))
    original = EphemeralWorkspace.close

    def failingClose(self):
        original(self)
        raise OSError("workspace")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(EphemeralWorkspace, "close", failingClose)
        with pytest.raises(JobCleanupFailed):
            runtime.runJob("test.done.cleanup@1", job=job)
    assert calls == [(JobState.FAILED, [True, True])]


def test_dispatch_on_interrupt_exit(harness):
    runtime, register = harness

    def interrupted(ctx, _payload):
        raise KeyboardInterrupt()

    register("test.done.interrupt@1", interrupted)
    job = Job.new()
    calls = []
    job.addDoneCallback(lambda j: calls.append(j.state))
    with pytest.raises(KeyboardInterrupt):
        runtime.runJob("test.done.interrupt@1", job=job)
    assert calls == [JobState.FAILED]


def test_registering_and_cancelling_inside_a_job_defers_to_boundary_exit(harness):
    """Inside an admitted Job, due callbacks never run synchronously."""
    runtime, register = harness
    finished = Job.new()
    finished.requestCancellation()
    pending = Job.new()
    events = []
    pending.addDoneCallback(lambda j: events.append(("pending", _probeLocks(runtime, j))))

    def handler(ctx, _payload):
        finished.addDoneCallback(lambda j: events.append(("finished", _probeLocks(runtime, j))))
        assert pending.requestCancellation() is True
        assert pending.state is JobState.CANCELLED
        events.append(("handler-end", None))
        return "ok"

    register("test.done.inside@1", handler)
    job = runtime.runJob("test.done.inside@1")
    assert job.state is JobState.SUCCEEDED
    assert events[0] == ("handler-end", None)
    assert sorted(events[1:]) == [("finished", [True, True]), ("pending", [True, True])]


def test_pre_cancelled_job_is_returned_and_dispatches_once(harness):
    runtime, register = harness
    register("test.done.never@1", lambda ctx, payload: pytest.fail("must not run"))
    job = Job.new()
    calls = []
    job.addDoneCallback(calls.append)
    job.requestCancellation()
    assert calls == [job]
    assert runtime.runJob("test.done.never@1", job=job) is job
    assert calls == [job] and job.state is JobState.CANCELLED


def test_cancellation_between_ownership_and_start_is_honoured(harness):
    """runJob uses tryStart: a cancel after ownership but before start is not lost."""
    runtime, register = harness
    register("test.done.raced@1", lambda ctx, payload: pytest.fail("must not run"))
    job = Job.new()
    calls = []
    job.addDoneCallback(calls.append)
    def cancelFirst(original):
        def patched(self):
            # The cancel lands after any state check, right before the start.
            self.requestCancellation()
            assert calls == []  # owned: not yet eligible
            return original(self)
        return patched

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Job, "tryStart", cancelFirst(Job.tryStart))
        patch.setattr(Job, "start", cancelFirst(Job.start))
        returned = runtime.runJob("test.done.raced@1", job=job)
    assert returned is job and job.state is JobState.CANCELLED
    assert calls == [job]


def test_refused_admission_releases_ownership(harness):
    """A request refused before admission leaves the supplied Job Pending and unowned."""
    runtime, register = harness
    register("test.done.refused@1", lambda ctx, payload: "ok")
    runtime.close()
    job = Job.new()
    with pytest.raises(Exception, match="closed|not active"):
        runtime.runJob("test.done.refused@1", job=job)
    assert job.state is JobState.PENDING
    assert job.beginExecution() is True
    job.endExecution()
