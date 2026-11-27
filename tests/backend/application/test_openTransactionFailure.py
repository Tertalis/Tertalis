# file: tests/backend/application/test_openTransactionFailure.py ; version: 1
"""
G8: a capability that leaves a nested memory transaction open and raises
must still produce a terminal Job with an authority disposition
(application-execution checkpoint CP-8.01, CP-8.02; plan step 2).
"""
from pathlib import Path

import pytest

from backend.application.applicationRuntime import ApplicationRuntime
from backend.context import CodeEntryIdentity
from backend.io.managedIo import IoWriteError, ManagedIoTransaction
from backend.orchestration import Job, JobState
from backend.orchestration.runtime import JobAuthority, JobCleanupFailed
from backend.packs.runtime import PackResolver
from backend.registration import RegistrationScope
from backend.values import MISSING
from backend.values.committed import CommittedValueTransaction


@pytest.fixture
def harness():
    """Yields (runtime, register) where register(capabilityId, handler) publishes a capability."""
    runtime = ApplicationRuntime(appPackId="test.app", packResolver=PackResolver(roots=()))
    runtime.start()
    identity = CodeEntryIdentity(
        applicationId=runtime.applicationRun.application.applicationId,
        applicationRunId=runtime.applicationRun.applicationRunId,
        packId="test.pack",
        packVersion="1.0.0",
        codeEntryId="entry",
        codeEntryInstanceId="g8-entry",
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


def _committed(runtime, address):
    return runtime.applicationRun.application.committedState.load(address)


def _registerWriter(register):
    def write(ctx, payload):
        transaction = ctx.memory.openTransaction()
        transaction.set(payload["address"], payload["value"])
        transaction.commit()
        return "written"
    register("test.g8.write@1", write)


def test_reproducer_open_child_and_raise_ends_failed_resolved(harness):
    """The AI Buddy reproducer: open a nested transaction, stage, raise."""
    runtime, register = harness
    primary = ValueError("Unknown dependency 'w000003'.")

    def failing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/staged", "must-not-commit")
        raise primary

    register("test.g8.fail@1", failing)
    _registerWriter(register)

    job = runtime.runJob("test.g8.fail@1")

    assert job.state is JobState.FAILED
    assert job.error is primary
    assert job.authority is JobAuthority.RESOLVED
    assert job.cleanupErrors == ()
    assert job.authoritativeStateAccepted is False
    assert _committed(runtime, "test/g8/staged") is MISSING

    following = runtime.runJob("test.g8.write@1", {"address": "test/g8/next", "value": 1})
    assert following.state is JobState.SUCCEEDED
    assert following.authority is JobAuthority.RESOLVED
    assert _committed(runtime, "test/g8/next") == 1


@pytest.mark.parametrize("depth", [2, 3])
def test_nested_open_chain_is_aborted_innermost_first(harness, depth):
    """A chain of open transactions at any depth is released before the unit aborts."""
    runtime, register = harness
    aborted = []
    original = CommittedValueTransaction.abort

    def recordingAbort(self):
        aborted.append(self.transactionId)
        return original(self)

    opened = []

    def failing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        opened.append(transaction._transaction.transactionId)
        transaction.set("test/g8/depth/0", 0)
        for level in range(1, depth):
            transaction = transaction.openTransaction()
            opened.append(transaction._transaction.transactionId)
            transaction.set(f"test/g8/depth/{level}", level)
        raise RuntimeError("deep failure")

    register("test.g8.deep@1", failing)
    _registerWriter(register)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CommittedValueTransaction, "abort", recordingAbort)
        job = runtime.runJob("test.g8.deep@1")

    assert job.state is JobState.FAILED and str(job.error) == "deep failure"
    assert job.authority is JobAuthority.RESOLVED and job.cleanupErrors == ()
    assert aborted[:depth] == list(reversed(opened))
    for level in range(depth):
        assert _committed(runtime, f"test/g8/depth/{level}") is MISSING
    assert runtime.runJob("test.g8.write@1", {"address": "test/g8/after", "value": "ok"}).state is JobState.SUCCEEDED


def test_already_resolved_children_are_skipped(harness):
    """Committed and aborted children are not aborted again; only the open one is."""
    runtime, register = harness
    aborted = []
    original = CommittedValueTransaction.abort

    def recordingAbort(self):
        aborted.append(self.transactionId)
        return original(self)

    ids = {}

    def failing(ctx, _payload):
        committedChild = ctx.memory.openTransaction()
        committedChild.set("test/g8/kept-in-unit", 1)
        committedChild.commit()
        ids["committed"] = committedChild._transaction.transactionId
        abortedChild = ctx.memory.openTransaction()
        abortedChild.set("test/g8/discarded", 2)
        abortedChild.abort()
        ids["aborted"] = abortedChild._transaction.transactionId
        openChild = ctx.memory.openTransaction()
        openChild.set("test/g8/open", 3)
        ids["open"] = openChild._transaction.transactionId
        raise LookupError("after mixed children")

    register("test.g8.mixed@1", failing)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CommittedValueTransaction, "abort", recordingAbort)
        job = runtime.runJob("test.g8.mixed@1")

    assert job.state is JobState.FAILED and isinstance(job.error, LookupError)
    assert job.authority is JobAuthority.RESOLVED and job.cleanupErrors == ()
    # The capability's own abort, then cleanup of the open child, then the unit.
    assert aborted.count(ids["aborted"]) == 1
    assert ids["committed"] not in aborted
    assert aborted.count(ids["open"]) == 1
    for address in ("test/g8/kept-in-unit", "test/g8/discarded", "test/g8/open"):
        assert _committed(runtime, address) is MISSING


def test_injected_cleanup_failure_is_terminal_unresolved_with_evidence(harness):
    """A nested abort that fails leaves a terminal Job, unresolved, with all evidence."""
    runtime, register = harness
    primary = ValueError("capability failed")
    injected = OSError("injected nested abort failure")
    original = CommittedValueTransaction.abort

    def failingNestedAbort(self):
        if self._transactionParent is not None:
            raise injected
        return original(self)

    def failing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/unresolved", "staged")
        raise primary

    register("test.g8.unresolved@1", failing)
    job = Job.new()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CommittedValueTransaction, "abort", failingNestedAbort)
        with pytest.raises(JobCleanupFailed) as caught:
            runtime.runJob("test.g8.unresolved@1", job=job)

    error = caught.value
    assert error.job is job and error.primary is primary and error.__cause__ is primary
    assert job.terminal and job.state is JobState.FAILED and job.error is primary
    assert job.authority is JobAuthority.UNRESOLVED is error.authority
    assert job.authority is not JobAuthority.RESOLVED
    assert job.cleanupErrors == error.cleanupErrors and len(job.cleanupErrors) >= 2

    def flatten(errors):
        for item in errors:
            if isinstance(item, BaseExceptionGroup):
                yield from flatten(item.exceptions)
            else:
                yield item

    flat = list(flatten(job.cleanupErrors))
    assert injected in flat
    assert any(isinstance(item, RuntimeError) and "child transaction" in str(item) for item in flat)
    assert job.authoritativeStateAccepted is False
    assert _committed(runtime, "test/g8/unresolved") is MISSING


def test_io_commit_failure_after_memory_acceptance_is_partially_accepted(harness, tmp_path):
    """Memory accepted, IO commit failed: terminal Failed Job, partially-accepted."""
    runtime, register = harness
    output = tmp_path / "g8-io.json"

    def writing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/partial", "accepted")
        transaction.commit()
        ctx.io.writeJsonAtomic(output, {"saved": True})
        return "ok"

    def failingCommit(self):
        raise IoWriteError("injected IO commit failure")

    register("test.g8.partial@1", writing)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ManagedIoTransaction, "commit", failingCommit)
        job = runtime.runJob("test.g8.partial@1")

    assert job.state is JobState.FAILED and isinstance(job.error, IoWriteError)
    assert job.authoritativeStateAccepted is True
    assert job.authority is JobAuthority.PARTIALLY_ACCEPTED
    assert _committed(runtime, "test/g8/partial") == "accepted"
    assert output.exists() is False


def test_success_and_pending_cancellation_report_resolved(harness):
    """Ordinary outcomes carry the resolved disposition."""
    runtime, register = harness
    _registerWriter(register)
    done = runtime.runJob("test.g8.write@1", {"address": "test/g8/ok", "value": True})
    assert done.state is JobState.SUCCEEDED and done.authority is JobAuthority.RESOLVED

    pending = Job.new()
    assert pending.requestCancellation() is True
    returned = runtime.runJob("test.g8.write@1", {"address": "test/g8/never", "value": 0}, job=pending)
    assert returned.state is JobState.CANCELLED and returned.authority is JobAuthority.RESOLVED
    assert _committed(runtime, "test/g8/never") is MISSING
