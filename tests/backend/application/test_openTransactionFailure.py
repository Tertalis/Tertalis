# file: tests/backend/application/test_openTransactionFailure.py ; version: 3
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


# ---------------------------------------------------------------------------
# Review corrections G8-R1 .. G8-R4
# ---------------------------------------------------------------------------

from backend.application.applicationRuntime import ApplicationRunUnsafe  # noqa: E402
from backend.llm.executionSessions import LlmMutationScope  # noqa: E402
from backend.orchestration.runtime import InvocationCleanupFailed, OrchestrationUnit  # noqa: E402
from backend.workspace import EphemeralWorkspace  # noqa: E402


def _flat(errors):
    for item in errors:
        if isinstance(item, BaseExceptionGroup):
            yield from _flat(item.exceptions)
        else:
            yield item


def _failingWorkspaceClose(patch, failures):
    """Every workspace close really closes, then raises the next injected error."""
    original = EphemeralWorkspace.close
    raised = []

    def close(self):
        original(self)
        error = failures[len(raised)] if len(raised) < len(failures) else None
        if error is not None:
            raised.append(error)
            raise error

    patch.setattr(EphemeralWorkspace, "close", close)
    return raised


def _staging(address):
    def handler(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set(address, "staged")
        transaction.commit()
        return "returned normally"
    return handler


def _recordJobScopes(patch):
    resolved = []
    original = LlmMutationScope.resolve

    def resolve(self, accepted):
        if self.parent is None and self.job is not None and self.active:
            resolved.append((self.job.jobId, accepted))
        return original(self, accepted)

    patch.setattr(LlmMutationScope, "resolve", resolve)
    return resolved


def test_r1_cleanup_only_failure_is_unresolved_and_typed(harness):
    """Successful handler, failing invocation cleanup: never reported resolved."""
    runtime, register = harness
    register("test.g8.cleanupOnly@1", _staging("test/g8/cleanup-only"))
    injected = OSError("injected workspace close failure")
    job = Job.new()
    with pytest.MonkeyPatch.context() as patch:
        _failingWorkspaceClose(patch, [injected])
        with pytest.raises(JobCleanupFailed) as caught:
            runtime.runJob("test.g8.cleanupOnly@1", job=job)
    assert caught.value.job is job and job.state is JobState.FAILED
    assert job.authority is JobAuthority.UNRESOLVED
    assert injected in list(_flat(job.cleanupErrors))
    assert job.authoritativeStateAccepted is False
    assert _committed(runtime, "test/g8/cleanup-only") is MISSING
    assert runtime.unsafeAuthority is job


def test_r1_cleanup_only_keeps_every_error(harness):
    """Several cleanup-only failures are all kept as evidence."""
    runtime, register = harness
    register("test.g8.cleanupMany@1", _staging("test/g8/cleanup-many"))
    workspaceError = OSError("workspace")
    observationError = ValueError("observations")
    job = Job.new()
    with pytest.MonkeyPatch.context() as patch:
        _failingWorkspaceClose(patch, [workspaceError])
        from backend.observation import ObservationFacade
        originalClose = ObservationFacade.close

        def observationsClose(self):
            originalClose(self)
            raise observationError

        patch.setattr(ObservationFacade, "close", observationsClose)
        with pytest.raises(JobCleanupFailed):
            runtime.runJob("test.g8.cleanupMany@1", job=job)
    flat = list(_flat(job.cleanupErrors))
    assert workspaceError in flat and observationError in flat
    assert job.authority is JobAuthority.UNRESOLVED


def test_r1_critical_cleanup_keeps_the_other_errors(harness):
    """A critical cleanup interrupt propagates; the ordinary cleanup errors are not lost."""
    runtime, register = harness
    register("test.g8.cleanupCritical@1", _staging("test/g8/cleanup-critical"))
    interrupt = KeyboardInterrupt()
    ordinary = OSError("workspace")
    with pytest.MonkeyPatch.context() as patch:
        _failingWorkspaceClose(patch, [ordinary])
        from backend.observation import ObservationFacade
        originalClose = ObservationFacade.close

        def observationsClose(self):
            originalClose(self)
            raise interrupt

        patch.setattr(ObservationFacade, "close", observationsClose)
        with pytest.raises(KeyboardInterrupt):
            runtime.runJob("test.g8.cleanupCritical@1")
    job = runtime.lastJob
    assert job.terminal and job.state is JobState.FAILED and job.error is interrupt
    flat = list(_flat(job.cleanupErrors))
    assert interrupt in flat and ordinary in flat
    assert job.authority is JobAuthority.UNRESOLVED
    assert _committed(runtime, "test/g8/cleanup-critical") is MISSING


def test_r1_ordinary_exception_group_is_not_cleanup(harness):
    """A capability's own ExceptionGroup is an ordinary failure with resolved authority."""
    runtime, register = harness
    group = ExceptionGroup("capability", [ValueError("a"), KeyError("b")])

    def failing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/group", 1)
        raise group

    register("test.g8.group@1", failing)
    _registerWriter(register)
    job = runtime.runJob("test.g8.group@1")
    assert job.state is JobState.FAILED and job.error is group
    assert job.authority is JobAuthority.RESOLVED and job.cleanupErrors == ()
    assert runtime.unsafeAuthority is None
    assert runtime.runJob("test.g8.write@1", {"address": "test/g8/after-group", "value": 1}).state is JobState.SUCCEEDED


def _nested(register, *, catchInner):
    def inner(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/inner", "staged")
        raise ValueError("inner failed")

    def outer(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/outer", "staged")
        transaction.commit()
        if catchInner:
            try:
                ctx.capabilities.call("test.g8.inner@1")
            except ValueError:
                return "swallowed"
        ctx.capabilities.call("test.g8.inner@1")
        return "unreachable"

    register("test.g8.inner@1", inner)
    register("test.g8.outer@1", outer)


def test_r2_nested_inner_and_outer_cleanup_failures_are_all_kept(harness):
    """Inner and outer invocation cleanup failures are aggregated, not overwritten."""
    runtime, register = harness
    _nested(register, catchInner=False)
    innerError, outerError = OSError("inner workspace"), OSError("outer workspace")
    job = Job.new()
    with pytest.MonkeyPatch.context() as patch:
        raised = _failingWorkspaceClose(patch, [innerError, outerError])
        with pytest.raises(JobCleanupFailed) as caught:
            runtime.runJob("test.g8.outer@1", job=job)
    assert raised == [innerError, outerError]
    assert isinstance(caught.value.primary, ValueError) and str(caught.value.primary) == "inner failed"
    flat = list(_flat(job.cleanupErrors))
    assert innerError in flat and outerError in flat
    assert job.authority is JobAuthority.UNRESOLVED
    for address in ("test/g8/inner", "test/g8/outer"):
        assert _committed(runtime, address) is MISSING


def test_r2_caught_inner_cleanup_failure_cannot_succeed(harness):
    """A capability that swallows a nested failure with failing cleanup does not succeed."""
    runtime, register = harness
    _nested(register, catchInner=True)
    innerError = OSError("inner workspace")
    job = Job.new()
    with pytest.MonkeyPatch.context() as patch:
        _failingWorkspaceClose(patch, [innerError])
        with pytest.raises(JobCleanupFailed):
            runtime.runJob("test.g8.outer@1", job=job)
    assert job.state is JobState.FAILED and isinstance(job.error, InvocationCleanupFailed)
    assert innerError in list(_flat(job.cleanupErrors))
    assert job.authority is JobAuthority.UNRESOLVED and job.authoritativeStateAccepted is False
    assert _committed(runtime, "test/g8/outer") is MISSING


def _countingWriter(register):
    calls = []

    def write(ctx, payload):
        calls.append(payload)
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/refused", "must-not-run")
        transaction.commit()
        return "written"

    register("test.g8.counted@1", write)
    return calls


def _assertRefused(runtime, calls, culprit):
    opened = []
    with pytest.MonkeyPatch.context() as patch:
        original = OrchestrationUnit.mutation.__func__

        def mutation(cls, **kwargs):
            opened.append(kwargs)
            return original(cls, **kwargs)

        patch.setattr(OrchestrationUnit, "mutation", classmethod(mutation))
        pending = Job.new()
        with pytest.raises(ApplicationRunUnsafe) as caught:
            runtime.runJob("test.g8.counted@1", {"n": 1}, job=pending)
        assert caught.value.job is culprit
        assert pending.state is JobState.PENDING
        with pytest.raises(ApplicationRunUnsafe):
            runtime.runJob("test.g8.counted@1", {"n": 2})
        with pytest.raises(ApplicationRunUnsafe):
            runtime.invokeCapability("test.g8.counted@1", {"n": 3})
    assert calls == [] and opened == []
    assert _committed(runtime, "test/g8/refused") is MISSING
    assert runtime.unsafeAuthority is culprit


def test_r3_partial_acceptance_refuses_later_admission(harness, tmp_path):
    """After partial acceptance no later Job or top-level invocation is admitted."""
    runtime, register = harness
    calls = _countingWriter(register)
    output = tmp_path / "g8-r3.json"

    def writing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/r3", "accepted")
        transaction.commit()
        ctx.io.writeJsonAtomic(output, {"saved": True})
        return "ok"

    register("test.g8.r3@1", writing)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ManagedIoTransaction, "commit", lambda self: (_ for _ in ()).throw(IoWriteError("injected")))
        job = runtime.runJob("test.g8.r3@1")
    assert job.authority is JobAuthority.PARTIALLY_ACCEPTED
    # IO commit is restored; a clean later transaction would succeed, but must not be admitted.
    _assertRefused(runtime, calls, job)


def test_r3_unresolved_refuses_later_admission(harness):
    """After unresolved authority no later Job or top-level invocation is admitted."""
    runtime, register = harness
    calls = _countingWriter(register)
    register("test.g8.r3unresolved@1", _staging("test/g8/r3-unresolved"))
    with pytest.MonkeyPatch.context() as patch:
        _failingWorkspaceClose(patch, [OSError("workspace")])
        with pytest.raises(JobCleanupFailed) as caught:
            runtime.runJob("test.g8.r3unresolved@1")
    _assertRefused(runtime, calls, caught.value.job)


def test_r4_io_open_failure_before_unit_start(harness):
    """IO transaction creation failing leaves a Failed resolved Job and a usable run."""
    runtime, register = harness
    _registerWriter(register)
    primary = OSError("injected IO open failure")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime.io, "openTransaction", lambda: (_ for _ in ()).throw(primary))
        job = runtime.runJob("test.g8.write@1", {"address": "test/g8/io-open", "value": 1})
    assert job.state is JobState.FAILED and job.error is primary
    assert job.authority is JobAuthority.RESOLVED and job.cleanupErrors == ()
    assert runtime.runJob("test.g8.write@1", {"address": "test/g8/io-open-after", "value": 2}).state is JobState.SUCCEEDED
    assert _committed(runtime, "test/g8/io-open") is MISSING


def test_r4_unit_start_failure(harness):
    """OrchestrationUnit start failing leaves a Failed resolved Job and a usable run."""
    runtime, register = harness
    _registerWriter(register)
    primary = RuntimeError("injected unit start failure")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(OrchestrationUnit, "start", lambda self: (_ for _ in ()).throw(primary))
        job = runtime.runJob("test.g8.write@1", {"address": "test/g8/unit-start", "value": 1})
    assert job.state is JobState.FAILED and job.error is primary
    assert job.authority is JobAuthority.RESOLVED and job.cleanupErrors == ()
    assert runtime.runJob("test.g8.write@1", {"address": "test/g8/unit-start-after", "value": 2}).state is JobState.SUCCEEDED


def test_r4_independent_io_abort_and_unit_cleanup_errors(harness):
    """IO abort and unit cleanup failures are both attempted and both kept."""
    runtime, register = harness
    primary = ValueError("capability failed")

    def failing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/both", "staged")
        raise primary

    register("test.g8.both@1", failing)
    ioError = OSError("injected IO abort failure")
    nestedError = OSError("injected nested abort failure")
    original = CommittedValueTransaction.abort

    def failingNestedAbort(self):
        if self._transactionParent is not None:
            raise nestedError
        return original(self)

    job = Job.new()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ManagedIoTransaction, "abort", lambda self: (_ for _ in ()).throw(ioError))
        patch.setattr(CommittedValueTransaction, "abort", failingNestedAbort)
        with pytest.raises(JobCleanupFailed):
            runtime.runJob("test.g8.both@1", job=job)
    flat = list(_flat(job.cleanupErrors))
    assert ioError in flat and nestedError in flat
    assert any(isinstance(e, RuntimeError) and "child transaction" in str(e) for e in flat)
    assert job.error is primary and job.authority is JobAuthority.UNRESOLVED


def test_r4_interrupt_before_commit(harness):
    """An interrupt from the capability propagates; the Job is terminal and resolved."""
    runtime, register = harness
    interrupt = KeyboardInterrupt()

    def interrupted(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/interrupt-before", "staged")
        raise interrupt

    register("test.g8.interruptBefore@1", interrupted)
    _registerWriter(register)
    with pytest.MonkeyPatch.context() as patch:
        resolved = _recordJobScopes(patch)
        with pytest.raises(KeyboardInterrupt) as caught:
            runtime.runJob("test.g8.interruptBefore@1")
    job = runtime.lastJob
    assert caught.value is interrupt and job.error is interrupt
    assert job.state is JobState.FAILED and job.authority is JobAuthority.RESOLVED
    assert resolved == [(job.jobId, False)]
    assert runtime.unsafeAuthority is None
    assert _committed(runtime, "test/g8/interrupt-before") is MISSING
    assert runtime.runJob("test.g8.write@1", {"address": "test/g8/after-interrupt", "value": 1}).state is JobState.SUCCEEDED


def test_r4_interrupt_after_memory_acceptance(harness, tmp_path):
    """An interrupt from IO commit after acceptance keeps the acceptance evidence."""
    runtime, register = harness
    interrupt = KeyboardInterrupt()
    output = tmp_path / "g8-interrupt.json"

    def writing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/interrupt-after", "accepted")
        transaction.commit()
        ctx.io.writeJsonAtomic(output, {"saved": True})
        return "ok"

    def interruptedCommit(self):
        raise interrupt

    register("test.g8.interruptAfter@1", writing)
    with pytest.MonkeyPatch.context() as patch:
        resolved = _recordJobScopes(patch)
        patch.setattr(ManagedIoTransaction, "commit", interruptedCommit)
        with pytest.raises(KeyboardInterrupt) as caught:
            runtime.runJob("test.g8.interruptAfter@1")
    job = runtime.lastJob
    assert caught.value is interrupt and job.error is interrupt
    assert job.authoritativeStateAccepted is True
    assert job.authority is JobAuthority.PARTIALLY_ACCEPTED
    assert resolved == [(job.jobId, True)]
    assert runtime.unsafeAuthority is job
    assert _committed(runtime, "test/g8/interrupt-after") == "accepted"


def test_r4_ordinary_success_and_failure_scope_compatibility(harness):
    """Ordinary success resolves the LLM scope accepted, ordinary failure not."""
    runtime, register = harness
    _registerWriter(register)

    def failing(ctx, _payload):
        raise ValueError("plain")

    register("test.g8.plain@1", failing)
    with pytest.MonkeyPatch.context() as patch:
        resolved = _recordJobScopes(patch)
        done = runtime.runJob("test.g8.write@1", {"address": "test/g8/compat", "value": 1})
        failed = runtime.runJob("test.g8.plain@1")
    assert done.state is JobState.SUCCEEDED and failed.state is JobState.FAILED
    assert resolved == [(done.jobId, True), (failed.jobId, False)]
    assert runtime.lastJob is failed and runtime.unsafeAuthority is None


# ---------------------------------------------------------------------------
# Review r2 corrections G8-R3a, G8-R3b
# ---------------------------------------------------------------------------

from backend.save import ApplicationStore  # noqa: E402


def _makePartial(runtime, register, tmp_path):
    output = tmp_path / "g8-partial.json"

    def writing(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/partial-save", "accepted")
        transaction.commit()
        ctx.io.writeJsonAtomic(output, {"saved": True})
        return "ok"

    register("test.g8.partialSave@1", writing)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ManagedIoTransaction, "commit", lambda self: (_ for _ in ()).throw(IoWriteError("injected")))
        job = runtime.runJob("test.g8.partialSave@1")
    assert job.authority is JobAuthority.PARTIALLY_ACCEPTED
    return job


def _makeUnresolved(runtime, register):
    register("test.g8.unresolvedSave@1", _staging("test/g8/unresolved-save"))
    with pytest.MonkeyPatch.context() as patch:
        _failingWorkspaceClose(patch, [OSError("workspace")])
        with pytest.raises(JobCleanupFailed) as caught:
            runtime.runJob("test.g8.unresolvedSave@1")
    return caught.value.job


@pytest.mark.parametrize("kind", ["partial", "unresolved"])
def test_r3a_save_and_capture_refused_while_unsafe(harness, tmp_path, kind):
    """No new SaveBundle generation is captured, accepted or published while unsafe."""
    runtime, register = harness
    store = ApplicationStore(tmp_path / "saves")
    accepted = runtime.saveApplication(store)
    application = runtime.applicationRun.application
    before = (runtime.acceptedSaveBundle, application.saveBundleId, application.durableGeneration)
    culprit = _makePartial(runtime, register, tmp_path) if kind == "partial" else _makeUnresolved(runtime, register)
    published = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ApplicationStore, "publish", lambda self, bundle: published.append(bundle))
        patch.setattr(ApplicationStore, "createApplication", lambda self, bundle: published.append(bundle))
        with pytest.raises(ApplicationRunUnsafe) as caught:
            runtime.saveApplication(store)
        assert caught.value.job is culprit
        with pytest.raises(ApplicationRunUnsafe):
            runtime.saveApplication()
        with pytest.raises(ApplicationRunUnsafe):
            runtime.captureSaveBundle()
    assert published == []
    assert (runtime.acceptedSaveBundle, application.saveBundleId, application.durableGeneration) == before
    assert runtime.acceptedSaveBundle is accepted
    # Diagnostic reads of committed state stay possible; unsafe status is explicit.
    if kind == "partial":
        assert _committed(runtime, "test/g8/partial-save") == "accepted"
    else:
        assert _committed(runtime, "test/g8/unresolved-save") is MISSING
    assert runtime.unsafeAuthority is culprit


def test_r3b_explicit_transaction_view_is_refused(harness, tmp_path):
    """A caller-supplied transaction view does not bypass containment."""
    runtime, register = harness
    calls = _countingWriter(register)
    culprit = _makePartial(runtime, register, tmp_path)
    transaction = runtime.applicationRun.application.committedState.openTransaction()
    try:
        with pytest.raises(ApplicationRunUnsafe) as caught:
            runtime.invokeCapability("test.g8.counted@1", {"n": 1}, memoryView=transaction)
        assert caught.value.job is culprit
    finally:
        transaction.abort()
    assert calls == []
    assert _committed(runtime, "test/g8/refused") is MISSING


def test_r3b_nested_job_poisoning_refuses_further_nested_invocation(harness, tmp_path):
    """After a nested Job poisons the run, the outer Job can neither invoke nor commit."""
    runtime, register = harness
    calls = _countingWriter(register)
    output = tmp_path / "g8-nested.json"
    observed = {}

    def innerPartial(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/nested-inner", "accepted")
        transaction.commit()
        ctx.io.writeJsonAtomic(output, {"saved": True})
        return "inner"

    def outer(ctx, _payload):
        transaction = ctx.memory.openTransaction()
        transaction.set("test/g8/nested-outer", "staged")
        transaction.commit()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ManagedIoTransaction, "commit", lambda self: (_ for _ in ()).throw(IoWriteError("injected")))
            observed["inner"] = runtime.runJob("test.g8.innerPartial@1")
        try:
            ctx.capabilities.call("test.g8.counted@1", {"n": 1})
        except ApplicationRunUnsafe as err:
            observed["refused"] = err
        return "outer"

    register("test.g8.innerPartial@1", innerPartial)
    register("test.g8.outerPoisoned@1", outer)
    job = runtime.runJob("test.g8.outerPoisoned@1")

    inner = observed["inner"]
    assert inner.authority is JobAuthority.PARTIALLY_ACCEPTED and runtime.unsafeAuthority is inner
    assert observed["refused"].job is inner
    assert calls == []
    assert job.state is JobState.FAILED and isinstance(job.error, ApplicationRunUnsafe)
    assert job.authoritativeStateAccepted is False
    assert _committed(runtime, "test/g8/nested-outer") is MISSING
    assert _committed(runtime, "test/g8/refused") is MISSING


def test_r3_termination_remains_possible_while_unsafe(harness, tmp_path):
    """Containment keeps termination: the unsafe run still closes."""
    runtime, register = harness
    _makePartial(runtime, register, tmp_path)
    runtime.close()
    assert runtime.unsafeAuthority is not None
