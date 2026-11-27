# file: backend/orchestration/runtime.py ; version: 11
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Condition, RLock, local

from backend.core.runtimeIds import newRuntimeId
from backend.orchestration.cancellation import CancellationSignal
from backend.values.committed import CommittedValueLayer, CommittedValueTransaction

__all__ = [
    "InvocationCleanupFailed",
    "Job",
    "JobAuthority",
    "JobCleanupEvidence",
    "JobCleanupFailed",
    "JobState",
    "OrchestrationUnit",
    "OrchestrationUnitOutcome",
    "JobDoneCallback",
    "cleanupEvidenceScope",
    "jobDispatchBoundary",
    "recordCleanupFailures",
]

_log = logging.getLogger("tertalis.orchestration.jobs")

JobDoneCallback = Callable[["Job"], object]
CallbackFailureReporter = Callable[["Job", JobDoneCallback, Exception], object]


# ---------------------------------------------------------------------------
# Done-callback dispatch (CP-10.02, CP-10.03)
# ---------------------------------------------------------------------------

_dispatchState = local()


def _dispatchDepth() -> int:
    return getattr(_dispatchState, "depth", 0)


@contextmanager
def jobDispatchBoundary() -> Iterator[None]:
    """
    Marks an execution boundary on this thread.

    Callbacks that become due while any boundary is open on the thread are
    queued and run when the outermost boundary exits (normally or by
    exception), after the code inside has left its lanes and locks. Outside
    every boundary callbacks run immediately on the calling thread.
    """
    depth = _dispatchDepth()
    if depth == 0:
        _dispatchState.pending = []
    _dispatchState.depth = depth + 1
    try:
        yield
    finally:
        _dispatchState.depth = depth
        if depth == 0:
            pending, _dispatchState.pending = _dispatchState.pending, []
            _runCallbacks(pending)


def _scheduleCallbacks(job: Job, callbacks: list[JobDoneCallback]) -> None:
    """Runs callbacks now, or defers them to the outermost open boundary."""
    if not callbacks:
        return
    entries = [(job, callback) for callback in callbacks]
    if _dispatchDepth() > 0:
        _dispatchState.pending.extend(entries)
    else:
        _runCallbacks(entries)


def _runCallbacks(entries: list[tuple[Job, JobDoneCallback]]) -> None:
    """Runs each callback once; a failing callback never affects the Job or the others."""
    for job, callback in entries:
        try:
            callback(job)
        except Exception as error:  # noqa: BLE001 - callbacks are arbitrary notification code.
            reporter = job._failureReporter
            try:
                if reporter is None:
                    _log.error("Job %s done callback %r failed.", job.jobId, callback, exc_info=error)
                else:
                    reporter(job, callback, error)
            except Exception:  # noqa: BLE001 - reporting must not stop dispatch.
                _log.exception("Job %s done callback failure could not be reported.", job.jobId)


class JobState(StrEnum):
    """Primary observable Job lifecycle states."""

    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    TIMED_OUT = "TimedOut"
    SUPERSEDED = "Superseded"


class JobAuthority(StrEnum):
    """
    What a terminal Job proves about the authority it worked under.

    RESOLVED: nothing was accepted, or everything was fully accepted, and all
    staging was released. UNRESOLVED: cleanup failed or staging could not be
    released, so later work must not assume a clean authority. PARTIALLY_ACCEPTED:
    authoritative memory was accepted but the paired persistent I/O was not.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    PARTIALLY_ACCEPTED = "partially-accepted"


class OrchestrationUnitOutcome(StrEnum):
    """Terminal outcomes for one OrchestrationUnit."""

    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    SUPERSEDED = "Superseded"


_JOB_TERMINAL_STATES = {
    JobState.SUCCEEDED,
    JobState.FAILED,
    JobState.CANCELLED,
    JobState.TIMED_OUT,
    JobState.SUPERSEDED,
}


class JobCleanupEvidence:
    """
    Execution-owned collector of cleanup failures for one admitted Job.

    Every invocation context invalidated while the Job's scope is current
    reports its cleanup failures here, whether or not an exception is
    propagating, whether or not the capability later catches it, and however
    deeply invocations nest. Nothing is overwritten; order is report order.
    """

    __slots__ = ("_errors", "_lane")

    def __init__(self) -> None:
        self._errors: list[BaseException] = []
        self._lane = RLock()

    def record(self, errors: Iterable[BaseException]) -> None:
        """Appends cleanup failures."""
        errors = tuple(errors)
        if not all(isinstance(error, BaseException) for error in errors):
            raise TypeError("Cleanup evidence must be BaseException instances.")
        with self._lane:
            self._errors.extend(errors)

    @property
    def errors(self) -> tuple[BaseException, ...]:
        """Returns every recorded failure."""
        with self._lane:
            return tuple(self._errors)


_cleanupEvidence: ContextVar[JobCleanupEvidence | None] = ContextVar("tertalis.job.cleanupEvidence", default=None)


@contextmanager
def cleanupEvidenceScope(evidence: JobCleanupEvidence) -> Iterator[JobCleanupEvidence]:
    """Makes evidence the current Job's collector for the enclosed execution."""
    if not isinstance(evidence, JobCleanupEvidence):
        raise TypeError("evidence must be a JobCleanupEvidence.")
    token = _cleanupEvidence.set(evidence)
    try:
        yield evidence
    finally:
        _cleanupEvidence.reset(token)


def recordCleanupFailures(errors: Iterable[BaseException]) -> bool:
    """Reports cleanup failures to the current Job; returns whether one was current."""
    evidence = _cleanupEvidence.get()
    if evidence is None:
        return False
    evidence.record(errors)
    return True


class InvocationCleanupFailed(RuntimeError):
    """
    Primary Job error when the capability returned normally but invocation
    cleanup failed somewhere inside the Job (for example in a nested
    invocation whose exception the capability caught). The Job's staging is
    not accepted.
    """


class JobCleanupFailed(RuntimeError):
    """
    Raised by Job execution after the Job is terminal when cleanup failed.

    The Job is already terminal and carries the same evidence: job.error is the
    primary error (None for a cancelled Job), job.cleanupErrors the cleanup
    failures, job.authority the authority disposition.
    """

    def __init__(self, job: Job) -> None:
        self.job = job
        self.primary = job.error
        self.cleanupErrors = job.cleanupErrors
        self.authority = job.authority
        super().__init__(
            f"Job {job.jobId} ended {job.state} but cleanup failed "
            f"({len(job.cleanupErrors)} error(s)); authority {job.authority}.",
        )


@dataclass(slots=True)
class Job:
    """Observable requested-work lifecycle independent of execution mechanism."""

    jobId: str
    state: JobState = JobState.PENDING
    result: object | None = None
    error: BaseException | None = None
    authoritativeStateAccepted: bool = False
    authority: JobAuthority | None = None
    cleanupErrors: tuple[BaseException, ...] = ()
    _cancellationSignal: CancellationSignal = field(
        default_factory=CancellationSignal,
        init=False,
        repr=False,
        compare=False,
    )
    _lane: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)
    _condition: Condition | None = field(default=None, init=False, repr=False, compare=False)
    _owned: bool = field(default=False, init=False, repr=False, compare=False)
    _eligible: bool = field(default=False, init=False, repr=False, compare=False)
    _callbacks: list[JobDoneCallback] = field(default_factory=list, init=False, repr=False, compare=False)
    _failureReporter: CallbackFailureReporter | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validates one Job's externally supplied lifecycle fields."""
        self._condition = Condition(self._lane)
        self._eligible = self.state in _JOB_TERMINAL_STATES
        if type(self.jobId) is not str or not self.jobId:
            raise ValueError("Job.jobId must be a non-empty exact string.")
        if not isinstance(self.state, JobState):
            raise TypeError("Job.state must be a JobState.")
        if self.error is not None and not isinstance(self.error, BaseException):
            raise TypeError("Job.error must be a BaseException or None.")
        if type(self.authoritativeStateAccepted) is not bool:
            raise TypeError("Job.authoritativeStateAccepted must be a bool.")
        if self.authority is not None and not isinstance(self.authority, JobAuthority):
            raise TypeError("Job.authority must be a JobAuthority or None.")
        if type(self.cleanupErrors) is not tuple or not all(isinstance(e, BaseException) for e in self.cleanupErrors):
            raise TypeError("Job.cleanupErrors must be a tuple of BaseException.")

    @classmethod
    def new(cls) -> Job:
        """Creates one Pending Job with a new runtime identity."""
        return cls(jobId=newRuntimeId())

    @property
    def terminal(self) -> bool:
        """Reports whether this Job has reached its final lifecycle outcome."""
        with self._lane:
            return self.state in _JOB_TERMINAL_STATES

    @property
    def cancellationRequested(self) -> bool:
        """Reports whether cancellation has been requested for this Job."""
        return self._cancellationSignal.requested

    @property
    def cancellationSignal(self) -> CancellationSignal:
        """Returns the execution-facing signal owned by this Job."""
        return self._cancellationSignal

    def requestCancellation(self) -> bool:
        """
        Requests cancellation without conflating request with running-job outcome.

        A Pending Job can be cancelled immediately because no execution mechanism
        has started. A Running Job remains Running until its executor acknowledges
        the request and records the Cancelled terminal outcome.
        """
        due: list[JobDoneCallback] = []
        with self._lane:
            if self.state in _JOB_TERMINAL_STATES:
                return False
            requested = self._cancellationSignal.request()
            if self.state is JobState.PENDING:
                self.state = JobState.CANCELLED
                self.authority = JobAuthority.RESOLVED
                due = self._publishTerminalLocked()
        _scheduleCallbacks(self, due)
        return requested

    def start(self) -> None:
        """Starts this Job unless it was cancelled while still Pending."""
        with self._lane:
            if self.state is not JobState.PENDING:
                raise RuntimeError("Only a Pending Job can start.")
        self.tryStart()

    def tryStart(self) -> bool:
        """
        Atomically starts a Pending Job or reports that it was cancelled (CP-10.05).

        Returns True when the Job is now Running, False when it is (or has
        just become) Cancelled. Any other state is an error.
        """
        due: list[JobDoneCallback] = []
        with self._lane:
            if self.state is JobState.CANCELLED:
                return False
            if self.state is not JobState.PENDING:
                raise RuntimeError("Only a Pending Job can start.")
            if self._cancellationSignal.requested:
                self.state = JobState.CANCELLED
                self.authority = JobAuthority.RESOLVED
                due = self._publishTerminalLocked()
                started = False
            else:
                self.state = JobState.RUNNING
                started = True
        _scheduleCallbacks(self, due)
        return started

    def wait(self, timeout: float | None = None) -> bool:
        """Blocks until terminal publication; returns False on timeout."""
        with self._lane:
            return self._condition.wait_for(lambda: self.state in _JOB_TERMINAL_STATES, timeout)

    def addDoneCallback(self, callback: JobDoneCallback) -> None:
        """
        Registers callback(job) to run exactly once after dispatch eligibility.

        On an eligible terminal Job it runs now (deferred to the outermost
        boundary when registered inside one); otherwise it is queued for the
        eligibility point.
        """
        if not callable(callback):
            raise TypeError("Job done callback must be callable.")
        with self._lane:
            if not self._eligible:
                self._callbacks.append(callback)
                return
        _scheduleCallbacks(self, [callback])

    def beginExecution(self, *, failureReporter: CallbackFailureReporter | None = None) -> bool:
        """
        Marks this Job as owned by one execution; eligibility waits for endExecution().

        Returns False (and takes no ownership) for an already terminal Job.
        """
        with self._lane:
            if self._owned:
                raise RuntimeError("Job is already owned by an execution.")
            if self.state in _JOB_TERMINAL_STATES:
                return False
            self._owned = True
            if failureReporter is not None:
                self._failureReporter = failureReporter
            return True

    def endExecution(self) -> None:
        """Ends execution ownership; a terminal Job becomes dispatch-eligible now."""
        due: list[JobDoneCallback] = []
        with self._lane:
            if not self._owned:
                raise RuntimeError("Job is not owned by an execution.")
            self._owned = False
            if self.state in _JOB_TERMINAL_STATES:
                due = self._takeEligibleLocked()
        _scheduleCallbacks(self, due)

    def _publishTerminalLocked(self) -> list[JobDoneCallback]:
        """Wakes waiters; an unowned Job also becomes eligible. Caller holds the lock."""
        self._condition.notify_all()
        if self._owned:
            return []
        return self._takeEligibleLocked()

    def _takeEligibleLocked(self) -> list[JobDoneCallback]:
        self._eligible = True
        due, self._callbacks = self._callbacks, []
        return due

    def succeed(self, result: object = None) -> None:
        """Records successful terminal completion with fully resolved authority."""
        self._finish(JobState.SUCCEEDED, result=result)

    def fail(
        self,
        error: BaseException,
        *,
        authority: JobAuthority = JobAuthority.RESOLVED,
        cleanupErrors: tuple[BaseException, ...] = (),
    ) -> None:
        """Records failure with the primary error, cleanup evidence and authority."""
        if not isinstance(error, BaseException):
            raise TypeError("Job failure must carry a BaseException.")
        self._finish(JobState.FAILED, error=error, authority=authority, cleanupErrors=cleanupErrors)

    def cancel(
        self,
        *,
        authority: JobAuthority = JobAuthority.RESOLVED,
        cleanupErrors: tuple[BaseException, ...] = (),
    ) -> None:
        """Records intentional cancellation after running execution acknowledges it."""
        self._finish(JobState.CANCELLED, authority=authority, cleanupErrors=cleanupErrors)

    def _finish(
        self,
        state: JobState,
        *,
        result: object | None = None,
        error: BaseException | None = None,
        authority: JobAuthority = JobAuthority.RESOLVED,
        cleanupErrors: tuple[BaseException, ...] = (),
    ) -> None:
        """Transitions a Running Job to exactly one terminal outcome."""
        with self._lane:
            if self.state is not JobState.RUNNING:
                raise RuntimeError("Only a Running Job can become terminal.")
            if not isinstance(state, JobState):
                raise TypeError("Terminal Job state must be a JobState.")
            if state not in _JOB_TERMINAL_STATES:
                raise ValueError("Job terminal transition requires a terminal JobState.")
            if error is not None and not isinstance(error, BaseException):
                raise TypeError("Terminal Job error must be a BaseException or None.")
            if not isinstance(authority, JobAuthority):
                raise TypeError("Terminal Job authority must be a JobAuthority.")
            cleanupErrors = tuple(cleanupErrors)
            if not all(isinstance(e, BaseException) for e in cleanupErrors):
                raise TypeError("Job cleanup errors must be BaseException instances.")
            self.state = state
            self.authority = authority
            self.cleanupErrors = cleanupErrors
            self.result = result
            self.error = error
            due = self._publishTerminalLocked()
        _scheduleCallbacks(self, due)


@dataclass(slots=True)
class OrchestrationUnit:
    """
    One bounded execution-authority unit inside an ApplicationRun.

    OrchestrationUnit outcome and transaction resolution remain distinct.
    Mutation-bearing units own one outer transaction, but successful workflow
    code must explicitly accept staged mutation before the unit is marked
    Completed. Failed, Cancelled, and Superseded units discard any unresolved
    staging.
    """

    orchestrationUnitId: str
    applicationRunId: str | None = None
    started: bool = False
    outcome: OrchestrationUnitOutcome | None = None
    _transaction: CommittedValueTransaction | None = None
    _mutationResolved: bool = False

    def __post_init__(self) -> None:
        """Validates one OrchestrationUnit's identity and transaction shape."""
        if type(self.orchestrationUnitId) is not str or not self.orchestrationUnitId:
            raise ValueError("OrchestrationUnit.orchestrationUnitId must be a non-empty exact string.")
        if self.applicationRunId is not None and (
            type(self.applicationRunId) is not str or not self.applicationRunId
        ):
            raise ValueError("OrchestrationUnit.applicationRunId must be a non-empty string or None.")
        if type(self.started) is not bool:
            raise TypeError("OrchestrationUnit.started must be a bool.")
        if self.outcome is not None and not isinstance(self.outcome, OrchestrationUnitOutcome):
            raise TypeError("OrchestrationUnit.outcome must be an OrchestrationUnitOutcome or None.")
        if self._transaction is not None and not isinstance(self._transaction, CommittedValueTransaction):
            raise TypeError("OrchestrationUnit transaction must be a CommittedValueTransaction or None.")
        if self._transaction is not None and self.applicationRunId is None:
            raise ValueError("A mutation-bearing OrchestrationUnit requires applicationRunId.")

    @classmethod
    def new(cls) -> OrchestrationUnit:
        """Creates one non-mutation-bearing OrchestrationUnit."""
        return cls(orchestrationUnitId=newRuntimeId())

    @classmethod
    def mutation(
        cls,
        *,
        applicationRunId: str,
        transactionBase: CommittedValueLayer | CommittedValueTransaction,
    ) -> OrchestrationUnit:
        """Creates one mutation-bearing unit with a new outer transaction."""
        if type(applicationRunId) is not str or not applicationRunId:
            raise ValueError("applicationRunId must be a non-empty string.")
        if not isinstance(transactionBase, (CommittedValueLayer, CommittedValueTransaction)):
            raise TypeError(
                "transactionBase must be a CommittedValueLayer or CommittedValueTransaction.",
            )
        return cls(
            orchestrationUnitId=newRuntimeId(),
            applicationRunId=applicationRunId,
            _transaction=transactionBase.openTransaction(),
        )

    def start(self) -> None:
        """Starts this unit exactly once."""
        if self.started:
            raise RuntimeError("OrchestrationUnit is already started.")
        if self.outcome is not None:
            raise RuntimeError("Terminal OrchestrationUnit cannot start.")
        self.started = True

    @property
    def transactionId(self) -> str | None:
        """Returns the owned transaction identity when mutation-bearing."""
        return None if self._transaction is None else self._transaction.transactionId

    @property
    def memoryView(self) -> CommittedValueTransaction | None:
        """Returns the owned speculative transaction when mutation-bearing."""
        return self._transaction

    @property
    def mutationResolved(self) -> bool:
        """Reports whether this unit has no unresolved mutation staging."""
        return self._transaction is None or self._mutationResolved

    def commitMutation(self) -> None:
        """Accepts this unit's staged mutation into its transaction parent."""
        self._requireRunning()
        transaction = self._requireMutationTransaction()
        if self._mutationResolved:
            raise RuntimeError("OrchestrationUnit mutation is already resolved.")
        transaction.commit()
        self._mutationResolved = True

    def abortMutation(self) -> None:
        """Discards this unit's unresolved staged mutation."""
        self._requireRunning()
        transaction = self._requireMutationTransaction()
        if self._mutationResolved:
            raise RuntimeError("OrchestrationUnit mutation is already resolved.")
        transaction.abort()
        self._mutationResolved = True

    def finish(self, outcome: OrchestrationUnitOutcome) -> None:
        """Records one declared terminal outcome and resolves staging as required."""
        self._requireRunning()
        if not isinstance(outcome, OrchestrationUnitOutcome):
            raise TypeError("OrchestrationUnit outcome must be an OrchestrationUnitOutcome.")

        if outcome is OrchestrationUnitOutcome.COMPLETED:
            if self._transaction is not None and not self._mutationResolved:
                raise RuntimeError(
                    "Completed mutation-bearing OrchestrationUnit requires explicit mutation resolution.",
                )
        elif self._transaction is not None and not self._mutationResolved:
            try:
                self._transaction.abort()
            except BaseException:
                # The outcome is still recorded; mutationResolved stays False
                # so callers can see the staging was not released.
                self.outcome = outcome
                raise
            self._mutationResolved = True

        self.outcome = outcome

    def _requireMutationTransaction(self) -> CommittedValueTransaction:
        """Returns this unit's transaction or rejects non-mutation-bearing use."""
        if self._transaction is None:
            raise RuntimeError("OrchestrationUnit is not mutation-bearing.")
        return self._transaction

    def _requireNonTerminal(self) -> None:
        """Requires this unit not to have a terminal outcome."""
        if self.outcome is not None:
            raise RuntimeError("OrchestrationUnit is already terminal.")

    def _requireRunning(self) -> None:
        """Requires this unit to be started and non-terminal."""
        self._requireNonTerminal()
        if not self.started:
            raise RuntimeError("OrchestrationUnit has not started.")
