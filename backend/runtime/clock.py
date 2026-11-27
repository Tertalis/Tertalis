# file: backend/runtime/clock.py ; version: 1
"""
Host-owned time source (application-execution checkpoint CP-11).

now() is always an aware local datetime; monotonic() is seconds. Waits are
monotonic, re-checked at least every 60 s so wall-clock targets are
reconciled. ManualClock is the deterministic test clock: every advance bumps a
generation, so an advance that lands between a waiter's check and its wait is
never lost.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta
from threading import Condition, RLock

__all__ = ["Clock", "ManualClock", "SystemClock", "WALL_RECONCILE_SECONDS", "requireAware"]

WALL_RECONCILE_SECONDS = 60.0


def requireAware(value: datetime, name: str = "datetime") -> datetime:
    """Rejects anything but an aware datetime."""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime.")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware; naive datetimes are rejected.")
    return value


class Clock(ABC):
    """One host time source shared by its runtimes."""

    @abstractmethod
    def now(self) -> datetime:
        """Returns the current aware local datetime."""

    @abstractmethod
    def monotonic(self) -> float:
        """Returns monotonic seconds."""

    @property
    def generation(self) -> int:
        """Advance counter; constant for clocks that are not advanced manually."""
        return 0

    @abstractmethod
    def waitFor(self, condition: Condition, predicate: Callable[[], bool], *, deadline: float | None = None) -> bool:
        """
        Waits on condition (held by the caller) until predicate() holds or
        monotonic() reaches deadline. Returns predicate's final truth.
        Anyone notifying condition wakes the wait for a re-check.
        """


class SystemClock(Clock):
    """Real time: local aware wall clock and time.monotonic()."""

    def now(self) -> datetime:
        return datetime.now().astimezone()

    def monotonic(self) -> float:
        return time.monotonic()

    def waitFor(self, condition: Condition, predicate: Callable[[], bool], *, deadline: float | None = None) -> bool:
        while True:
            if predicate():
                return True
            slice_ = WALL_RECONCILE_SECONDS
            if deadline is not None:
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    return bool(predicate())
                slice_ = min(slice_, remaining)
            condition.wait(slice_)


class ManualClock(Clock):
    """
    Deterministic clock that moves only through advance() / setWall().

    Waiting conditions are registered for the duration of waitFor and are
    notified on every advance; the generation check closes the window
    between a waiter's predicate check and its wait.
    """

    def __init__(self, start: datetime, *, monotonicStart: float = 0.0) -> None:
        self._lock = RLock()
        self._wall = requireAware(start, "start")
        self._monotonic = float(monotonicStart)
        self._generation = 0
        self._waiting: dict[int, tuple[Condition, int]] = {}

    def now(self) -> datetime:
        with self._lock:
            return self._wall

    def monotonic(self) -> float:
        with self._lock:
            return self._monotonic

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def advance(self, seconds: float) -> None:
        """Moves wall and monotonic time forward together."""
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
            raise ValueError("ManualClock.advance requires non-negative seconds.")
        with self._lock:
            self._wall = self._wall + timedelta(seconds=seconds)
            self._monotonic += float(seconds)
        self._bump()

    def setWall(self, value: datetime) -> None:
        """Jumps the wall clock only (monotonic time is unchanged)."""
        requireAware(value, "value")
        with self._lock:
            self._wall = value
        self._bump()

    def _bump(self) -> None:
        with self._lock:
            self._generation += 1
            conditions = [condition for condition, _count in self._waiting.values()]
        for condition in conditions:
            with condition:
                condition.notify_all()

    def waitFor(self, condition: Condition, predicate: Callable[[], bool], *, deadline: float | None = None) -> bool:
        key = id(condition)
        with self._lock:
            registered, count = self._waiting.get(key, (condition, 0))
            self._waiting[key] = (registered, count + 1)
        try:
            while True:
                seen = self.generation
                if predicate():
                    return True
                if deadline is not None and self.monotonic() >= deadline:
                    return bool(predicate())
                if self.generation != seen:
                    continue
                condition.wait()
        finally:
            with self._lock:
                registered, count = self._waiting[key]
                if count <= 1:
                    del self._waiting[key]
                else:
                    self._waiting[key] = (registered, count - 1)
