"""Cooperative shutdown on scheduler termination signals.

Slurm preemption sends ``SIGTERM``, waits for the partition's ``GraceTime``,
then sends ``SIGKILL``. A chunk or shard write interrupted between those two
signals leaves a partially written file on disk. Because a partial write of a
sharded array is repaired by a read-modify-write cycle, the next read *or*
write of that file fails deterministically, so retrying the job cannot get
past it.

The helpers here turn the grace period into useful time:

``cooperative_shutdown``
    Replaces the default "die immediately" disposition with a flag, so a
    long-running loop can finish the unit of work it is in and stop
    dispatching new ones.

``deferred_termination``
    Blocks termination signals for the duration of a block, so a write that
    has already started is not interrupted part-way through.

Neither helper can defend against ``SIGKILL``, which cannot be caught or
blocked. Callers must still be able to repair a partially written file.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

#: Signals that mean "the scheduler is reclaiming this job".
#: SIGTERM is sent by Slurm preemption and by ``scancel``; SIGUSR2 is the
#: conventional "approaching walltime" warning (``sbatch --signal=USR2@...``).
_TERMINATION_SIGNAL_NAMES = ("SIGTERM", "SIGUSR2")

_shutdown_requested = threading.Event()
_received_signal: int | None = None


def _termination_signals() -> tuple[signal.Signals, ...]:
    """Termination signals available on this platform."""
    return tuple(getattr(signal, name) for name in _TERMINATION_SIGNAL_NAMES if hasattr(signal, name))


def shutdown_requested() -> bool:
    """Whether a termination signal has been received by this process."""
    return _shutdown_requested.is_set()


def request_shutdown() -> None:
    """Request shutdown without a signal.

    Intended for tests and for callers that decide to stop for their own
    reasons. Because no signal was received, :meth:`ShutdownState.reraise`
    is a no-op afterwards.
    """
    _shutdown_requested.set()


def clear_shutdown() -> None:
    """Reset the shutdown flag. Intended for tests."""
    global _received_signal
    _shutdown_requested.clear()
    _received_signal = None


def install_worker_handlers() -> None:
    """Install flag-setting handlers in a worker process.

    Used as the ``initializer`` of a :class:`ProcessPoolExecutor`. Worker
    processes are in the same process group as the parent, so the scheduler
    signals them too; without this, the default disposition kills a worker
    mid-write.
    """
    _install(_termination_signals())


def _handle(signum: int, _frame) -> None:
    global _received_signal
    if _received_signal is None:
        _received_signal = signum
    _shutdown_requested.set()


def _install(signals: tuple[signal.Signals, ...]) -> dict[signal.Signals, object]:
    """Install flag-setting handlers, returning the handlers they replaced."""
    previous: dict[signal.Signals, object] = {}
    for sig in signals:
        try:
            previous[sig] = signal.signal(sig, _handle)
        except (OSError, ValueError):
            # ValueError: not the main thread of the main interpreter.
            # OSError: signal cannot be caught on this platform.
            continue
    return previous


@dataclass
class ShutdownState:
    """Handle on an active :func:`cooperative_shutdown` block."""

    _previous: dict[signal.Signals, object] = field(default_factory=dict)

    @property
    def requested(self) -> bool:
        """Whether a termination signal (or :func:`request_shutdown`) arrived."""
        return shutdown_requested()

    @property
    def signal_number(self) -> int | None:
        """The first termination signal received, or None if none arrived."""
        return _received_signal

    def reraise(self) -> None:
        """Exit with the disposition the received signal would have had.

        Slurm and shell conventions report a signalled process as exit code
        ``128 + signum`` (143 for SIGTERM, 140 for SIGUSR2). Workflow engines
        distinguish those from ordinary failures — Nextflow's default
        ``errorStrategy`` retries signal exits and terminates on exit 1 — so a
        job that shut down cleanly must still die *by the signal* rather than
        returning an error. Restores the original handler and re-sends the
        signal to this process.

        Does nothing if no signal was received (e.g. after
        :func:`request_shutdown`).
        """
        signum = _received_signal
        if signum is None:
            return
        sig = signal.Signals(signum)
        signal.signal(sig, self._previous.get(sig, signal.SIG_DFL))
        signal.raise_signal(sig)


@contextmanager
def cooperative_shutdown() -> Iterator[ShutdownState]:
    """Turn termination signals into a flag for the duration of the block.

    Handlers are only installed from the main thread of the main interpreter;
    elsewhere this is a no-op and :attr:`ShutdownState.requested` simply stays
    False unless :func:`request_shutdown` is called. Previous handlers are
    restored on exit, so importing library code does not leave global signal
    state behind.
    """
    state = ShutdownState()
    state._previous = _install(_termination_signals())
    try:
        yield state
    finally:
        for sig, handler in state._previous.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                continue


@contextmanager
def deferred_termination() -> Iterator[None]:
    """Block termination signals for the duration of the block.

    The kernel queues a blocked signal and delivers it once the block exits,
    so a write inside this context runs to completion even if the handler for
    the signal would otherwise terminate the process. Only affects the calling
    thread, which is what is wanted: the write happens in the main thread of a
    worker process, and Python only runs signal handlers on the main thread of
    the main interpreter anyway.

    A no-op where ``pthread_sigmask`` is unavailable (notably Windows).
    """
    mask = getattr(signal, "pthread_sigmask", None)
    signals = _termination_signals()
    if mask is None or not signals:
        yield
        return
    previous = mask(signal.SIG_BLOCK, signals)
    try:
        yield
    finally:
        mask(signal.SIG_SETMASK, previous)
