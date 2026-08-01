import signal
import time

import pytest

from iohub.core.interrupt import (
    clear_shutdown,
    cooperative_shutdown,
    deferred_termination,
    request_shutdown,
    shutdown_requested,
)

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="termination signals are not maskable on this platform",
)


@pytest.fixture(autouse=True)
def _reset_shutdown_state():
    clear_shutdown()
    yield
    clear_shutdown()


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    """Wait for a Python-level signal handler to run."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_sigterm_sets_flag_instead_of_terminating():
    with cooperative_shutdown() as shutdown:
        assert not shutdown.requested
        signal.raise_signal(signal.SIGTERM)
        assert _wait_for(lambda: shutdown.requested)
        assert shutdown.signal_number == signal.SIGTERM


def test_previous_handlers_are_restored():
    original = signal.getsignal(signal.SIGTERM)
    with cooperative_shutdown():
        assert signal.getsignal(signal.SIGTERM) is not original
    assert signal.getsignal(signal.SIGTERM) is original


def test_deferred_termination_delays_delivery():
    """A signal raised inside the block is queued until the block exits."""
    with cooperative_shutdown() as shutdown:
        with deferred_termination():
            signal.raise_signal(signal.SIGTERM)
            # Still blocked: a write in this position would run to completion.
            assert not shutdown.requested
        assert _wait_for(lambda: shutdown.requested)


def test_reraise_is_a_noop_without_a_signal():
    """request_shutdown stops work without claiming the process was signalled."""
    with cooperative_shutdown() as shutdown:
        request_shutdown()
        assert shutdown.requested
        assert shutdown.signal_number is None
        shutdown.reraise()
    assert shutdown_requested()


def test_shutdown_flag_is_process_wide():
    request_shutdown()
    assert shutdown_requested()
    clear_shutdown()
    assert not shutdown_requested()
