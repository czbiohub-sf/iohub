"""Unit tests for iohub.core.registry."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import iohub.core.registry as _reg
from iohub.core.errors import ImplementationNotFoundError
from iohub.core.registry import (
    _reset,
    available_implementations,
    get_implementation,
    register_implementation,
    set_default_implementation,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry state before and after every test in this module."""
    _reset()
    yield
    _reset()


# ---------------------------------------------------------------------------
# _discover()
# ---------------------------------------------------------------------------


def test_discover_zero_plugins_sets_discovered():
    """_discovered must be True even when no entrypoint plugins are installed."""
    with patch("iohub.core.registry.entry_points", return_value=[]):
        # trigger discovery
        available_implementations()
    assert _reg._discovered is True


def test_discover_zero_plugins_loads_builtins():
    """Builtins must be loaded even when no entrypoint plugins are installed."""
    with patch("iohub.core.registry.entry_points", return_value=[]):
        impls = available_implementations()
    assert "zarr" in impls


def test_discover_idempotent():
    """Calling _discover() twice must not duplicate or clear _IMPLEMENTATIONS."""
    with patch("iohub.core.registry.entry_points", return_value=[]):
        available_implementations()
        first = dict(_reg._IMPLEMENTATIONS)
        available_implementations()
        second = dict(_reg._IMPLEMENTATIONS)
    assert first == second


def test_discover_failed_builtin_logs_and_defers_error(caplog):
    """A failing builtin import is logged; _discovered is still set to True;
    the error surfaces as ImplementationNotFoundError on the next call."""
    with patch("iohub.core.registry.entry_points", return_value=[]):
        with patch("iohub.core.registry.importlib.import_module", side_effect=ImportError("mock")):
            with caplog.at_level("ERROR", logger="iohub.core.registry"):
                available_implementations()

    assert _reg._discovered is True
    assert "zarr" not in _reg._IMPLEMENTATIONS
    with pytest.raises(ImplementationNotFoundError):
        get_implementation("zarr")


# ---------------------------------------------------------------------------
# _reset()
# ---------------------------------------------------------------------------


def test_reset_roundtrip():
    """After _reset(), get_implementation() re-discovers and returns the zarr impl."""
    get_implementation("zarr")  # populate
    _reset()
    impl = get_implementation("zarr")
    assert impl is not None


def test_reset_clears_default():
    """_reset() restores _default to 'zarr'."""
    set_default_implementation("zarr")
    _reset()
    assert _reg._default == "zarr"


# ---------------------------------------------------------------------------
# get_implementation()
# ---------------------------------------------------------------------------


def test_get_implementation_zarr_returns_instance():
    impl = get_implementation("zarr")
    from iohub.core.implementations.zarr_python import ZarrPythonImplementation

    assert isinstance(impl, ZarrPythonImplementation)


def test_get_implementation_unknown_raises():
    get_implementation("zarr")  # ensure discovered
    with pytest.raises(ImplementationNotFoundError, match="nonexistent"):
        get_implementation("nonexistent")


def test_get_implementation_default_is_zarr():
    impl = get_implementation()
    from iohub.core.implementations.zarr_python import ZarrPythonImplementation

    assert isinstance(impl, ZarrPythonImplementation)


# ---------------------------------------------------------------------------
# set_default_implementation()
# ---------------------------------------------------------------------------


def test_set_default_implementation_happy():
    register_implementation("custom", type(get_implementation("zarr")))
    set_default_implementation("custom")
    assert _reg._default == "custom"


def test_set_default_implementation_unknown_raises():
    get_implementation("zarr")  # ensure discovered
    with pytest.raises(ImplementationNotFoundError, match="nonexistent"):
        set_default_implementation("nonexistent")


# ---------------------------------------------------------------------------
# available_implementations()
# ---------------------------------------------------------------------------


def test_available_implementations_includes_zarr():
    assert "zarr" in available_implementations()
