"""Entrypoint-based implementation discovery and registry."""

from __future__ import annotations

import importlib
import logging
import threading
from importlib.metadata import entry_points

from iohub.core.config import ImplementationConfig
from iohub.core.errors import ImplementationNotFoundError

_logger = logging.getLogger(__name__)
_default: str = "zarr"
_IMPLEMENTATIONS: dict[str, type] = {}
_discovered: bool = False
_lock = threading.Lock()

# Built-in implementations (always available, loaded before entrypoints)
_BUILTINS: dict[str, str] = {
    "zarr": "iohub.core.implementations.zarr_python:ZarrPythonImplementation",
}


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    with _lock:
        if _discovered:
            return
        # Load builtins first
    for name, path in _BUILTINS.items():
        module_path, cls_name = path.rsplit(":", 1)
        try:
            mod = importlib.import_module(module_path)
            _IMPLEMENTATIONS[name] = getattr(mod, cls_name)
        except Exception as e:
            _logger.error(
                f"Failed to load built-in implementation {name!r} from {path!r}. "
                f"iohub may be non-functional. Error: {e}",
                exc_info=True,
            )
    # Entrypoint plugins can override builtins
    for ep in entry_points(group="iohub.zarr-implementations"):
        try:
            _IMPLEMENTATIONS[ep.name] = ep.load()
        except Exception as err:  # noqa: BLE001 — plugin loading may fail in many ways
            _logger.warning(
                f"Failed to load plugin implementation {ep.name!r}. This plugin will be unavailable. Error: {err}",
                exc_info=True,
            )
        _discovered = True


def register_implementation(name: str, cls: type) -> None:
    """Register a custom implementation class by name."""
    _IMPLEMENTATIONS[name] = cls


def get_implementation(name: str | None = None, config: ImplementationConfig | None = None):
    """Instantiate and return an implementation by name."""
    _discover()
    name = name or _default
    if name not in _IMPLEMENTATIONS:
        raise ImplementationNotFoundError(name, list(_IMPLEMENTATIONS))
    return _IMPLEMENTATIONS[name](config=config)


def set_default_implementation(name: str) -> None:
    """Change the default implementation name."""
    global _default
    _discover()
    if name not in _IMPLEMENTATIONS:
        raise ImplementationNotFoundError(name, list(_IMPLEMENTATIONS))
    _default = name


def available_implementations() -> list[str]:
    """Return names of all registered implementations."""
    _discover()
    return list(_IMPLEMENTATIONS.keys())


def _reset() -> None:
    """Reset discovery state. For testing only."""
    global _discovered, _default
    _IMPLEMENTATIONS.clear()
    _discovered = False
    _default = "zarr"
