"""Shared entrypoint-based strategy resolution for plugin protocols.

Used by both compositors and blenders to resolve names to instances
via built-in dicts and ``importlib.metadata`` entrypoints.
"""

from __future__ import annotations

from importlib.metadata import entry_points

_RUNTIME_REGISTRY: dict[str, dict[str, type]] = {}


def register_strategy(
    name: str,
    cls: type,
    entrypoint_group: str,
    *,
    overwrite: bool = False,
    aliases: list[str] | None = None,
) -> None:
    """Register a strategy class at runtime.

    Parameters
    ----------
    name : str
        Primary name for the strategy.
    cls : type
        The strategy class (must be instantiable with no args).
    entrypoint_group : str
        The entrypoint group, e.g. ``"iohub.blenders"``.
    overwrite : bool
        If False (default), raises ValueError if name is already registered.
    aliases : list[str] | None
        Optional alternative names.
    """
    group = _RUNTIME_REGISTRY.setdefault(entrypoint_group, {})
    for n in [name] + (aliases or []):
        if n in group and not overwrite:
            raise ValueError(f"Strategy {n!r} already registered in {entrypoint_group}. Use overwrite=True to replace.")
        group[n] = cls


def _clear_runtime_registry(entrypoint_group: str) -> None:
    """Remove all runtime-registered strategies for a group. For testing."""
    _RUNTIME_REGISTRY.pop(entrypoint_group, None)


def resolve_strategy(
    name: str | object,
    builtins: dict[str, type],
    entrypoint_group: str,
    kind: str,
) -> object:
    """Resolve a strategy by name or pass through an existing instance.

    Parameters
    ----------
    name : str | object
        Strategy name (looked up in builtins then entrypoints)
        or an already-instantiated strategy object (returned as-is).
    builtins : dict[str, type]
        Built-in name → class mapping.
    entrypoint_group : str
        Entrypoint group to search, e.g. ``"iohub.blenders"``.
    kind : str
        Human-readable label for error messages (e.g. ``"blender"``).
    """
    if not isinstance(name, str):
        return name

    # Runtime registry (highest priority)
    runtime = _RUNTIME_REGISTRY.get(entrypoint_group, {})
    if name in runtime:
        return runtime[name]()

    # Builtins
    if name in builtins:
        return builtins[name]()

    # Entrypoints
    eps = entry_points(group=entrypoint_group)
    for ep in eps:
        if ep.name == name:
            return ep.load()()

    available = list(runtime) + list(builtins) + [ep.name for ep in eps]
    raise ValueError(f"Unknown {kind}: {name!r}. Available: {available}")
