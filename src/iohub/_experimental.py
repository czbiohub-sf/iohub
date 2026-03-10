"""Decorator for marking APIs as experimental.

Experimental APIs may change or be removed without a deprecation cycle.
Inspired by Polars' ``@unstable`` decorator and PEP 702's ``__deprecated__``
attribute convention.

Usage::

    from iohub._experimental import experimental


    @experimental
    class Slicer: ...


    @experimental(since="0.8.0")
    def some_function(): ...

Users see a single warning per process (Python's default filter) on first use::

    >>> from iohub.tile import Slicer
    >>> slicer = Slicer(data, tile_size={"y": 1024, "x": 1024})
    ExperimentalWarning: Slicer is experimental and may change without notice.

To suppress::

    import warnings
    from iohub._experimental import ExperimentalWarning

    warnings.filterwarnings("ignore", category=ExperimentalWarning)
"""

from __future__ import annotations

import functools
import warnings
from typing import TypeVar, overload

_T = TypeVar("_T")


class ExperimentalWarning(FutureWarning):
    """Warning emitted when an experimental API is used."""

    pass


@overload
def experimental(obj: _T) -> _T: ...


@overload
def experimental(
    obj: None = None,
    *,
    message: str | None = None,
    since: str | None = None,
) -> _T: ...


def experimental(
    obj=None,
    *,
    message: str | None = None,
    since: str | None = None,
):
    """Mark a class or function as experimental.

    Emits :class:`ExperimentalWarning` on first use (once per process).
    Sets ``__experimental__`` attribute for introspection.

    Parameters
    ----------
    message : str | None
        Custom warning message. Auto-generated from the object name if None.
    since : str | None
        Version when the feature was introduced as experimental.
    """

    def _decorator(obj):
        msg = message or f"{obj.__qualname__} is experimental. API may change between versions."
        if since:
            msg += f" Since version {since}."

        if isinstance(obj, type):
            return _wrap_class(obj, msg)
        elif callable(obj):
            return _wrap_callable(obj, msg)
        else:
            raise TypeError(f"@experimental can only decorate classes and callables, got {type(obj)}")

    if obj is not None:
        # Called as @experimental without parens
        return _decorator(obj)
    # Called as @experimental(...) with parens
    return _decorator


def _wrap_class(cls: type, msg: str) -> type:
    """Wrap a class to warn on instantiation."""
    cls.__experimental__ = msg

    original_init = cls.__init__

    @functools.wraps(original_init)
    def _init_wrapper(self, *args, **kwargs):
        warnings.warn(msg, ExperimentalWarning, stacklevel=2)
        return original_init(self, *args, **kwargs)

    cls.__init__ = _init_wrapper
    return cls


def _wrap_callable(fn, msg: str):
    """Wrap a callable to warn on each call."""
    fn.__experimental__ = msg

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        warnings.warn(msg, ExperimentalWarning, stacklevel=2)
        return fn(*args, **kwargs)

    wrapper.__experimental__ = msg
    return wrapper
