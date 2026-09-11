"""Shared utility functions for iohub core."""

from __future__ import annotations

from iohub.core.errors import PathNormalizationError


def pad_shape(shape: tuple[int, ...], target: int = 5) -> tuple[int, ...]:
    """Pad shape tuple to a target length by prepending 1s."""
    pad = target - len(shape)
    return (1,) * pad + shape


def normalize_path(path: str) -> str:
    """Normalize a zarr path string.

    Replacement for ``zarr.storage._utils.normalize_path``.
    Handles empty strings, collapses slashes, rejects ``..``.
    """
    if not path:
        return ""
    parts = [p for p in path.replace("\\", "/").split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise PathNormalizationError(f"Path {path!r} contains '..'")
    return "/".join(parts)
