"""Shared type aliases for iohub core."""

from __future__ import annotations

from typing import Literal, TypeAlias

from zarr.storage import StoreLike

# Zarr format / NGFF version
ZarrFormat: TypeAlias = Literal[2, 3]
NGFFVersion: TypeAlias = Literal["0.4", "0.5"]

# Store access mode
AccessMode: TypeAlias = Literal["r", "r+", "a", "w", "w-"]

# Store input -- zarr-python's StoreLike (Store, StorePath, FSMap, Path, str,
# dict[str, Buffer]) is accepted everywhere a store path is expected.
StorePath: TypeAlias = StoreLike
