"""Shared type aliases for iohub core."""

from __future__ import annotations

from typing import Literal

from zarr.storage import StoreLike

# Zarr format / NGFF version
type ZarrFormat = Literal[2, 3]
# Plain alias (not a PEP 695 ``type``) so tools can introspect it as a runtime
# Literal — e.g. Typer reads it directly to build the CLI's version choices.
NGFFVersion = Literal["0.4", "0.5"]

# Store access mode
type AccessMode = Literal["r", "r+", "a", "w", "w-"]

# Store input -- zarr-python's StoreLike (Store, StorePath, FSMap, Path, str,
# dict[str, Buffer]) is accepted everywhere a store path is expected.
type StorePath = StoreLike
