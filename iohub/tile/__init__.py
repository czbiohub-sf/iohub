from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, overload

import numpy as np
import xarray as xr

from iohub._experimental import ExperimentalWarning, experimental
from iohub.tile._assembler import Assembler
from iohub.tile._blend import _blend_tiles
from iohub.tile._blenders import (
    BlendContext,
    Blender,
    DistanceBlender,
    GaussianBlender,
    UniformBlender,
    get_blender,
)
from iohub.tile._compositors import (
    CompositeContext,
    Compositor,
    FirstCompositor,
    MaxCompositor,
    MeanCompositor,
    get_compositor,
)
from iohub.tile._registry import register_strategy
from iohub.tile._resolvers import (
    LayoutResolver,
    StitchingYAMLResolver,
    TransformResolver,
)
from iohub.tile._slicer import SamplingMode, Slicer, TileSpec

if TYPE_CHECKING:
    import zarr

    from iohub.ngff.nodes import Position, Well


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Keys that Well.to_xarray() accepts but Position.to_xarray() does not.
_WELL_KWARGS = {"layout_resolver", "compositor"}

CacheMode = Literal["persist", "bfs"]


def _to_xarray(data: xr.DataArray | Position | Well, **kwargs) -> xr.DataArray:
    """Convert *data* to xr.DataArray, forwarding Well kwargs if needed."""
    if isinstance(data, xr.DataArray):
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments for xr.DataArray input: {unexpected}")
        return data
    if not hasattr(data, "to_xarray"):
        raise TypeError(f"data must be an xr.DataArray or have a .to_xarray() method, got {type(data).__name__}")
    return data.to_xarray(**kwargs)


def _apply_cache(slicer: Slicer, cache: CacheMode | None) -> None:
    """Pre-load overlap regions if cache mode requires it."""
    if cache == "persist":
        from iohub.tile._cache import _persist_overlaps

        _persist_overlaps(slicer)


def _tile_order(slicer: Slicer, cache: CacheMode | None) -> list[int]:
    """Return tile processing order — BFS for cache locality, else sequential."""
    if cache == "bfs":
        from iohub.tile._cache import _bfs_tile_order

        return _bfs_tile_order(slicer)
    return list(range(len(slicer)))


# ---------------------------------------------------------------------------
# tile_and_assemble
# ---------------------------------------------------------------------------


@overload
def tile_and_assemble(
    data: xr.DataArray,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    tile_size: dict[str, int],
    output: str | Path | zarr.Group,
    *,
    overlap: dict[str, int] | None = ...,
    weights: str | Blender = ...,
    dtype: np.dtype | None = ...,
    chunks: dict[str, int] | None = ...,
    mode: SamplingMode = ...,
    align_to_chunks: bool = ...,
    cache: CacheMode | None = ...,
) -> xr.DataArray: ...


@overload
def tile_and_assemble(
    data: Position,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    tile_size: dict[str, int],
    output: str | Path | zarr.Group,
    *,
    overlap: dict[str, int] | None = ...,
    weights: str | Blender = ...,
    dtype: np.dtype | None = ...,
    chunks: dict[str, int] | None = ...,
    mode: SamplingMode = ...,
    align_to_chunks: bool = ...,
    cache: CacheMode | None = ...,
) -> xr.DataArray: ...


@overload
def tile_and_assemble(
    data: Well,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    tile_size: dict[str, int],
    output: str | Path | zarr.Group,
    *,
    overlap: dict[str, int] | None = ...,
    weights: str | Blender = ...,
    dtype: np.dtype | None = ...,
    chunks: dict[str, int] | None = ...,
    mode: SamplingMode = ...,
    align_to_chunks: bool = ...,
    cache: CacheMode | None = ...,
    layout_resolver: LayoutResolver | None = ...,
    compositor: str | Compositor = ...,
) -> xr.DataArray: ...


@experimental
def tile_and_assemble(
    data: xr.DataArray | Position | Well,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    tile_size: dict[str, int],
    output: str | Path | zarr.Group,
    *,
    overlap: dict[str, int] | None = None,
    weights: str | Blender = "gaussian",
    dtype: np.dtype | None = None,
    chunks: dict[str, int] | None = None,
    mode: SamplingMode = SamplingMode.SQUEEZE,
    align_to_chunks: bool = False,
    cache: Literal["persist", "bfs"] | None = None,
    # Well-specific (forwarded to Well.to_xarray)
    layout_resolver: LayoutResolver | None = None,
    compositor: str | Compositor = "mean",
) -> xr.DataArray:
    """Tile a volume, apply a function to each tile, and reassemble.

    Convenience wrapper around :class:`Slicer` + :class:`Assembler`.

    Parameters
    ----------
    data : xr.DataArray | Position | Well
        Input volume. An ``xr.DataArray`` with "y" and "x" dimensions,
        or a :class:`~iohub.ngff.nodes.Position` /
        :class:`~iohub.ngff.nodes.Well` (calls ``.to_xarray()``
        automatically).
    fn : callable
        Function applied to each tile. Receives an xr.DataArray,
        returns an xr.DataArray or np.ndarray of the same shape.
    tile_size : dict[str, int]
        Tile size, e.g. ``{"y": 1024, "x": 1024}``.
    output : str | Path | zarr.Group
        Output zarr path or group.
    overlap : dict[str, int] | None
        Overlap between tiles, e.g. ``{"y": 128, "x": 128}``.
    weights : str | Blender
        Blending strategy for overlaps. Default: ``"gaussian"``.
    dtype : np.dtype | None
        Output dtype. Defaults to float32.
    chunks : dict[str, int] | None
        Output zarr chunk sizes.
    mode : SamplingMode
        Border handling. Default: SQUEEZE.
    align_to_chunks : bool
        Snap tile boundaries to chunk multiples.
    cache : ``"persist"`` | ``"bfs"`` | None
        Overlap caching strategy. ``"persist"`` pre-loads overlap
        strips via ``dask.persist()``. ``"bfs"`` reorders tile
        processing via graph BFS for cache locality. Default: None.
    layout_resolver : LayoutResolver | None
        *Well only.* Forwarded to ``Well.to_xarray()``.
    compositor : str | Compositor
        *Well only.* Forwarded to ``Well.to_xarray()``.
        Default: ``"mean"``.

    Returns
    -------
    xr.DataArray
        Reassembled result backed by the output zarr.

    Examples
    --------
    >>> result = tile_and_assemble(
    ...     data,
    ...     fn=lambda tile: tile * 2,
    ...     tile_size={"y": 1024, "x": 1024},
    ...     output="/tmp/result.zarr",
    ...     overlap={"y": 128, "x": 128},
    ... )
    """
    # Collect Well kwargs (only forward non-default values)
    well_kwargs: dict = {}
    if layout_resolver is not None:
        well_kwargs["layout_resolver"] = layout_resolver
    if compositor != "mean":
        well_kwargs["compositor"] = compositor

    xa = _to_xarray(data, **well_kwargs)

    slicer = Slicer(
        xa,
        tile_size=tile_size,
        overlap=overlap,
        mode=mode,
        align_to_chunks=align_to_chunks,
    )

    _apply_cache(slicer, cache)

    assembler = Assembler(
        slicer,
        output=output,
        weights=weights,
        dtype=dtype,
        chunks=chunks,
    )

    tile_order = _tile_order(slicer, cache)
    tiles = list(slicer)
    for idx in tile_order:
        result = fn(tiles[idx].to_xarray())
        assembler.append(tiles[idx], result)
    return assembler.get_output()


# ---------------------------------------------------------------------------
# map_tiles
# ---------------------------------------------------------------------------


@overload
def map_tiles(
    data: xr.DataArray,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    tile_size: dict[str, int],
    *,
    overlap: dict[str, int] | None = ...,
    weights: str | Blender = ...,
    mode: SamplingMode = ...,
    align_to_chunks: bool = ...,
    cache: CacheMode | None = ...,
) -> xr.DataArray: ...


@overload
def map_tiles(
    data: Position,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    tile_size: dict[str, int],
    *,
    overlap: dict[str, int] | None = ...,
    weights: str | Blender = ...,
    mode: SamplingMode = ...,
    align_to_chunks: bool = ...,
    cache: CacheMode | None = ...,
) -> xr.DataArray: ...


@overload
def map_tiles(
    data: Well,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    tile_size: dict[str, int],
    *,
    overlap: dict[str, int] | None = ...,
    weights: str | Blender = ...,
    mode: SamplingMode = ...,
    align_to_chunks: bool = ...,
    cache: CacheMode | None = ...,
    layout_resolver: LayoutResolver | None = ...,
    compositor: str | Compositor = ...,
) -> xr.DataArray: ...


@experimental
def map_tiles(
    data: xr.DataArray | Position | Well,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    tile_size: dict[str, int],
    *,
    overlap: dict[str, int] | None = None,
    weights: str | Blender = "gaussian",
    mode: SamplingMode = SamplingMode.SQUEEZE,
    align_to_chunks: bool = False,
    cache: Literal["persist", "bfs"] | None = None,
    # Well-specific (forwarded to Well.to_xarray)
    layout_resolver: LayoutResolver | None = None,
    compositor: str | Compositor = "mean",
) -> xr.DataArray:
    """Tile a volume, apply a function, and blend back — pure xarray.

    Like :func:`tile_and_assemble` but returns a lazy dask-backed
    ``xr.DataArray`` without writing to zarr. Useful for single-machine
    workflows where the result fits in memory or feeds into further
    xarray/dask pipelines.

    Parameters
    ----------
    data : xr.DataArray | Position | Well
        Input volume. An ``xr.DataArray`` with "y" and "x" dimensions,
        or a :class:`~iohub.ngff.nodes.Position` /
        :class:`~iohub.ngff.nodes.Well` (calls ``.to_xarray()``
        automatically).
    fn : callable
        Function applied to each tile. Receives an xr.DataArray,
        returns an xr.DataArray or np.ndarray of the same shape.
    tile_size : dict[str, int]
        Tile size, e.g. ``{"y": 1024, "x": 1024}``.
    overlap : dict[str, int] | None
        Overlap between tiles, e.g. ``{"y": 128, "x": 128}``.
    weights : str | Blender
        Blending strategy for overlaps. Default: ``"gaussian"``.
    mode : SamplingMode
        Border handling. Default: SQUEEZE.
    align_to_chunks : bool
        Snap tile boundaries to chunk multiples.
    cache : ``"persist"`` | ``"bfs"`` | None
        Overlap caching strategy. ``"persist"`` pre-loads overlap
        strips via ``dask.persist()``. ``"bfs"`` reorders tile
        processing via graph BFS for cache locality. Default: None.
    layout_resolver : LayoutResolver | None
        *Well only.* Forwarded to ``Well.to_xarray()``.
    compositor : str | Compositor
        *Well only.* Forwarded to ``Well.to_xarray()``.
        Default: ``"mean"``.

    Returns
    -------
    xr.DataArray
        Lazy dask-backed result with the same shape and coordinates
        as the input.

    Examples
    --------
    >>> result = map_tiles(
    ...     data,
    ...     fn=lambda tile: tile * 2,
    ...     tile_size={"y": 1024, "x": 1024},
    ...     overlap={"y": 128, "x": 128},
    ... )
    >>> result.values  # triggers computation
    """
    # Collect Well kwargs (only forward non-default values)
    well_kwargs: dict = {}
    if layout_resolver is not None:
        well_kwargs["layout_resolver"] = layout_resolver
    if compositor != "mean":
        well_kwargs["compositor"] = compositor

    xa = _to_xarray(data, **well_kwargs)

    slicer = Slicer(
        xa,
        tile_size=tile_size,
        overlap=overlap,
        mode=mode,
        align_to_chunks=align_to_chunks,
    )

    _apply_cache(slicer, cache)

    tile_specs = list(slicer)
    tile_order = _tile_order(slicer, cache)
    processed: list[xr.DataArray | None] = [None] * len(tile_specs)
    for idx in tile_order:
        tile_xa = tile_specs[idx].to_xarray()
        result = fn(tile_xa)
        if isinstance(result, np.ndarray):
            result = xr.DataArray(
                result,
                dims=tile_xa.dims,
                coords=tile_xa.coords,
            )
        processed[idx] = result

    blender = get_blender(weights)
    return _blend_tiles(processed, tile_specs, blender, slicer)


__all__ = [
    "ExperimentalWarning",
    "Assembler",
    "BlendContext",
    "Blender",
    "DistanceBlender",
    "CompositeContext",
    "Compositor",
    "FirstCompositor",
    "GaussianBlender",
    "get_blender",
    "get_compositor",
    "LayoutResolver",
    "map_tiles",
    "MaxCompositor",
    "MeanCompositor",
    "register_strategy",
    "SamplingMode",
    "Slicer",
    "StitchingYAMLResolver",
    "tile_and_assemble",
    "TileSpec",
    "TransformResolver",
    "UniformBlender",
]
