from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, overload

import numpy as np
import xarray as xr
import zarr

from iohub._experimental import experimental
from iohub.tile._blend import _blend_tiles
from iohub.tile._blenders import (
    Blender,
    get_blender,
)
from iohub.tile._compositors import (
    Compositor,
)
from iohub.tile._resolvers import (
    LayoutResolver,
)
from iohub.tile._tiler import SamplingMode, Tile, Tiler

if TYPE_CHECKING:
    from iohub.ngff.nodes import Position, Well

logger = logging.getLogger(__name__)

CacheMode = Literal["persist", "bfs"]

# Fixed well path in the temp HCS store
_WELL_ROW = "A"
_WELL_COL = "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_xarray(data: Position | Well, **kwargs) -> xr.DataArray:
    """Convert Position or Well to xr.DataArray."""
    return data.to_xarray(**kwargs)


def _tile_position_path(store: str | Path, tile_id: int) -> Path:
    """Return path to a tile's standalone FOV zarr."""
    return Path(store) / _WELL_ROW / _WELL_COL / f"tile_{tile_id}"


def _read_store_meta(store: str | Path) -> dict:
    """Read iohub store-level metadata from temp tile store."""
    root = zarr.open_group(str(store), mode="r")
    meta = dict(root.attrs.get("iohub", {}))
    for required in ("tile_size", "n_tiles", "tile_dims"):
        if required not in meta:
            raise ValueError(
                f"Temp tile store at {store} is missing required metadata key '{required}'. "
                f"Was create_tile_store called for this store?"
            )
    return meta


def _pixel_spacing(coords: np.ndarray) -> float:
    """Infer pixel spacing from a physical coordinate array."""
    if len(coords) > 1:
        return float(coords[1] - coords[0])
    return 1.0


# ---------------------------------------------------------------------------
# Phase 1: create_tile_store
# ---------------------------------------------------------------------------


@experimental
def create_tile_store(
    data: Position,
    tile_size: dict[str, int],
    store: str | Path,
    *,
    overlap: dict[str, int] | None = None,
    tile_batch_size: int = 16,
    mode: SamplingMode = SamplingMode.SQUEEZE,
) -> list[list[int]]:
    """Set up a temp HCS store for tiled processing. Returns batched tile IDs.

    Creates the store directory structure and writes store-level metadata.
    Each tile result will later be written as a standalone OME-Zarr FOV at
    ``store/A/1/tile_{tile_id}/``.

    Parameters
    ----------
    data : Position
        Source OME-Zarr Position to tile.
    tile_size : dict[str, int]
        Tile size per dimension, e.g. ``{"z": 32, "y": 256, "x": 256}``.
    store : str | Path
        Path for the temp tile store. Must not already exist.
    overlap : dict[str, int] | None
        Overlap between tiles, e.g. ``{"z": 16, "y": 32, "x": 32}``.
    tile_batch_size : int
        Number of tiles per batch (one batch = one SLURM job). Default: 16.
    mode : SamplingMode
        Border handling strategy. Default: SQUEEZE.

    Returns
    -------
    list[list[int]]
        Batched tile IDs, e.g. ``[[0,1,...,15], [16,...,31], ...]``.
        Pass each batch to :func:`process_tiles`.
    """
    overlap = overlap or {}
    store = Path(store)

    if store.exists():
        raise FileExistsError(f"Temp tile store already exists: {store}")

    xa = data.to_xarray()
    tiler = Tiler(xa, tile_size=tile_size, overlap=overlap, mode=mode)
    n_tiles = len(tiler)

    logger.info(
        "Creating tile store: %s (%d tiles, grid=%s, tile_dims=%s)",
        store,
        n_tiles,
        tiler.tile_grid_shape,
        tiler.tile_dims,
    )

    # Create directory structure and root zarr group
    well_dir = store / _WELL_ROW / _WELL_COL
    well_dir.mkdir(parents=True)

    # Write store-level metadata via zarr's attr API (handles v0.4 .zattrs and v0.5 zarr.json)
    root = zarr.open_group(str(store), mode="w")
    root.attrs["iohub"] = {
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_dims": list(tiler.tile_dims),
        "n_tiles": n_tiles,
    }

    # Batch tile IDs
    all_ids = list(range(n_tiles))
    batches = [all_ids[i : i + tile_batch_size] for i in range(0, n_tiles, tile_batch_size)]

    logger.info(
        "Created %d batches of up to %d tiles each",
        len(batches),
        tile_batch_size,
    )
    return batches


# ---------------------------------------------------------------------------
# Phase 2: process_tiles
# ---------------------------------------------------------------------------


@experimental
def process_tiles(
    data: Position,
    fn: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
    store: str | Path,
    tile_ids: list[int],
) -> None:
    """Apply fn to a batch of tiles and write results to the temp tile store.

    Designed to run as a SLURM job (via submitit). Each tile is written as
    a standalone OME-Zarr Position at ``store/A/1/tile_{tile_id}/``.

    Parameters
    ----------
    data : Position
        Source OME-Zarr Position (same as passed to :func:`create_tile_store`).
    fn : callable
        Function to apply to each tile. Receives an xr.DataArray (5D TCZYX),
        returns an xr.DataArray or np.ndarray of the same shape.
    store : str | Path
        Path to the temp tile store created by :func:`create_tile_store`.
    tile_ids : list[int]
        Tile IDs to process (one batch from :func:`create_tile_store`).
    """
    from iohub.ngff import open_ome_zarr
    from iohub.ngff.models import TransformationMeta

    store = Path(store)
    meta = _read_store_meta(store)
    tile_size = meta["tile_size"]
    overlap = meta.get("overlap", {})
    mode_str = meta.get("mode", "squeeze")
    mode = SamplingMode(mode_str) if isinstance(mode_str, str) else SamplingMode.SQUEEZE

    xa = data.to_xarray()
    tiler = Tiler(xa, tile_size=tile_size, overlap=overlap, mode=mode)
    tiles = list(tiler)

    # Get source scale and channel names
    src_transforms = data.metadata.multiscales[0].datasets[0].coordinate_transformations
    channel_names = list(data.channel_names)

    # Infer pixel spacings per tiled dim
    spacings: dict[str, float] = {}
    for dim in tiler.tile_dims:
        if dim in xa.coords:
            spacings[dim] = _pixel_spacing(xa.coords[dim].values)
        else:
            spacings[dim] = 1.0

    # Full dim order for building translation
    all_dims = tuple(str(d) for d in xa.dims)

    for tile_id in tile_ids:
        tile = tiles[tile_id]
        tile_xa = tile.to_xarray()

        result = fn(tile_xa)
        if isinstance(result, np.ndarray):
            result_arr = result
        else:
            result_arr = result.values

        # Validate result shape
        if result_arr.ndim != 5:
            raise ValueError(f"fn must return a 5D array (TCZYX), got shape {result_arr.shape} for tile {tile_id}")
        if result_arr.shape != tile_xa.shape:
            raise ValueError(
                f"fn returned shape {result_arr.shape} but tile {tile_id} has shape {tile_xa.shape}. "
                f"fn must preserve spatial dimensions."
            )

        # Build translation transform from tile start positions
        translation = []
        for dim in all_dims:
            if dim in tile.slices:
                start = tile.slices[dim].start
                translation.append(float(start) * spacings.get(dim, 1.0))
            else:
                translation.append(0.0)

        # Build transforms: copy scale from source, set translation from tile position
        scale_meta = None
        for tr in src_transforms or []:
            if tr.type == "scale":
                scale_meta = deepcopy(tr)
                break
        if scale_meta is None:
            scale_meta = TransformationMeta(type="scale", scale=[1.0] * len(all_dims))

        translation_meta = TransformationMeta(type="translation", translation=translation)

        # Write tile as standalone OME-Zarr FOV
        tile_path = str(_tile_position_path(store, tile_id))
        tile_meta = {
            "tile_id": tile_id,
            "slices": {d: [s.start, s.stop] for d, s in tile.slices.items()},
        }
        pos = open_ome_zarr(tile_path, layout="fov", mode="w-", channel_names=channel_names)
        pos.create_image(
            "0",
            result_arr,
            transform=[scale_meta, translation_meta],
        )
        # Write iohub metadata after image data — if this is interrupted the tile
        # directory exists but stitch_from_store will raise a clear error on missing "slices".
        pos.zattrs["iohub"] = tile_meta

        logger.info("Wrote tile %d to %s", tile_id, tile_path)


# ---------------------------------------------------------------------------
# Phase 3: stitch_from_store
# ---------------------------------------------------------------------------


@experimental
def stitch_from_store(
    store: str | Path,
    output: str | Path,
    source_position: Position,
    *,
    weights: str | Blender = "gaussian",
) -> None:
    """Blend tile results from temp store into a final output OME-Zarr.

    Reads all tile FOVs from the temp store, reconstructs their spatial
    positions, blends overlapping regions, and writes the result to
    ``output``.

    Parameters
    ----------
    store : str | Path
        Temp tile store created by :func:`create_tile_store` and populated
        by :func:`process_tiles`.
    output : str | Path
        Path for the output OME-Zarr store (Position layout).
    source_position : Position
        Original source Position — used to reconstruct Tiler and copy
        OME-Zarr metadata to the output.
    weights : str | Blender
        Blending strategy. Default: ``"gaussian"``.
    """
    from iohub.ngff import open_ome_zarr

    store = Path(store)
    meta = _read_store_meta(store)
    tile_size = meta["tile_size"]
    overlap = meta.get("overlap", {})
    n_tiles = meta["n_tiles"]
    mode_str = meta.get("mode", "squeeze")
    mode = SamplingMode(mode_str) if isinstance(mode_str, str) else SamplingMode.SQUEEZE

    source_xa = source_position.to_xarray()
    tiler = Tiler(source_xa, tile_size=tile_size, overlap=overlap, mode=mode)

    logger.info("Stitching %d tiles from %s → %s", n_tiles, store, output)

    # Read all tile results + reconstruct Tile specs
    tile_xarrays: list[xr.DataArray] = []
    tile_specs: list[Tile] = []

    for i in range(n_tiles):
        tile_path = str(_tile_position_path(store, i))
        try:
            pos = open_ome_zarr(tile_path, layout="fov")
        except Exception as e:
            raise FileNotFoundError(
                f"Tile {i} not found in store {store}. Was process_tiles run for tile_id={i}? Path: {tile_path}"
            ) from e

        tile_meta = pos.zattrs.get("iohub", {})
        if "slices" not in tile_meta:
            raise ValueError(
                f"Tile {i} at {tile_path} is missing 'slices' in iohub metadata. "
                f"The tile may have been partially written (e.g. process killed mid-write)."
            )

        slices = {d: slice(v[0], v[1]) for d, v in tile_meta["slices"].items()}
        tile = Tile(tile_id=i, slices=slices, data=source_xa)

        tile_xarrays.append(pos.to_xarray())
        tile_specs.append(tile)

    # Blend
    blender = get_blender(weights)
    blended = _blend_tiles(tile_xarrays, tile_specs, blender, tiler)

    # Write output OME-Zarr
    logger.info("Writing output to %s", output)
    dst = open_ome_zarr(
        str(output),
        layout="fov",
        mode="w-",
        channel_names=list(source_position.channel_names),
    )

    shape = tuple(blended.sizes[d] for d in blended.dims)
    dtype = source_position.data.dtype

    # Copy coordinate transforms from source
    src_transforms = deepcopy(source_position.metadata.multiscales[0].datasets[0].coordinate_transformations)
    dst.create_zeros("0", shape=shape, dtype=dtype)
    dst.metadata.multiscales[0].datasets[0].coordinate_transformations = src_transforms
    dst.dump_meta()

    output_arr = dst["0"]

    # Compute blended result and write chunk-by-chunk
    blended_computed = blended.values.astype(dtype)
    output_arr[...] = blended_computed

    logger.info("Stitch complete: %s (shape=%s)", output, shape)


# ---------------------------------------------------------------------------
# apply_func_tiled (in-memory, unchanged)
# ---------------------------------------------------------------------------


@overload
def apply_func_tiled(
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
def apply_func_tiled(
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
def apply_func_tiled(
    data: Position | Well,
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
    """Tile a volume, apply a function, and blend back in memory.

    Returns a lazy dask-backed ``xr.DataArray`` without writing to zarr.
    For large volumes use :func:`create_tile_store` / :func:`process_tiles` /
    :func:`stitch_from_store` instead.

    Parameters
    ----------
    data : Position | Well
        Input volume.
    fn : callable
        Function applied to each tile.
    tile_size : dict[str, int]
        Tile size, e.g. ``{"y": 1024, "x": 1024}``.
    overlap : dict[str, int] | None
        Overlap between tiles.
    weights : str | Blender
        Blending strategy. Default: ``"gaussian"``.
    mode : SamplingMode
        Border handling. Default: SQUEEZE.
    align_to_chunks : bool
        Snap tile boundaries to chunk multiples.
    cache : ``"persist"`` | ``"bfs"`` | None
        Overlap caching strategy.

    Returns
    -------
    xr.DataArray
        Lazy dask-backed result.
    """
    from iohub.tile._cache import _bfs_tile_order, _persist_overlaps

    well_kwargs: dict = {}
    if layout_resolver is not None:
        well_kwargs["layout_resolver"] = layout_resolver
    if compositor != "mean":
        well_kwargs["compositor"] = compositor

    xa = _to_xarray(data, **well_kwargs)

    tiler = Tiler(
        xa,
        tile_size=tile_size,
        overlap=overlap,
        mode=mode,
        align_to_chunks=align_to_chunks,
    )

    if cache == "persist":
        _persist_overlaps(tiler)

    tile_specs = list(tiler)
    if cache == "bfs":
        tile_order = _bfs_tile_order(tiler)
    else:
        tile_order = list(range(len(tile_specs)))

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
    return _blend_tiles(processed, tile_specs, blender, tiler)
