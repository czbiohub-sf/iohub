"""Graph-informed overlap caching for tile processing.

Uses the Tiler's neighborhood graph to identify overlapping regions
and optimize tile processing order for cache locality.
"""

from __future__ import annotations

from collections import deque
from functools import reduce
from operator import mul

import dask
import xarray as xr

from iohub.tile._tiler import Tiler


def _overlap_regions(tiler: Tiler) -> list[dict[str, slice]]:
    """Compute unique overlap strips from the neighborhood graph.

    For each edge ``(tile_a, tile_b)`` in the graph, the overlap is
    the bounding-box intersection across all tiled dimensions. Strips
    covering identical pixel ranges are deduplicated.

    Parameters
    ----------
    tiler : Tiler
        Tiler with overlap > 0 in at least one dimension.

    Returns
    -------
    list[dict[str, slice]]
        Overlap regions as dicts of slices keyed by dim name.
    """
    tile_dims = tiler.tile_dims
    seen: set[tuple[tuple[str, int, int], ...]] = set()
    regions: list[dict[str, slice]] = []

    for a, b in tiler.graph.edges():
        spec_a, spec_b = tiler[a], tiler[b]

        overlap_slices: dict[str, slice] = {}
        valid = True
        for dim in tile_dims:
            start = max(spec_a.slices[dim].start, spec_b.slices[dim].start)
            end = min(spec_a.slices[dim].stop, spec_b.slices[dim].stop)
            if end <= start:
                valid = False
                break
            overlap_slices[dim] = slice(start, end)

        if not valid:
            continue

        key = tuple((d, s.start, s.stop) for d, s in overlap_slices.items())
        if key not in seen:
            seen.add(key)
            regions.append(overlap_slices)

    return regions


def _bfs_tile_order(tiler: Tiler) -> list[int]:
    """BFS traversal of the tile neighborhood graph.

    Processes adjacent tiles consecutively so their shared overlap
    chunks stay hot in cache. Starts from tile 0.

    Parameters
    ----------
    tiler : Tiler
        Tiler with a neighborhood graph.

    Returns
    -------
    list[int]
        Tile IDs in BFS order.
    """
    graph = tiler.graph
    if graph.number_of_nodes() == 0:
        return []

    visited: set[int] = set()
    order: list[int] = []
    queue: deque[int] = deque([0])
    visited.add(0)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # Include any disconnected tiles (no overlap)
    for tile_id in range(len(tiler)):
        if tile_id not in visited:
            order.append(tile_id)

    return order


def _estimate_overlap_bytes(tiler: Tiler) -> int:
    """Estimate total bytes needed to cache all overlap regions.

    Parameters
    ----------
    tiler : Tiler
        Tiler with overlap.

    Returns
    -------
    int
        Approximate byte count for all overlap strips.
    """
    data = tiler.data
    tile_dims = tiler.tile_dims

    # Leading dims: all dims NOT in tiled dims
    leading = 1
    for dim in data.dims:
        if dim not in tile_dims:
            leading *= data.sizes[dim]
    itemsize = data.dtype.itemsize

    total = 0
    for overlap_slices in _overlap_regions(tiler):
        region_size = reduce(
            mul,
            (s.stop - s.start for s in overlap_slices.values()),
            1,
        )
        total += leading * region_size * itemsize
    return total


def _persist_overlaps(tiler: Tiler) -> None:
    """Pre-compute and cache overlap regions in memory.

    Slices each overlap strip from the source data and calls
    ``dask.persist()`` so they are loaded once and shared across
    tiles that read the same region.

    Parameters
    ----------
    tiler : Tiler
        Tiler with overlap > 0.
    """
    regions = _overlap_regions(tiler)
    if not regions:
        return

    data = tiler.data
    strips: list[xr.DataArray] = []
    for overlap_slices in regions:
        strips.append(data.isel(**overlap_slices))

    dask.persist(*strips)
