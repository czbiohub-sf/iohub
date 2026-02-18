"""Graph-informed overlap caching for tile processing.

Uses the Slicer's neighborhood graph to identify overlapping regions
and optimize tile processing order for cache locality.
"""

from __future__ import annotations

from collections import deque

import dask
import xarray as xr

from iohub.tile._slicer import Slicer


def _overlap_regions(slicer: Slicer) -> list[tuple[slice, slice]]:
    """Compute unique overlap strips from the neighborhood graph.

    For each edge ``(tile_a, tile_b)`` in the graph, the overlap is
    the bounding-box intersection. Strips covering identical pixel
    ranges are deduplicated.

    Parameters
    ----------
    slicer : Slicer
        Slicer with overlap > 0 in at least one dimension.

    Returns
    -------
    list[tuple[slice, slice]]
        ``(y_slice, x_slice)`` pairs into the source data, one per
        unique overlap strip.
    """
    seen: set[tuple[int, int, int, int]] = set()
    regions: list[tuple[slice, slice]] = []

    for a, b in slicer.graph.edges():
        spec_a, spec_b = slicer[a], slicer[b]
        oy0 = max(spec_a.y_slice.start, spec_b.y_slice.start)
        oy1 = min(spec_a.y_slice.stop, spec_b.y_slice.stop)
        ox0 = max(spec_a.x_slice.start, spec_b.x_slice.start)
        ox1 = min(spec_a.x_slice.stop, spec_b.x_slice.stop)

        if oy1 <= oy0 or ox1 <= ox0:
            continue

        key = (oy0, oy1, ox0, ox1)
        if key not in seen:
            seen.add(key)
            regions.append((slice(oy0, oy1), slice(ox0, ox1)))

    return regions


def _bfs_tile_order(slicer: Slicer) -> list[int]:
    """BFS traversal of the tile neighborhood graph.

    Processes adjacent tiles consecutively so their shared overlap
    chunks stay hot in cache. Starts from tile 0.

    Parameters
    ----------
    slicer : Slicer
        Slicer with a neighborhood graph.

    Returns
    -------
    list[int]
        Tile IDs in BFS order.
    """
    graph = slicer.graph
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
    for tile_id in range(len(slicer)):
        if tile_id not in visited:
            order.append(tile_id)

    return order


def _estimate_overlap_bytes(slicer: Slicer) -> int:
    """Estimate total bytes needed to cache all overlap regions.

    Parameters
    ----------
    slicer : Slicer
        Slicer with overlap.

    Returns
    -------
    int
        Approximate byte count for all overlap strips.
    """
    data = slicer.data
    leading = 1
    for dim in data.dims:
        if dim not in ("y", "x"):
            leading *= data.sizes[dim]
    itemsize = data.dtype.itemsize

    total = 0
    for y_sl, x_sl in _overlap_regions(slicer):
        h = y_sl.stop - y_sl.start
        w = x_sl.stop - x_sl.start
        total += leading * h * w * itemsize
    return total


def _persist_overlaps(slicer: Slicer) -> None:
    """Pre-compute and cache overlap regions in memory.

    Slices each overlap strip from the source data and calls
    ``dask.persist()`` so they are loaded once and shared across
    tiles that read the same region.

    Parameters
    ----------
    slicer : Slicer
        Slicer with overlap > 0.
    """
    regions = _overlap_regions(slicer)
    if not regions:
        return

    data = slicer.data
    strips: list[xr.DataArray] = []
    for y_sl, x_sl in regions:
        strips.append(data[..., y_sl, x_sl])

    dask.persist(*strips)
