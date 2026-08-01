"""Tracking of shard-aligned write units so an interrupted run can resume.

:func:`iohub.ngff.utils.process_single_position` splits its work into units of
one shard-aligned batch of timepoints for one channel group. Each unit owns a
set of files on disk outright, which makes two things possible:

*Repair* — a unit that owns its files can unlink them before writing rather
than writing over them. That skips the read-modify-write cycle a partial shard
write would otherwise trigger, and it means a file left half-written by a
killed job is replaced wholesale instead of being read back and rejected.

*Resume* — a marker file per unit records that the unit finished, so a retried
or resumed run recomputes only the units that did not.

Ownership is the precondition for both. A unit owns a file only if the unit
writes every element of that file that lies inside the array bounds; otherwise
unlinking it would discard a neighbouring unit's data, and its presence says
nothing about whether *this* unit finished. Units whose time indices are not
shard-aligned are therefore not tracked, and fall back to the previous
read-modify-write behaviour.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Directory holding per-unit progress markers, inside the array directory.
#: Dot-prefixed and nested under the array (not the position group) so that
#: neither group member listing nor the chunk read path ever sees it.
MARKER_DIRNAME = ".iohub-write-progress"

#: Errors raised by the zarr-python and zarrs codec pipelines when a stored
#: chunk or shard cannot be decoded (truncated file, bad checksum, short
#: shard index).
DECODE_ERRORS = (RuntimeError, ValueError, OSError)


@dataclass(frozen=True)
class WriteUnit:
    """One shard-aligned write and the files it owns exclusively."""

    time_indices: tuple[int, ...]
    channel_indices: tuple[int, ...]
    #: Files this unit owns, paired with the origin element of each, which is
    #: used to probe whether the file decodes.
    shards: tuple[tuple[Path, tuple[int, ...]], ...]
    marker_dir: Path
    #: Opaque string mixed into the unit's identity, so that a caller whose
    #: inputs or settings changed does not match the previous run's markers.
    token: str = ""

    @property
    def name(self) -> str:
        """Stable identifier for this unit, reproducible across processes."""
        payload = repr((self.time_indices, self.channel_indices, self.token)).encode()
        digest = hashlib.sha256(payload).hexdigest()[:12]
        times = f"t{self.time_indices[0]}-{self.time_indices[-1]}"
        channels = f"c{self.channel_indices[0]}-{self.channel_indices[-1]}"
        return f"{times}_{channels}_{digest}"

    @property
    def done_marker(self) -> Path:
        return self.marker_dir / f"{self.name}.done"

    @property
    def inflight_marker(self) -> Path:
        return self.marker_dir / f"{self.name}.inflight"

    def begin(self) -> None:
        """Mark this unit in flight and clear the files it owns.

        Ordered so that no interruption can leave a "done" marker next to
        data this unit has not finished writing: the done marker is removed
        first, the in-flight marker is created next, and only then are the
        owned files unlinked.
        """
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        self.done_marker.unlink(missing_ok=True)
        self.inflight_marker.touch()
        self.clear()

    def clear(self) -> None:
        """Remove the files this unit owns, so the write cannot read them back.

        Useful on its own for a write that is not tracked for resume: it still
        wants a file torn by an earlier kill replaced rather than merged into.
        """
        for path, _ in self.shards:
            path.unlink(missing_ok=True)

    def complete(self) -> None:
        """Record this unit as finished.

        ``os.replace`` within one directory is atomic, so the marker is never
        observed in a half-written state.
        """
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        if self.inflight_marker.exists():
            self.inflight_marker.replace(self.done_marker)
        else:
            self.done_marker.touch()


def tracking_available(array) -> bool:
    """Whether write units can be tracked for ``array`` at all.

    False for a store whose chunk key layout is not modelled here (Zarr v2,
    i.e. OME-Zarr v0.4), one not backed by a filesystem, or one reached
    through an implementation other than zarr-python. Says nothing about
    whether a *particular* write owns its files; see :func:`plan_write_unit`
    for that.
    """
    return _array_directory(array) is not None and _chunk_key_encoding(array) is not None


def plan_write_unit(
    array,
    time_indices: Sequence[int],
    channel_indices: Sequence[int] | slice,
    token: str = "",
) -> WriteUnit | None:
    """Describe the files a ``(timepoints, channels)`` write owns.

    Returns None when ownership cannot be established, in which case the
    caller should write as it always has. That happens for a store whose
    chunk keys are not modelled here (Zarr v2, i.e. OME-Zarr v0.4, which has
    no sharding), for a store that is not on a filesystem, and — the case
    worth understanding — when the requested timepoints or channels do not
    cover the full in-bounds extent of the files they land in.

    Assumes the write spans the entire ZYX extent of the array, which is what
    :func:`iohub.ngff.utils.apply_transform_to_tczyx_and_save` does.

    ``token`` is mixed into the unit's identity. Pass a fingerprint of
    whatever determines the output — settings, input revision — so that a run
    with different parameters does not match the markers of the previous one
    and skip work that would now produce different data.
    """
    directory = _array_directory(array)
    encoding = _chunk_key_encoding(array)
    if directory is None or encoding is None:
        return None

    shape = tuple(array.shape)
    grid = tuple(array.shards or array.chunks)
    if len(grid) != len(shape):
        return None

    times = tuple(sorted(dict.fromkeys(time_indices)))
    channels = tuple(sorted(dict.fromkeys(_as_indices(channel_indices, shape[1]))))
    if not times or not channels:
        return None

    time_cells = _covered_cells(times, grid[0], shape[0])
    channel_cells = _covered_cells(channels, grid[1], shape[1])
    if time_cells is None or channel_cells is None:
        return None

    # The write spans every spatial cell, so all of them are owned.
    spatial_cells = [range(math.ceil(dim / step)) for dim, step in zip(shape[2:], grid[2:], strict=True)]

    shards = tuple(
        (
            directory / encoding.encode_chunk_key(cell),
            tuple(index * step for index, step in zip(cell, grid, strict=True)),
        )
        for cell in itertools.product(time_cells, channel_cells, *spatial_cells)
    )
    return WriteUnit(
        time_indices=times,
        channel_indices=channels,
        shards=shards,
        marker_dir=directory / MARKER_DIRNAME,
        token=token,
    )


def unit_is_complete(unit: WriteUnit, array) -> bool:
    """Whether ``unit`` finished in an earlier run and can be skipped.

    Requires the unit's done marker, and additionally probes each owned file
    to confirm it still decodes. The probe reads one element, which forces the
    shard index and its checksum to be validated; that catches a file whose
    marker was recorded but whose bytes did not all reach disk, for instance
    if a striped filesystem flushed them out of order during a node crash. It
    does not verify every inner chunk, so it is a cheap guard rather than a
    proof of integrity.
    """
    if not unit.done_marker.exists() or unit.inflight_marker.exists():
        return False
    return all(_decodes(array, path, origin) for path, origin in unit.shards)


def _decodes(array, path: Path, origin: tuple[int, ...]) -> bool:
    """Whether one element can be read out of ``path``."""
    if not path.exists():
        # An all-fill-value shard is never written, so a missing file is
        # consistent with a completed unit that had nothing to store.
        return True
    selection = (*origin[:2], *(slice(start, start + 1) for start in origin[2:]))
    try:
        array._impl.read(array.native, selection)
    except DECODE_ERRORS:
        return False
    return True


def _covered_cells(indices: tuple[int, ...], step: int, extent: int) -> tuple[int, ...] | None:
    """Cells along one dimension that ``indices`` covers completely.

    Returns None if any touched cell is only partially covered, since the
    unit would then share that file with another write.
    """
    cells = sorted({index // step for index in indices})
    selected = set(indices)
    for cell in cells:
        in_bounds = range(cell * step, min((cell + 1) * step, extent))
        if not selected.issuperset(in_bounds):
            return None
    return tuple(cells)


def _as_indices(indices: Sequence[int] | slice, extent: int) -> tuple[int, ...]:
    if isinstance(indices, slice):
        return tuple(range(*indices.indices(extent)))
    return tuple(indices)


def _chunk_key_encoding(array):
    """Chunk key encoding of ``array``, or None if it does not expose one.

    Absent for Zarr v2 metadata and for handles from implementations other
    than zarr-python (e.g. tensorstore), which are left untracked.
    """
    metadata = getattr(array.native, "metadata", None)
    return getattr(metadata, "chunk_key_encoding", None)


def _array_directory(array) -> Path | None:
    """Filesystem directory backing ``array``, or None if it is not local."""
    native = array.native
    root = getattr(getattr(native, "store", None), "root", None)
    path = getattr(native, "path", None)
    if root is None or path is None:
        return None
    return Path(root) / path
