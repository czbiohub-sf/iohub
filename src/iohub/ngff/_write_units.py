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
import json
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Directory holding per-unit progress markers. Written *beside* the store, not
#: inside it, so that nothing iohub owns ends up in a published data artifact and
#: a ``cp -r``/``rsync`` of a finished store does not carry progress state that
#: would make a later resume skip everything. Dot-prefixed so it never matches a
#: shell glob, and namespaced by store name because one directory can hold
#: several stores (e.g. a reconstruction output alongside its transfer function).
#:
#:     parent/
#:       my_plate.zarr/                  <- untouched
#:       .iohub-progress/
#:         my_plate.zarr/
#:           A/1/0/t0-0_c0-0_<digest>.done
PROGRESS_DIRNAME = ".iohub-progress"

#: Pre-existing marker directory from when progress lived inside the array.
#: Only used to point the reader at the new location.
LEGACY_MARKER_DIRNAME = ".iohub-write-progress"

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
    #: Directory holding this array's chunk/shard files, which the recorded
    #: shard keys are relative to.
    array_dir: Path
    #: Where completion is recorded. None when the caller did not ask for
    #: tracking (repair only), or when the store root could not be located.
    marker_dir: Path | None = None
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
    def tracked(self) -> bool:
        """Whether completion of this unit is recorded anywhere."""
        return self.marker_dir is not None

    @property
    def done_marker(self) -> Path | None:
        return None if self.marker_dir is None else self.marker_dir / f"{self.name}.done"

    @property
    def inflight_marker(self) -> Path | None:
        return None if self.marker_dir is None else self.marker_dir / f"{self.name}.inflight"

    def begin(self) -> None:
        """Mark this unit in flight and clear the files it owns.

        Ordered so that no interruption can leave a "done" marker next to
        data this unit has not finished writing: the done marker is removed
        first, the in-flight marker is created next, and only then are the
        owned files unlinked.
        """
        if self.marker_dir is not None:
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

    def complete(self, *, wrote: bool = True) -> None:
        """Record this unit as finished, and which shards it guarantees.

        The recorded keys are what makes the record safe to keep outside the
        store: a later resume requires every one of them to still be present
        and decodable, so deleting the store's data does not leave markers
        claiming work that no longer exists. A unit whose input was all zeros
        writes no shard and records an empty list, and is still skippable.

        Pass ``wrote=False`` when the unit produced nothing — an all-zero or
        all-NaN input is skipped rather than written — so the record claims
        nothing. Recording whatever happened to be on disk would otherwise
        attribute a leftover file from an earlier run to this unit.

        Written to the in-flight marker and then renamed, which is atomic
        within one directory, so the record is never read half-written.
        """
        if self.marker_dir is None:
            return
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        written = [self._key(path) for path, _ in self.shards if path.exists()] if wrote else []
        scratch = self.inflight_marker
        scratch.write_text(json.dumps({"shards": written}))
        scratch.replace(self.done_marker)

    def _key(self, path: Path) -> str:
        """Shard path relative to the array directory, e.g. ``c/0/0/0/0/0``."""
        return path.relative_to(self.array_dir).as_posix()


def progress_dir_for(position_path: Path) -> Path | None:
    """Where to record progress for the position at ``position_path``.

    Returns a directory beside the store — ``<store>/../.iohub-progress/
    <store name>/<position path within the store>`` — or None if the store root
    cannot be located, in which case the caller gets repair without resume.

    The store root is found by walking up while the directory is still a Zarr
    node, which handles an HCS plate (three levels above the position) and a
    standalone FOV store (the position *is* the store) without assuming a depth.
    """
    position_path = Path(position_path).resolve()
    if not (position_path / "zarr.json").exists():
        return None
    root = position_path
    while (root.parent / "zarr.json").exists():
        root = root.parent
        if root.parent == root:  # reached the filesystem root
            return None
    relative = position_path.relative_to(root)
    return root.parent / PROGRESS_DIRNAME / root.name / relative


def legacy_progress_dir(array) -> Path | None:
    """Where progress used to be recorded for ``array``, inside the array.

    None if the array is not on a filesystem, or if no such directory exists.
    """
    directory = _array_directory(array)
    if directory is None:
        return None
    legacy = directory / LEGACY_MARKER_DIRNAME
    return legacy if legacy.is_dir() else None


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
    time_indices: int | Sequence[int] | slice,
    channel_indices: int | Sequence[int] | slice,
    token: str = "",
    progress_dir: Path | None = None,
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

    ``progress_dir`` is where completion is recorded, normally from
    :func:`progress_dir_for`. Omit it for repair without resume: the shard
    paths are still computed, so a torn shard is still replaced, but nothing
    is written outside the store. This is what :meth:`Position.write_xarray`
    uses, since it has no notion of a resumable unit of work.
    """
    directory = _array_directory(array)
    encoding = _chunk_key_encoding(array)
    if directory is None or encoding is None:
        return None

    shape = tuple(array.shape)
    grid = tuple(array.shards or array.chunks)
    if len(grid) != len(shape):
        return None

    times = tuple(sorted(dict.fromkeys(_as_indices(time_indices, shape[0]))))
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
        array_dir=directory,
        marker_dir=progress_dir,
        token=token,
    )


def unit_is_complete(unit: WriteUnit, array) -> bool:
    """Whether ``unit`` finished in an earlier run and can be skipped.

    Requires a done marker, and requires every shard the marker says it wrote
    to still be present and decodable. Because the record lives outside the
    store, presence has to be re-checked: deleting the store's data would
    otherwise leave markers claiming work that no longer exists, and a resume
    would skip straight past an empty store.

    The probe reads one element per shard, which forces the shard index and its
    checksum to be validated; that catches a file whose marker was recorded but
    whose bytes did not all reach disk, for instance if a striped filesystem
    flushed them out of order during a node crash. It does not verify every
    inner chunk, so it is a cheap guard rather than a proof of integrity.

    A marker that cannot be parsed — including a zero-byte one written by an
    older version, which recorded no shard list — counts as incomplete, so the
    unit is simply recomputed.
    """
    if not unit.tracked or unit.inflight_marker.exists():
        return False
    if not unit.done_marker.exists():
        return False
    try:
        recorded = set(json.loads(unit.done_marker.read_text())["shards"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all(_decodes(array, path, origin) for path, origin in unit.shards if unit._key(path) in recorded)


def _decodes(array, path: Path, origin: tuple[int, ...]) -> bool:
    """Whether one element can be read out of ``path``."""
    if not path.exists():
        return False
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


def _as_indices(indices: int | Sequence[int] | slice, extent: int) -> tuple[int, ...]:
    """Normalise a selection along one dimension to a tuple of indices.

    Accepts every form a caller may pass to an orthogonal selection: a slice, a
    sequence of indices, or a bare scalar. A scalar is a legal selection —
    ``arr.oindex[[t], 3]`` addresses one channel — and
    ``process_single_position`` produces one whenever a caller passes a flat
    list of channel indices rather than a list of channel groups, which
    ``biahub concatenate`` does.
    """
    if isinstance(indices, slice):
        return tuple(range(*indices.indices(extent)))
    try:
        return (operator.index(indices),)
    except TypeError:
        return tuple(operator.index(index) for index in indices)


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
