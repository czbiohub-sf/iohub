from __future__ import annotations

import json
import logging
import mmap
import os
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from importlib.metadata import version as _get_package_version
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal, Protocol, override

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from iohub.mm_fov import FrameLocation, MicroManagerFOV
from iohub.nd2 import ND2Dataset
from iohub.ngff.models import TransformationMeta
from iohub.ngff.nodes import Plate, Position, open_ome_zarr
from iohub.ngff.utils import (
    _clamp_chunks_to_shape,
    _limit_zyx_chunk_size,
)
from iohub.reader import MMStack, NDTiffDataset, read_images

__all__ = ["FrameSlab", "TIFFConverter"]
_logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 500e6  # in bytes
DEFAULT_NUM_WORKERS = 4
_PLANE_METADATA_FILENAME = "image_plane_metadata.json"
_PLANE_METADATA_PARTS_DIRNAME = "image_plane_metadata.parts"

type FrameMetadata = dict[str, dict[str, Any]]
"""Plane metadata of one FOV keyed by ``"t/c/z"``."""

type Region = tuple[slice, slice, slice]
"""T, C, Z slices of a :class:`FrameSlab` within its FOV array."""


@dataclass(frozen=True, slots=True)
class FrameSlab:
    """A block of whole frames of one FOV, bounded in T, C and Z: the unit of conversion.

    Slabs never overlap, so any subset can be converted in any process or job;
    ``TIFFConverter.slabs`` enumerates them deterministically.
    """

    zarr_path: str
    t: tuple[int, int]
    c: tuple[int, int]
    z: tuple[int, int]

    @property
    def region(self) -> Region:
        return slice(*self.t), slice(*self.c), slice(*self.z)

    def frame_key(self, t: int, c: int, z: int) -> str:
        """Key of a frame given its indices relative to this slab."""
        return f"{self.t[0] + t}/{self.c[0] + c}/{self.z[0] + z}"


# ------------------------------------------------------------------ chunk sources


class ChunkSource(Protocol):
    """Where the conversion engine gets pixels and plane metadata for one :class:`FrameSlab`.

    Implementations must be safe to call from several threads at once; serialize
    internally if the underlying reader is not.
    """

    writes_metadata: bool
    """Whether :meth:`read` returns plane metadata (some formats have none to give)."""

    def read(self, slab: FrameSlab) -> tuple[np.ndarray, FrameMetadata]:
        """Return the slab's TCZYX volume and its plane metadata keyed by ``"t/c/z"``."""
        ...

    def close(self) -> None: ...


class _DaskSource(ChunkSource):
    """Read slabs through the reader's array interface; plane metadata through ``frame_metadata``.

    Readers are not known to be thread-safe, so reads are serialized with a lock.
    """

    def __init__(
        self,
        fovs: Mapping[str, MicroManagerFOV],
        channel_key: Callable[[int], int | str],
        writes_metadata: bool,
    ) -> None:
        self._fovs = fovs
        self._channel_key = channel_key
        self.writes_metadata = writes_metadata
        self._lock = threading.Lock()

    @override
    def read(self, slab: FrameSlab) -> tuple[np.ndarray, FrameMetadata]:
        fov = self._fovs[slab.zarr_path]
        metadata: FrameMetadata = {}
        with self._lock:
            volume = np.asarray(fov.xdata.data[slab.region])
            if self.writes_metadata:
                missing = False
                for t in range(*slab.t):
                    for c in range(*slab.c):
                        for z in range(*slab.z):
                            frame = fov.frame_metadata(t=t, c=self._channel_key(c), z=z)
                            if frame is None:
                                missing = True
                                continue
                            metadata[f"{t}/{c}/{z}"] = frame
                if missing:
                    _warn_missing(slab)
        return volume, metadata

    @override
    def close(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _RunFrame:
    t: int
    c: int
    z: int
    pixel_offset: int  # relative to the run start
    metadata_offset: int  # relative to the run start
    metadata_length: int


@dataclass(frozen=True, slots=True)
class _ReadRun:
    file: Path
    offset: int
    length: int
    frames: tuple[_RunFrame, ...]


_DIRECT_ALIGNMENT = 4096


class _ByteRangeSource(ChunkSource):
    """Read slabs with one contiguous read per run of adjacent frames.

    Uses ``O_DIRECT`` when the filesystem allows it (whole request submitted at once,
    no page-cache churn), otherwise buffered ``preadv``. Plane metadata JSON sits in
    the gaps between frames and is decoded from the same bytes.
    """

    writes_metadata = True

    def __init__(
        self,
        locations: Iterable[FrameLocation],
        zarr_paths: Mapping[str, str],
        frame_shape: tuple[int, int],
        dtype: np.dtype,
        gap_threshold: int = 1 << 20,
    ) -> None:
        self._frames: dict[str, dict[tuple[int, int, int], FrameLocation]] = {}
        for location in locations:
            self._frames.setdefault(zarr_paths[location.position], {})[location.t, location.c, location.z] = location
        self._frame_shape = frame_shape
        self._dtype = dtype
        self._gap_threshold = gap_threshold
        self._fds: dict[Path, int] = {}
        self._lock = threading.Lock()
        self._direct: bool | None = None

    # -- files ---------------------------------------------------------------

    def _fd(self, file: Path) -> int:
        fd = self._fds.get(file)
        if fd is None:
            with self._lock:
                fd = self._fds.get(file)
                if fd is None:
                    if self._direct is None:
                        self._direct = self._probe_direct(file)
                    fd = os.open(file, os.O_RDONLY | (os.O_DIRECT if self._direct else 0))
                    self._fds[file] = fd
        return fd

    @staticmethod
    def _probe_direct(file: Path) -> bool:
        try:
            os.close(os.open(file, os.O_RDONLY | os.O_DIRECT))
        except OSError as error:
            _logger.info(f"O_DIRECT unavailable for {file.parent} ({error.strerror}); using buffered reads.")
            return False
        _logger.info("Reading NDTiff frames with O_DIRECT.")
        return True

    @override
    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    # -- reads ----------------------------------------------------------------

    def _runs(self, slab: FrameSlab) -> tuple[list[_ReadRun], int]:
        """Coalesce the slab's frames into contiguous byte ranges; returns (runs, missing frame count)."""
        frames = self._frames.get(slab.zarr_path, {})
        located = []
        expected = 0
        for t in range(*slab.t):
            for c in range(*slab.c):
                for z in range(*slab.z):
                    expected += 1
                    location = frames.get((t, c, z))
                    if location is not None:
                        located.append((location, t - slab.t[0], c - slab.c[0], z - slab.z[0]))
        located.sort(key=lambda entry: (str(entry[0].file), min(entry[0].pixel_offset, entry[0].metadata_offset)))
        runs: list[_ReadRun] = []
        current: list[tuple[FrameLocation, int, int, int]] = []
        current_end = 0

        def flush() -> None:
            if not current:
                return
            start = min(current[0][0].pixel_offset, current[0][0].metadata_offset)
            runs.append(
                _ReadRun(
                    file=current[0][0].file,
                    offset=start,
                    length=current_end - start,
                    frames=tuple(
                        _RunFrame(t, c, z, loc.pixel_offset - start, loc.metadata_offset - start, loc.metadata_length)
                        for loc, t, c, z in current
                    ),
                )
            )

        for location, t, c, z in located:
            start = min(location.pixel_offset, location.metadata_offset)
            end = max(
                location.pixel_offset + location.pixel_nbytes, location.metadata_offset + location.metadata_length
            )
            if current and (location.file != current[0][0].file or start - current_end > self._gap_threshold):
                flush()
                current = []
            current_end = end if not current else max(current_end, end)
            current.append((location, t, c, z))
        flush()
        return runs, expected - len(located)

    def _read_run(self, run: _ReadRun) -> tuple[Any, int]:
        """Return (buffer, offset of the run's first byte within the buffer)."""
        fd = self._fd(run.file)
        if self._direct:
            start = run.offset - run.offset % _DIRECT_ALIGNMENT
            end = -(-(run.offset + run.length) // _DIRECT_ALIGNMENT) * _DIRECT_ALIGNMENT
            buffer = mmap.mmap(-1, end - start)  # page-aligned, as O_DIRECT requires
            needed = run.offset + run.length - start
            if _preadv_all(fd, buffer, start, allow_eof=True) < needed:
                raise EOFError(f"{run.file}: short read at offset {start}")
            return buffer, run.offset - start
        buffer = bytearray(run.length)
        _preadv_all(fd, buffer, run.offset)
        return buffer, 0

    @override
    def read(self, slab: FrameSlab) -> tuple[np.ndarray, FrameMetadata]:
        runs, missing = self._runs(slab)
        shape = (slab.t[1] - slab.t[0], slab.c[1] - slab.c[0], slab.z[1] - slab.z[0], *self._frame_shape)
        volume = np.zeros(shape, self._dtype) if missing else np.empty(shape, self._dtype)
        count = self._frame_shape[0] * self._frame_shape[1]
        metadata: FrameMetadata = {}
        for run in runs:
            buffer, base = self._read_run(run)
            for frame in run.frames:
                volume[frame.t, frame.c, frame.z] = np.frombuffer(
                    buffer, self._dtype, count=count, offset=base + frame.pixel_offset
                ).reshape(self._frame_shape)
                start = base + frame.metadata_offset
                key = slab.frame_key(frame.t, frame.c, frame.z)
                try:
                    metadata[key] = json.loads(buffer[start : start + frame.metadata_length])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    _logger.warning(f"Unable to decode metadata for {slab.zarr_path} frame {key}")
            if isinstance(buffer, mmap.mmap):
                buffer.close()
        if missing:
            _warn_missing(slab)
        return volume, metadata


def _preadv_all(fd: int, buffer: Any, offset: int, *, allow_eof: bool = False) -> int:
    """Fill ``buffer`` from ``fd`` starting at ``offset``, looping on short reads."""
    view = memoryview(buffer)
    total = len(view)
    read = 0
    while read < total:
        n = os.preadv(fd, [view[read:]], offset + read)
        if n == 0:
            if allow_eof:
                break
            raise EOFError(f"fd {fd}: hit EOF after {read} of {total} bytes at offset {offset}")
        read += n
    return read


def _warn_missing(slab: FrameSlab) -> None:
    _logger.warning(
        f"Cannot load data at P: {slab.zarr_path}, T: {slab.t[0]}, C: {slab.c[0]}, "
        "filling with zeros. Raw data may be incomplete."
    )


# --------------------------------------------------------------- plane metadata


def _write_plane_metadata(output: Path, zarr_path: str, frames: FrameMetadata) -> None:
    """Write one FOV's ``image_plane_metadata.json``: frames in T/C/Z order, keys sorted (no-dash first)."""
    sorted_keys: list[str] = []
    document: dict[str, dict[str, Any]] = {}
    for key in sorted(frames, key=lambda k: tuple(int(i) for i in k.split("/"))):
        frame = frames[key]
        if not sorted_keys:
            sorted_keys = sorted(frame.keys(), key=lambda x: ("-" in x, x))
        document[key] = {k: frame[k] for k in sorted_keys}
    # write(dumps()) uses the C encoder; json.dump(fp) always takes the pure-Python path
    with (output / zarr_path / "0" / _PLANE_METADATA_FILENAME).open(mode="x") as handle:
        handle.write(json.dumps(document, indent=4))


def _write_plane_metadata_part(output: Path, zarr_path: str, frames: FrameMetadata, label: str) -> None:
    parts = output / zarr_path / "0" / _PLANE_METADATA_PARTS_DIRNAME
    parts.mkdir(exist_ok=True)
    (parts / f"{label}.json").write_text(json.dumps(frames))


def _merge_plane_metadata_parts(output: Path, zarr_path: str) -> None:
    parts = output / zarr_path / "0" / _PLANE_METADATA_PARTS_DIRNAME
    frames: FrameMetadata = {}
    part_files = sorted(parts.glob("*.json")) if parts.is_dir() else []
    for part in part_files:
        frames.update(json.loads(part.read_text()))
    _write_plane_metadata(output, zarr_path, frames)
    for part in part_files:
        part.unlink()
    if parts.is_dir():
        parts.rmdir()


def _create_grid_from_coordinates(xy_coords: list[tuple[float, float]], rows: int, columns: int):
    """Create a grid from XY-position coordinates.

    Parameters
    ----------
    xy_coords : list[tuple[float, float]]
        (X, Y) stage position list in the order in which it was acquired.
    rows : int
        number of rows in the grid-like acquisition
    columns : int
        number of columns in the grid-like acquisition

    Returns
    -------
    NDArray
        A grid-like array mimicking the shape of the acquisition where the
        value in the array corresponds to the position index at that location.
    """
    coords = {}

    coords_list = []
    for idx, pos in enumerate(xy_coords):
        coords[idx] = pos
        coords_list.append(pos)

    # sort by X and then by Y
    coords_list.sort(key=lambda x: x[0])
    coords_list.sort(key=lambda x: x[1])

    # reshape XY coordinates into their proper 2D shape
    grid = np.reshape(coords_list, (rows, columns, 2))
    pos_index_grid = np.zeros((rows, columns), "uint16")
    keys = list(coords.keys())
    vals = list(coords.values())

    for row in range(rows):
        for col in range(columns):
            # append position index (key) into a final grid
            # by indexed into the coordinate map (values)
            pos_index_grid[row, col] = keys[vals.index(list(grid[row, col]))]

    return pos_index_grid


class TIFFConverter:
    """Convert supported microscopy datasets to OME-Zarr.

    Supports Micro-Manager TIFF formats (OME-TIFF and NDTiff) and Nikon ND2.
    Each FOV is written to a separate well in the plate layout.

    Parameters
    ----------
    input_dir : str | Path
        Input directory path
    output_dir : str | Path
        Output zarr directory path
    grid_layout : bool, optional
        Whether to lay out the positions in a grid-like format
        based on how the data was acquired
        (useful for tiled acquisitions), by default False
    chunks : tuple[int] or Literal['XY', 'XYZ'], optional
        Chunk size of the output Zarr arrays, by default None
        (chunk by XYZ volumes or 500 MB size limit, whichever is smaller)
    hcs_plate : bool, optional
        Create NGFF HCS layout based on position names from the
        HCS Site Generator in Micro-Manager (only available for OME-TIFF),
        and is ignored for other formats, by default None
        (attempt to apply to OME-TIFF datasets, disable this with ``False``)
    version : Literal["0.4", "0.5"], optional
        OME-NGFF version for the output Zarr store, by default "0.4".
        "0.4" uses Zarr v2 format; "0.5" uses Zarr v3 format.
    implementation : str, optional
        Zarr backend implementation to use for writing.
        None (default) uses zarr-python. Pass "tensorstore"
        to write via TensorStore (requires the optional tensorstore dependency).
    num_workers : int, optional
        Threads copying pixels concurrently, by default 4. Each in-flight slab holds
        a few times its size in memory (~150 MB slabs -> ~600 MB per worker).
        NDTiff datasets are read with byte-range I/O and scale with workers; other
        formats are read through their reader serially.

    Notes
    -----
    The image plane metadata for each FOV is aggregated into a JSON file,
    and placed under the Zarr array directory
    (e.g. ``/row/column/fov/0/image_plane_metadata.json``).

    Conversion is a list of independent :class:`FrameSlab` (blocks of whole frames of
    one FOV). ``converter()`` runs everything; schedulers may instead call
    :meth:`init_store` once, :meth:`convert` on disjoint slab slices in separate
    processes or jobs, then :meth:`finalize` once.
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        grid_layout: int = False,
        chunks: tuple[int] | Literal["XY", "XYZ"] | None = None,
        hcs_plate: bool | None = None,
        version: Literal["0.4", "0.5"] = "0.4",
        implementation: str | None = None,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        if num_workers < 1:
            raise ValueError(f"num_workers must be at least 1, got {num_workers}.")
        self.num_workers = num_workers
        self.implementation = implementation
        if version not in ("0.4", "0.5"):
            raise ValueError(f"Unsupported OME-NGFF version '{version}'. Supported versions are '0.4' and '0.5'.")
        self.version = version
        _logger.debug("Checking output.")
        output_dir = Path(output_dir)
        if ".zarr" not in output_dir.suffixes:
            raise ValueError("Please specify .zarr at the end of your output")
        self.output_dir = output_dir
        _logger.info("Initializing data.")
        self.reader = read_images(input_dir)
        if reader_type := type(self.reader) not in (
            MMStack,
            NDTiffDataset,
            ND2Dataset,
        ):
            raise TypeError(f"Reader type {reader_type} not supported for conversion.")
        _logger.debug("Finished initializing data.")
        self.summary_metadata = self.reader.micromanager_summary
        self.save_name = output_dir.name
        _logger.debug("Getting dataset summary information.")
        self.coord_map = {}
        self.p = len(self.reader)
        self.t = self.reader.frames
        self.c = self.reader.channels
        self.z = self.reader.slices
        self.y = self.reader.height
        self.x = self.reader.width
        self.dim = (self.p, self.t, self.c, self.z, self.y, self.x)
        self.prefix_list = []
        self.hcs_plate = hcs_plate
        self._check_hcs_sites()
        self._get_pos_names()
        _logger.info(f"Found Dataset {input_dir} with dimensions (P, T, C, Z, Y, X): {self.dim}")
        self.metadata = {}
        self.metadata["iohub_version"] = _get_package_version("iohub")
        self.metadata["Summary"] = self.summary_metadata
        if grid_layout:
            if hcs_plate:
                raise ValueError("grid_layout and hcs_plate must not be both true")
            _logger.info("Generating HCS plate level grid.")
            try:
                self.position_grid = _create_grid_from_coordinates(*self._get_position_coords())
            except ValueError as e:
                _logger.warning(f"Failed to generate grid layout: {e}")
                self._make_default_grid()
        else:
            self._make_default_grid()
        self.chunks = self._gen_chunks(chunks)
        self.transform = self._scale_voxels()
        self.zarr_position_names = self._zarr_position_names()

    def _check_hcs_sites(self):
        if self.hcs_plate:
            self.hcs_sites = self.reader.hcs_position_labels
        elif self.hcs_plate is None:
            try:
                self.hcs_sites = self.reader.hcs_position_labels
                self.hcs_plate = True
            except ValueError:
                _logger.debug("HCS sites not detected, dumping all position into a single row.")

    def _make_default_grid(self):
        if isinstance(self.reader, NDTiffDataset):
            self.position_grid = np.array([self.pos_names])
        else:
            self.position_grid = np.expand_dims(np.arange(self.p, dtype=int), axis=0)

    def _get_position_coords(self):
        """Get the position coordinates from the reader metadata.

        Raises
        ------
            ValueError: If stage positions are not available.

        Returns
        -------
            list: XY stage position coordinates.
            int: Number of grid rows.
            int: Number of grid columns.
        """
        rows = set()

        cols = set()
        xy_coords = []

        # TODO: account for non MM2gamma meta?
        if not self.reader.stage_positions:
            raise ValueError("Stage positions not available.")
        for idx, pos in enumerate(self.reader.stage_positions):
            try:
                xy_stage = pos["DefaultXYStage"]
                stage_pos = pos[xy_stage]
            except KeyError as err:
                raise ValueError(f"Stage position is not available for position {idx}") from err
            xy_coords.append(stage_pos)
            try:
                rows.add(pos["GridRow"])
                cols.add(pos["GridCol"])
            except KeyError as err:
                raise ValueError(f"Grid indices not available for position {idx}") from err

        return xy_coords, len(rows), len(cols)

    def _get_pos_names(self):
        """Append a list of pos names in ascending order

        (order in which they were acquired).
        """
        self.pos_names = []

        for p in range(self.p):
            try:
                name = self.reader.stage_positions[p]["Label"]
            except (IndexError, KeyError):
                name = str(p)
            self.pos_names.append(name)

    def _gen_chunks(self, input_chunks):
        """Generate valid chunk sizes for the output Zarr array.

        input_chunks may be a string ("XY", "XYZ") or a tuple of chunk
        dimensions. Chunk size will be limited to MAX_CHUNK_SIZE and adjusted
        to divide evenly into dimensions.
        """
        if not input_chunks:
            _logger.debug("No chunk size specified, using ZYX.")
            chunks = [1, 1, self.z, self.y, self.x]
        elif isinstance(input_chunks, tuple):
            if not len(input_chunks) == 5:
                raise ValueError(f"Input chunks must be a tuple of 5 dimensions, got {len(input_chunks)} dimensions.")
            chunks = list(input_chunks)
        elif isinstance(input_chunks, str):
            if input_chunks.lower() == "xy":
                chunks = [1, 1, 1, self.y, self.x]
            elif input_chunks.lower() == "xyz":
                chunks = [1, 1, self.z, self.y, self.x]
            else:
                raise ValueError(f"{input_chunks} chunks are not supported.")
        else:
            raise TypeError(f"Chunk type {type(input_chunks)} is not supported.")

        shape = (self.t, self.c, self.z, self.y, self.x)
        original_chunks = chunks.copy()

        # Limit chunk size to MAX_CHUNK_SIZE by halving Z
        bytes_per_pixel = np.dtype(self.reader.dtype).itemsize
        chunk_zyx_shape = _limit_zyx_chunk_size(shape, bytes_per_pixel, MAX_CHUNK_SIZE, chunks=chunks)
        chunks[-3:] = list(chunk_zyx_shape)

        # Clamp chunks so they don't exceed dimension sizes
        chunks = _clamp_chunks_to_shape(shape, chunks)
        for i, (orig, adj, dim) in enumerate(zip(original_chunks, chunks, shape, strict=False)):
            if adj < orig:
                _logger.warning(f"Chunk size {orig} on axis {i} clamped to {adj} (dimension size {dim}).")

        _logger.debug(f"Zarr store chunk size will be set to {chunks}.")

        return tuple(chunks)

    def _scale_voxels(self):
        example_fov = next(iter(self.reader))[1]
        return [
            TransformationMeta(
                type="scale",
                scale=[example_fov.t_scale, 1.0, *example_fov.zyx_scale],
            )
        ]

    def _zarr_position_names(self) -> list[str]:
        """Output position paths (``row/column/fov``) in reader iteration order."""
        if self.hcs_plate:
            return [f"{row}/{col}/{fov}" for row, col, fov in self.hcs_sites]
        return [f"{row}/{column}/0" for row, columns in enumerate(self.position_grid) for column in columns]

    def _init_zarr_arrays(self):
        zarr_format = 3 if self.version == "0.5" else 2
        _logger.info(f"Converting to OME-Zarr version {self.version} (Zarr format v{zarr_format}).")
        self.writer = open_ome_zarr(
            self.output_dir,
            layout="hcs",
            mode="w-",
            channel_names=self.reader.channel_names,
            version=self.version,
            implementation=self.implementation,
        )
        self._created = 0
        arr_kwargs = {
            "name": "0",
            "shape": (
                self.t if self.t != 0 else 1,
                self.c if self.c != 0 else 1,
                self.z if self.z != 0 else 1,
                self.y,
                self.x,
            ),
            "dtype": self.reader.dtype,
            "chunks": self.chunks,
            "transform": self.transform,
        }
        if self.hcs_plate:
            self._init_hcs_arrays(arr_kwargs)
        else:
            self._init_grid_arrays(arr_kwargs)

    def _init_hcs_arrays(self, arr_kwargs):
        for row, col, fov in self.hcs_sites:
            self._create_zeros_array(row, col, fov, arr_kwargs)
        _logger.info("Created HCS NGFF layout from Micro-Manager HCS position labels.")
        self.writer.print_tree()

    def _init_grid_arrays(self, arr_kwargs):
        for row, columns in enumerate(self.position_grid):
            for column in columns:
                self._create_zeros_array(str(row), str(column), "0", arr_kwargs)

    def _create_zeros_array(self, row_name: str, col_name: str, pos_name: str, arr_kwargs: dict) -> Position:
        pos = self.writer.create_position(row_name, col_name, pos_name)
        _ = pos.create_zeros(**arr_kwargs)
        pos.metadata.omero.name = self.pos_names[self._created]
        self._created += 1
        pos.dump_meta()

    # -- staged conversion ------------------------------------------------------

    @cached_property
    def slabs(self) -> list[FrameSlab]:
        """All slabs, in reader order; deterministic for a given dataset and chunking."""
        t_chunk, c_chunk, z_chunk = self.chunks[:3]
        shape = (max(self.t, 1), max(self.c, 1), max(self.z, 1))
        return [
            FrameSlab(
                path,
                (t0, min(t0 + t_chunk, shape[0])),
                (c0, min(c0 + c_chunk, shape[1])),
                (z0, min(z0 + z_chunk, shape[2])),
            )
            for path in self.zarr_position_names
            for t0 in range(0, shape[0], t_chunk)
            for c0 in range(0, shape[1], c_chunk)
            for z0 in range(0, shape[2], z_chunk)
        ]

    def init_store(self) -> None:
        """Create the OME-Zarr store, positions, empty arrays and root attributes. Run once."""
        self._init_zarr_arrays()
        self.writer.zgroup.attrs.update(self.metadata)
        self.writer.close()

    def _make_source(self) -> ChunkSource:
        """Byte-range reads when the reader offers them for this dataset; its array interface otherwise."""
        try:
            locations = list(self.reader.frame_locations())
        except NotImplementedError as error:
            _logger.info(f"Reading through the {type(self.reader).__name__} array interface: {error}")
        else:
            position_paths = dict(zip((key for key, _ in self.reader), self.zarr_position_names, strict=True))
            return _ByteRangeSource(locations, position_paths, (self.y, self.x), np.dtype(self.reader.dtype))

        def channel_key(c: int) -> int | str:
            if isinstance(self.reader, NDTiffDataset) and self.reader.str_channel_axis:
                return self.reader.channel_names[c]
            return c

        fovs = dict(zip(self.zarr_position_names, (fov for _, fov in self.reader), strict=True))
        # ND2 has no per-frame plane metadata export in v1.
        return _DaskSource(fovs, channel_key, writes_metadata=not isinstance(self.reader, ND2Dataset))

    def convert(self, slabs: slice | None = None) -> None:
        """Copy pixels (and plane metadata) for ``slabs`` into an existing store.

        With ``slabs=None`` every slab is converted and each FOV's plane metadata is
        written when its last slab lands. With a slice, metadata is written as a part
        file for :meth:`finalize` to merge, so disjoint slices may run in different
        processes or jobs concurrently.
        """
        selected = self.slabs if slabs is None else self.slabs[slabs]
        if not selected:
            return
        partial = slabs is not None and len(selected) != len(self.slabs)
        start, stop, _ = slabs.indices(len(self.slabs)) if partial else (0, 0, 1)
        label = f"{start:06d}-{stop:06d}"
        pending: dict[str, int] = {}
        for slab in selected:
            pending[slab.zarr_path] = pending.get(slab.zarr_path, 0) + 1
        collected: dict[str, FrameMetadata] = {path: {} for path in pending}
        collect_lock = threading.Lock()

        source = self._make_source()
        plate: Plate = open_ome_zarr(
            self.output_dir, layout="hcs", mode="r+", version=self.version, implementation=self.implementation
        )
        arrays = {path: plate[path]["0"] for path in pending}

        def run(slab: FrameSlab) -> None:
            volume, metadata = source.read(slab)
            arrays[slab.zarr_path][slab.region] = volume
            with collect_lock:
                collected[slab.zarr_path].update(metadata)
                pending[slab.zarr_path] -= 1
                done = pending[slab.zarr_path] == 0
            if done and source.writes_metadata:
                frames = collected.pop(slab.zarr_path)
                if partial:
                    _write_plane_metadata_part(self.output_dir, slab.zarr_path, frames, label)
                else:
                    _write_plane_metadata(self.output_dir, slab.zarr_path, frames)

        _logger.info(f"Converting {len(selected)} slabs with {self.num_workers} workers.")
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor, logging_redirect_tqdm():
                futures = [executor.submit(run, slab) for slab in selected]
                try:
                    for future in tqdm(
                        as_completed(futures), total=len(futures), desc="Converting images", unit="slab", ncols=80
                    ):
                        future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        finally:
            source.close()
            plate.close()

    def finalize(self) -> None:
        """Merge plane metadata parts written by sliced :meth:`convert` runs. Run once, after all slices."""
        if isinstance(self.reader, ND2Dataset):
            return
        pending = [
            path
            for path in self.zarr_position_names
            if not (self.output_dir / path / "0" / _PLANE_METADATA_FILENAME).exists()
        ]
        if not pending:
            return
        # spawn, not fork: this process has threads (zarr's event loop) and fork-after-threads can deadlock children
        with ProcessPoolExecutor(max_workers=min(8, len(pending)), mp_context=get_context("spawn")) as executor:
            for _ in executor.map(_merge_plane_metadata_parts, [self.output_dir] * len(pending), pending):
                pass

    def __call__(self) -> None:
        """
        Runs the conversion.

        Examples
        --------
        >>> from iohub.convert import TIFFConverter
        >>> converter = TIFFConverter("input/path/", "output/path/")
        >>> converter()
        """
        _logger.debug("Setting up Zarr store.")
        self.init_store()
        _logger.debug("Converting images.")
        self.convert()
        self.finalize()
        self.reader.close()
