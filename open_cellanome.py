"""
Cellanome dataset reader and converter for iohub.

Reads Cellanome imaging datasets and converts them to OME-Zarr format.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Optional, Type

import dask
import dask.array as da
import numpy as np
from dask.base import tokenize
from numpy.typing import ArrayLike
from PIL import Image
from pydantic import BaseModel, ConfigDict

from iohub.fov import BaseFOV, BaseFOVMapping

_logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pydantic metadata models
# -----------------------------------------------------------------------------


class FilterInfo(BaseModel):
    """Filter information from per-image metadata.

    Parameters
    ----------
    name : str
        Filter name (e.g., "White", "Red-CY5 (700)").
    is_brightfield : bool
        Whether this is a brightfield filter.
    display_color : list[int]
        RGB display color values.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    is_brightfield: bool
    display_color: list[int]


class ImageMetadata(BaseModel):
    """Metadata for individual images from .metadata/ directory.

    Parameters
    ----------
    X : float
        Stage X position in micrometers.
    Y : float
        Stage Y position in micrometers.
    Z : int
        Stage Z position.
    Scan : int
        Scan number.
    Lane : int
        Lane number.
    Row : int
        Row number.
    Surface : int
        Surface number.
    FOV : int
        FOV index.
    Objective : str
        Objective used (e.g., "4x").
    PixelsPerMicron : float
        Pixel density in pixels per micron.
    Filter : FilterInfo
        Filter information.
    Exposure : float
        Exposure time.
    timestamp : str | None
        Optional timestamp.
    """

    model_config = ConfigDict(extra="allow")

    X: float
    Y: float
    Z: int
    Scan: int
    Lane: int
    Row: int
    Surface: int
    FOV: int
    Objective: str
    PixelsPerMicron: float
    Filter: FilterInfo
    Exposure: float
    timestamp: str | None = None


class ScanLaneInfo(BaseModel):
    """Lane information within a scan.

    Parameters
    ----------
    lane_name : str
        Name of the lane.
    total_cells : int | None
        Total number of cells detected.
    total_cages : int | None
        Total number of cages detected.
    """

    model_config = ConfigDict(extra="allow")

    lane_name: str
    total_cells: int | None = None
    total_cages: int | None = None


class ScanInfo(BaseModel):
    """Scan information from experiment metadata.

    Parameters
    ----------
    scan_name : str
        Name of the scan.
    lanes : list[ScanLaneInfo]
        List of lanes in this scan.
    """

    model_config = ConfigDict(extra="allow")

    scan_name: str
    lanes: list[ScanLaneInfo]


class ExperimentMetadata(BaseModel):
    """Parse experiment_metadata.json for global experiment info.

    Parameters
    ----------
    title : str
        Experiment title.
    name : str
        Experiment name.
    scan_id : str
        Unique scan identifier.
    instrument_name : str
        Name of the instrument.
    created_at : str
        Creation timestamp.
    status : str
        Experiment status.
    scans : list[ScanInfo]
        List of scans in the experiment.
    """

    model_config = ConfigDict(extra="allow")

    title: str
    name: str
    scan_id: str
    instrument_name: str
    created_at: str
    status: str
    scans: list[ScanInfo]


class InstrumentInfo(BaseModel):
    """Instrument information from experiment manifest.

    Parameters
    ----------
    id : str
        Instrument identifier.
    name : str
        Instrument name.
    friendlyName : str
        Human-friendly instrument name.
    serialNumber : str
        Instrument serial number.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    friendlyName: str
    serialNumber: str


class ExperimentManifest(BaseModel):
    """Parse experiment_manifest.json for experiment settings.

    Parameters
    ----------
    name : str
        Experiment name.
    localId : str
        Local experiment identifier.
    projectId : str | None
        Optional project identifier.
    operators : list[str]
        List of operators.
    description : str
        Experiment description.
    instrument : InstrumentInfo
        Instrument information.
    settings : dict[str, Any]
        Complex nested settings structure.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    localId: str
    projectId: str | None = None
    operators: list[str] = []
    description: str = ""
    instrument: InstrumentInfo
    settings: dict[str, Any]

    @classmethod
    def from_file(cls, path: Path) -> "ExperimentManifest":
        """Load manifest from JSON file.

        Parameters
        ----------
        path : Path
            Path to experiment_manifest.json file.

        Returns
        -------
        ExperimentManifest
            Parsed manifest object.
        """
        return cls.model_validate_json(path.read_text())


# -----------------------------------------------------------------------------
# FOV info dataclass
# -----------------------------------------------------------------------------


@dataclass
class CellanomeFOVInfo:
    """Parsed information from a Cellanome FOV filename."""

    scan: int
    lane: int
    row: int
    fov_idx: int
    surface: int
    x_stage: int
    y_stage: int
    z_stage: int
    channel: str
    filepath: Path

    @classmethod
    def from_filename(cls, filepath: Path) -> "CellanomeFOVInfo":
        """Parse FOV info from filename.

        Expected format:
            {scan}_{lane}_{row}_{fov_idx}_{surface}_{x_stage}_{y_stage}_{z_stage}_{channel}.png

        Example:
            1_3_1_15_1_026389_022585_-17218_White.png
        """
        stem = filepath.stem
        # Handle channel names with spaces/special chars like "Red-CY5 (700)"
        # Pattern: numbers separated by underscores, then channel name
        pattern = r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(-?\d+)_(.+)$"
        match = re.match(pattern, stem)
        if not match:
            raise ValueError(f"Cannot parse Cellanome filename: {filepath.name}")

        return cls(
            scan=int(match.group(1)),
            lane=int(match.group(2)),
            row=int(match.group(3)),
            fov_idx=int(match.group(4)),
            surface=int(match.group(5)),
            x_stage=int(match.group(6)),
            y_stage=int(match.group(7)),
            z_stage=int(match.group(8)),
            channel=match.group(9),
            filepath=filepath,
        )

    @property
    def position_key(self) -> str:
        """Unique key for this FOV position (excludes channel).

        Note: Includes scan number. For cross-scan grouping,
        use spatial_position_key instead.
        """
        return f"{self.scan}_{self.lane}_{self.row}_{self.fov_idx}"

    @property
    def spatial_position_key(self) -> str:
        """Unique key for spatial position (excludes scan and channel).

        Returns
        -------
        str
            Position key as "{lane}_{row}_{fov_idx}" for cross-scan grouping.
        """
        return f"{self.lane}_{self.row}_{self.fov_idx}"


class CellanomeFOV(BaseFOV):
    """A single FOV from a Cellanome dataset.

    Provides a 5D TCZYX view of the FOV data.
    T dimension corresponds to scans (timepoints), Z=1 (single focal plane).
    When multiple scans exist for the same spatial position, T > 1.
    """

    # Default pixel size for 4x objective (um/pixel) when metadata unavailable
    _DEFAULT_PIXEL_SIZE_UM = 0.61

    def __init__(
        self,
        position_key: str,
        scan_infos: dict[int, list[CellanomeFOVInfo]],
        channel_names: list[str],
    ):
        """Initialize CellanomeFOV for multi-scan data.

        Parameters
        ----------
        position_key : str
            Spatial position key "{lane}_{row}_{fov_idx}".
        scan_infos : dict[int, list[CellanomeFOVInfo]]
            Mapping from scan number to list of FOV infos (one per channel).
        channel_names : list[str]
            Global channel names (union across all scans, sorted, filtered).
        """
        self._position_key = position_key
        self._scan_infos = scan_infos
        self._channel_names = channel_names
        self._scan_numbers = sorted(scan_infos.keys())

        # Get dimensions from first available image in first scan
        first_scan = self._scan_numbers[0]
        first_info = scan_infos[first_scan][0]
        with Image.open(first_info.filepath) as img:
            self._height, self._width = img.size[1], img.size[0]
            # PIL mode "I;16" is 16-bit unsigned int
            self._dtype = np.dtype(np.uint16)

        # Root is the directory of the first image
        self._root = first_info.filepath.parent

        # Experiment root is 3 levels up: lane_dir -> scan_dir -> experiment_root
        self._experiment_root = first_info.filepath.parent.parent.parent

        # Stage position from first info (used for scale)
        self._x_stage = first_info.x_stage
        self._y_stage = first_info.y_stage
        self._z_stage = first_info.z_stage

        # Build lazy dask array for data access
        self._xdata = self._build_dask_array()

    @property
    def scan_numbers(self) -> list[int]:
        """Return list of scan numbers, mapping T indices to scans.

        Returns
        -------
        list[int]
            Scan numbers in T order. scan_numbers[t] gives the scan
            number for T index t.
        """
        return self._scan_numbers

    def _find_metadata_file_for_info(
        self, info: CellanomeFOVInfo
    ) -> Path | None:
        """Find metadata file for a specific FOV info.

        Parameters
        ----------
        info : CellanomeFOVInfo
            FOV info to find metadata for.

        Returns
        -------
        Path | None
            Path to metadata file if it exists, None otherwise.
        """
        filename_stem = info.filepath.stem
        metadata_path = (
            self._experiment_root / ".metadata" / f"{filename_stem}.metadata.json"
        )
        if metadata_path.exists():
            return metadata_path
        return None

    def _find_metadata_file(self) -> Path | None:
        """Find the metadata file for this FOV.

        The metadata file is located in the .metadata/ directory at the
        experiment root, with the same stem as the image file.

        Metadata files typically exist only for scan_1 (caging scan).
        This method tries all available scans to find metadata.

        Returns
        -------
        Path | None
            Path to metadata file if it exists, None otherwise.
        """
        # Try each scan in order (scan_1 typically has metadata)
        for scan in self._scan_numbers:
            infos = self._scan_infos[scan]
            if infos:
                metadata_path = self._find_metadata_file_for_info(infos[0])
                if metadata_path is not None:
                    return metadata_path

        return None

    def _get_pixel_size_um(self) -> float:
        """Get pixel size in micrometers per pixel.

        Attempts to load pixel size from the per-image metadata file.
        Falls back to default value if metadata is unavailable.

        Returns
        -------
        float
            Pixel size in micrometers per pixel.
        """
        metadata_file = self._find_metadata_file()
        if metadata_file is not None:
            try:
                meta = ImageMetadata.model_validate_json(metadata_file.read_text())
                # Convert pixels/micron to microns/pixel
                return 1.0 / meta.PixelsPerMicron
            except Exception as e:
                _logger.warning(
                    f"Failed to parse metadata file {metadata_file}: {e}. "
                    f"Using default pixel size: {self._DEFAULT_PIXEL_SIZE_UM} um/pixel"
                )
                return self._DEFAULT_PIXEL_SIZE_UM

        _logger.warning(
            f"Metadata file not found for FOV, "
            f"using default pixel size: {self._DEFAULT_PIXEL_SIZE_UM} um/pixel"
        )
        return self._DEFAULT_PIXEL_SIZE_UM

    def _build_dask_array(self) -> da.Array:
        """Build 5D dask array (T, C, Z, Y, X) from multi-scan data.

        Uses dask's tokenize function to create deterministic task names
        based on file paths and modification times. This enables proper
        caching and reuse of computed results when files haven't changed.

        For each timepoint (scan), builds a CYX array by iterating over
        channels. Missing channels at specific timepoints are zero-filled.

        Returns
        -------
        da.Array
            5D dask array with shape (T, C, 1, Y, X).
        """
        # Collect ALL filepaths across all scans for tokenization
        all_filepaths = []
        for scan in self._scan_numbers:
            for info in self._scan_infos[scan]:
                all_filepaths.append(info.filepath)

        # Tokenize on filenames AND mtimes for cache invalidation
        mtimes = [os.path.getmtime(f) for f in all_filepaths]
        token = tokenize(all_filepaths, mtimes)

        # Build per-timepoint CZYX arrays
        t_chunks = []
        for t_idx, scan in enumerate(self._scan_numbers):
            # Build channel->info mapping for this scan
            infos_by_channel = {info.channel: info for info in self._scan_infos[scan]}

            c_chunks = []
            for channel in self._channel_names:
                if channel in infos_by_channel:
                    filepath = infos_by_channel[channel].filepath
                    # Use pure=True for deterministic behavior
                    delayed_load = dask.delayed(_load_png, pure=True)(filepath)
                    chunk = da.from_delayed(
                        delayed_load,
                        shape=(self._height, self._width),
                        dtype=self._dtype,
                        name=f"imread-{token}-T{t_idx}-{channel}",
                    )
                else:
                    # Missing channel at this timepoint - fill with zeros
                    chunk = da.zeros(
                        (self._height, self._width),
                        dtype=self._dtype,
                    )
                c_chunks.append(chunk)

            # Stack channels: shape (C, Y, X)
            cyx_array = da.stack(c_chunks, axis=0)
            # Add Z dimension: shape (C, Z=1, Y, X)
            czyx_array = cyx_array[:, np.newaxis, :, :]
            t_chunks.append(czyx_array)

        # Stack timepoints: shape (T, C, Z, Y, X)
        return da.stack(t_chunks, axis=0)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def axes_names(self) -> list[str]:
        return ["T", "C", "Z", "Y", "X"]

    @property
    def channel_names(self) -> list[str]:
        return self._channel_names

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """5D shape (T, C, Z, Y, X).

        T is the number of scans containing this spatial position.
        """
        return (
            len(self._scan_numbers),  # T = number of scans
            len(self._channel_names),  # C = global channel union
            1,                          # Z = 1 (single focal plane)
            self._height,
            self._width,
        )

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def _get_stage_position(self) -> tuple[float, float] | None:
        """Extract stage position from metadata or filename.

        Priority:
        1. .metadata/*.metadata.json X, Y fields
        2. Filename x_stage, y_stage fields (integer stage units)

        Returns
        -------
        tuple[float, float] | None
            (X, Y) in stage units, or None if unavailable.
        """
        # Try metadata first (any available scan, prefer earliest for caging)
        for scan in sorted(self._scan_numbers):
            if scan in self._scan_infos:
                first_info = self._scan_infos[scan][0]
                metadata_file = self._find_metadata_file_for_info(first_info)
                if metadata_file is not None:
                    try:
                        meta = ImageMetadata.model_validate_json(
                            metadata_file.read_text()
                        )
                        return (meta.X, meta.Y)
                    except Exception:
                        continue

        # Fallback to filename from first available scan
        first_scan = self._scan_numbers[0]
        first_info = self._scan_infos[first_scan][0]
        return (float(first_info.x_stage), float(first_info.y_stage))

    @property
    def stage_position(self) -> tuple[float, float] | None:
        """Stage position (X, Y) in raw stage units.

        Extracts position from per-image metadata files in .metadata/
        directory. Falls back to filename x_stage, y_stage values
        if metadata is unavailable.

        Returns
        -------
        tuple[float, float] | None
            (X, Y) position in stage units.
        """
        return self._get_stage_position()

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        """Spatial scale in micrometers.

        Extracts pixel size from per-image metadata files in .metadata/
        directory. Falls back to default (0.61 um/pixel for 4x objective)
        if metadata is unavailable.

        Returns
        -------
        tuple[float, float, float]
            (Z, Y, X) scale in micrometers. Z is always 1.0 (single plane).
        """
        pixel_size_um = self._get_pixel_size_um()
        return (1.0, pixel_size_um, pixel_size_um)

    @property
    def t_scale(self) -> float:
        """Time scale in seconds."""
        # Single timepoint per FOV
        return 1.0

    @property
    def xdata(self) -> da.Array:
        """Lazy dask array of the image data."""
        return self._xdata

    def __getitem__(
        self,
        key: int | slice | tuple[int | slice, ...],
    ) -> ArrayLike:
        """Index into the 5D array."""
        return self._xdata[key]


def _load_png(filepath: Path) -> np.ndarray:
    """Load a PNG file as numpy array.

    Parameters
    ----------
    filepath : Path
        Path to the PNG file.

    Returns
    -------
    np.ndarray
        Image data as uint16 numpy array.
    """
    with Image.open(filepath) as img:
        return np.array(img, dtype=np.uint16)


class CellanomeReader(BaseFOVMapping):
    """Reader for Cellanome imaging datasets.

    Discovers and provides access to FOVs organized by spatial position.
    One FOV per spatial position with T > 1 when multiple scans exist.

    FOVs are grouped by spatial position (lane, row, fov_idx), not by scan.
    This allows the same physical location to be tracked across multiple
    timepoints (scans).

    Parameters
    ----------
    root_dir : Path | str
        Path to the Cellanome experiment directory.
    scan : int | str | None
        Specific scan to read (e.g., 1 or "scan_1").
        If None, reads all scans.
    lane : int | str | None
        Specific lane to read (e.g., 3 or "lane_3").
        If None, reads all lanes.
    """

    def __init__(
        self,
        root_dir: Path | str,
        scan: int | str | None = None,
        lane: int | str | None = None,
    ):
        self._root = Path(root_dir)
        self._scan_filter = self._parse_filter(scan, "scan")
        self._lane_filter = self._parse_filter(lane, "lane")

        # Load experiment metadata
        self._metadata = self._load_metadata()

        # Discover FOVs
        self._fovs: dict[str, CellanomeFOV] = {}
        self._channel_names: list[str] = []
        self._discover_fovs()

    def _parse_filter(self, value: int | str | None, prefix: str) -> int | None:
        """Parse scan/lane filter to integer."""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            if value.startswith(prefix + "_"):
                return int(value.split("_")[1])
            return int(value)
        raise ValueError(f"Invalid {prefix} filter: {value}")

    def _load_metadata(self) -> dict:
        """Load experiment metadata."""
        metadata_path = self._root / "experiment_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    def _discover_fovs(self):
        """Discover FOVs grouped by spatial position across all scans.

        Groups FOV images by spatial position (lane, row, fov_idx) rather
        than by scan+position. This results in one FOV per spatial position
        with T > 1 when multiple scans exist for that position.
        """
        # Find all scan directories
        scan_dirs = sorted(self._root.glob("scan_*"))

        # Nested grouping: spatial_position_key -> scan -> [CellanomeFOVInfo]
        position_scan_infos: dict[str, dict[int, list[CellanomeFOVInfo]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        all_channels: set[str] = set()

        for scan_dir in scan_dirs:
            scan_num = int(scan_dir.name.split("_")[1])
            if self._scan_filter is not None and scan_num != self._scan_filter:
                continue

            # Find lane directories
            lane_dirs = sorted(scan_dir.glob("lane_*"))

            for lane_dir in lane_dirs:
                lane_num = int(lane_dir.name.split("_")[1])
                if self._lane_filter is not None and lane_num != self._lane_filter:
                    continue

                # Find raw FOV PNG files (exclude .caging.png, .segmentation.png)
                png_files = [
                    f
                    for f in lane_dir.glob("*.png")
                    if not f.stem.endswith(".caging")
                    and not f.stem.endswith(".segmentation")
                    and not f.name.startswith("Fiducial")
                ]

                for png_file in png_files:
                    try:
                        info = CellanomeFOVInfo.from_filename(png_file)
                        # Use spatial_position_key (excludes scan) for grouping
                        spatial_pos_key = info.spatial_position_key
                        position_scan_infos[spatial_pos_key][scan_num].append(info)
                        all_channels.add(info.channel)
                    except ValueError as e:
                        # Skip files that don't match expected pattern
                        print(f"Skipping file: {e}")
                        continue

        # Filter channels: exclude .segmentation channels, sort alphabetically
        self._channel_names = sorted(
            c for c in all_channels if ".segmentation" not in c
        )

        # Create CellanomeFOV objects with multi-scan structure
        for pos_key, scan_infos in position_scan_infos.items():
            self._fovs[pos_key] = CellanomeFOV(
                position_key=pos_key,
                scan_infos=dict(scan_infos),  # Convert defaultdict to regular dict
                channel_names=self._channel_names,
            )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def channel_names(self) -> list[str]:
        return self._channel_names

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __enter__(self) -> "CellanomeReader":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        return False

    def __contains__(self, position_key: str) -> bool:
        return position_key in self._fovs

    def __len__(self) -> int:
        return len(self._fovs)

    def __getitem__(self, position_key: str) -> CellanomeFOV:
        return self._fovs[position_key]

    def __iter__(self) -> Iterable[tuple[str, CellanomeFOV]]:
        return iter(self._fovs.items())

    def close(self):
        """Close the reader (no-op for this reader)."""
        pass


def open_cellanome(
    path: Path | str,
    scan: int | str | None = None,
    lane: int | str | None = None,
) -> CellanomeReader:
    """Open a Cellanome dataset for reading.

    Parameters
    ----------
    path : Path | str
        Path to the Cellanome experiment directory.
    scan : int | str | None
        Specific scan to read (e.g., 1 or "scan_1").
    lane : int | str | None
        Specific lane to read (e.g., 3 or "lane_3").

    Returns
    -------
    CellanomeReader
        Reader object providing access to FOVs.

    Examples
    --------
    >>> reader = open_cellanome("/path/to/experiment")
    >>> for pos_key, fov in reader:
    ...     print(pos_key, fov.shape)
    >>> # Access specific FOV
    >>> fov = reader["1_3_1_15"]
    >>> data = np.asarray(fov[:])
    """
    return CellanomeReader(path, scan=scan, lane=lane)


# Quick test when run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "/hpc/instruments/cm.r3200/20251203141914_P-05_R000414_FC_BH_120325_try4_Adherent_with_SRA_training_4lanes"

    print(f"Opening Cellanome dataset: {path}")
    reader = open_cellanome(path, scan=1, lane=3)
    print(f"Found {len(reader)} FOVs")
    print(f"Channel names: {reader.channel_names}")

    # Print first few FOVs
    for i, (pos_key, fov) in enumerate(reader):
        print(f"  {pos_key}: shape={fov.shape}, dtype={fov.dtype}")
        if i >= 2:
            print("  ...")
            break
