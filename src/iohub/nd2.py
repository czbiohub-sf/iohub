from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from nd2 import ND2File
from numpy.typing import ArrayLike
from xarray import DataArray
from xarray import Dataset as XDataset

from iohub.mm_fov import MicroManagerFOV, MicroManagerFOVMapping

__all__ = ["ND2FOV", "ND2Dataset"]
_logger = logging.getLogger(__name__)

# Canonical (P, T, C, Z, Y, X) axis order. nd2 uses these same single-letter
# dim names (see ``nd2.AXIS``); RGB ("S") is not supported.
_ND2_AXES = ("P", "T", "C", "Z", "Y", "X")


class ND2FOV(MicroManagerFOV):
    def __init__(self, parent: ND2Dataset, key: str) -> None:
        super().__init__(parent, key)
        self._xdata = parent.xdata[key]

    @property
    def axes_names(self) -> list[str]:
        return list(self._xdata.dims)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return self._xdata.shape

    @property
    def dtype(self) -> np.dtype:
        return self._xdata.dtype

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> ArrayLike:
        return self._xdata[key]

    @property
    def xdata(self) -> DataArray:
        return self._xdata

    def frame_metadata(self, t: int, c: int, z: int) -> dict | None:
        # Per-frame metadata export is not implemented for ND2 (v1).
        return None


class ND2Dataset(MicroManagerFOVMapping):
    """Reader for Nikon ND2 datasets, wrapping :class:`nd2.ND2File`.

    Each ND2 stage position (``P``) is exposed as a separate FOV. Files
    without a position axis are read as a single FOV. Data are returned in
    canonical ``(T, C, Z, Y, X)`` order with native dtype preserved.
    """

    def __init__(self, data_path: Path | str):
        super().__init__()
        data_path = Path(data_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"{data_path} is not a valid ND2 file.")
        self._file = ND2File(str(data_path))
        self._root = data_path
        self.dirname = data_path.name

        sizes = dict(self._file.sizes)
        if "S" in sizes:
            raise NotImplementedError("RGB ND2 files (axis 'S') are not supported.")
        self.frames = sizes.get("T", 1)
        self.channels = sizes.get("C", 1)
        self.slices = sizes.get("Z", 1)
        self.height = sizes["Y"]
        self.width = sizes["X"]
        self.dtype = np.dtype(self._file.dtype)
        self._n_pos = sizes.get("P", 1)

        self.channel_names = self._parse_channel_names()
        # No plate/well metadata read in v1 -> converter falls back to a
        # single-row grid layout. Populated empty to satisfy the base class.
        self._mm_meta = {}
        self.stage_positions = []

        voxel = self._file.voxel_size()  # (x, y, z) in micrometers
        self._zyx_scale = (
            float(voxel.z or 1.0),
            float(voxel.y or 1.0),
            float(voxel.x or 1.0),
        )
        self._t_scale = self._parse_t_scale()
        self._gather_xdata()

    def __enter__(self) -> ND2Dataset:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __iter__(self) -> Iterable[tuple[str, ND2FOV]]:
        for key in self._xdata.keys():
            key = str(key)
            yield key, ND2FOV(self, key)

    def __contains__(self, key: str | int) -> bool:
        return str(key) in self._xdata

    def __len__(self) -> int:
        return self._n_pos

    def __getitem__(self, key: int | str) -> ND2FOV:
        return ND2FOV(self, str(key))

    def close(self) -> None:
        self._file.close()

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        return self._zyx_scale

    @property
    def t_scale(self) -> float:
        return self._t_scale

    @property
    def xdata(self) -> XDataset:
        return self._xdata

    def _parse_channel_names(self) -> list[str]:
        try:
            channels = self._file.metadata.channels
            if channels:
                return [c.channel.name for c in channels]
        except (AttributeError, TypeError):
            pass
        _logger.warning("No channel names found in metadata. Using defaults.")
        return [f"Channel{i}" for i in range(self.channels)]

    def _parse_t_scale(self) -> float:
        """Time interval in seconds, from the first time loop; 1.0 if absent."""
        try:
            for loop in self._file.experiment:
                period_ms = getattr(getattr(loop, "parameters", None), "periodMs", None)
                if period_ms:
                    return float(period_ms) / 1000.0
        except (AttributeError, TypeError):
            pass
        _logger.warning("No time interval found in metadata. Returning 1.0 as a placeholder.")
        return 1.0

    def _gather_xdata(self) -> None:
        # Labeled, dask-backed DataArray; squeeze=False keeps all present axes.
        da = self._file.to_xarray(delayed=True, squeeze=False)
        # Pad missing leading axes so output is always (P, T, C, Z, Y, X).
        for dim in ("P", "T", "C", "Z"):
            if dim not in da.dims:
                da = da.expand_dims(dim)
        da = da.transpose(*_ND2_AXES)
        pkeys = [str(i) for i in range(self._n_pos)]
        da = da.assign_coords(P=pkeys, C=self.channel_names)
        da.name = self.dirname
        self._xdata = da.to_dataset(dim="P")
