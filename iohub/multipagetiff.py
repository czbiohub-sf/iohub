from __future__ import annotations

import glob
import logging
import os
from copy import copy
from typing import TYPE_CHECKING, Iterable, Union

import dask.array as da
import numpy as np
from xarray import DataArray
import zarr
from numpy.typing import ArrayLike
from tifffile import TiffFile

from iohub.mm_fov import MicroManagerFOV, MicroManagerFOVMapping

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def _normalize_mm_pos_key(key: Union[str, int]) -> int:
    try:
        return int(key)
    except TypeError:
        raise TypeError("Micro-Manager position keys must be integers.")


class MMOmeTiffFOV(MicroManagerFOV):
    def __init__(self, parent: MMStack, key: int) -> None:
        super().__init__(parent, key)
        self._xdata = parent.xdata[key]

    @property
    def axes_names(self) -> list[str]:
        return list(self.xdata.dims)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return self.xdata.shape

    @property
    def dtype(self) -> np.dtype:
        return self.xdata.dtype

    @property
    def t_scale(self) -> float:
        return 1

    def __getitem__(
        self, key: Union[int, slice, tuple[Union[int, slice], ...]]
    ) -> ArrayLike:
        return self.xdata[key]

    @property
    def xdata(self) -> DataArray:
        return self._xdata


class MMStack(MicroManagerFOVMapping):
    """Micro-Manager multi-file OME-TIFF (MMStack) reader.

    Parameters
    ----------
    data_path : StrOrBytesPath
        Path to the directory containing OME-TIFF files
        or the path to the first OME-TIFF file in the series
    """

    def __init__(self, data_path: StrOrBytesPath):
        super().__init__()
        data_path = str(data_path)
        if os.path.isfile(data_path):
            if "ome.tif" in os.path.basename(data_path):
                first_file = data_path
            else:
                raise ValueError("{data_path} is not a OME-TIFF file.")
        elif os.path.isdir(data_path):
            files = glob.glob(os.path.join(data_path, "*.ome.tif"))
            if not files:
                raise FileNotFoundError(
                    f"Path {data_path} contains no OME-TIFF files, "
                )
            else:
                first_file = files[0]
        self.root = os.path.dirname(first_file)
        self.dirname = os.path.basename(self.root)
        self._first_tif = TiffFile(first_file, is_mmstack=True)
        self._parse_data()
        self._store = None

    def _parse_data(self):
        series = self._first_tif.series[0]
        raw_dims = dict(
            (axis, size)
            for axis, size in zip(series.get_axes(), series.get_shape())
        )
        axes = ("R", "T", "C", "Z", "Y", "X")
        dims = dict((ax, raw_dims.get(ax, 1)) for ax in axes)
        logging.debug(f"Got dataset dimensions from tifffile: {dims}.")
        (
            self.positions,
            self.frames,
            self.channels,
            self.slices,
            self.height,
            self.width,
        ) = dims.values()
        self._store = series.aszarr()
        logging.debug(f"Opened {self._store}.")
        data = da.from_zarr(zarr.open(self._store))
        img = DataArray(data, dims=raw_dims, name=self.dirname)
        xarr = img.expand_dims(
            [ax for ax in axes if ax not in img.dims]
        ).transpose(*axes)
        xset = xarr.to_dataset(dim="R")
        self._xdata = xset
        self._set_mm_meta

    @property
    def xdata(self):
        return self._xdata

    def __len__(self) -> int:
        return self.positions

    def __getitem__(self, key: Union[str, int]) -> MMOmeTiffFOV:
        key = _normalize_mm_pos_key(key)
        return MMOmeTiffFOV(self, key)

    def __setitem__(self, key, value) -> None:
        raise PermissionError("MMStack is read-only.")

    def __delitem__(self, key, value) -> None:
        raise PermissionError("MMStack is read-only.")

    def __contains__(self, key: Union[str, int]) -> bool:
        key = _normalize_mm_pos_key(key)
        return key in self.xdata

    def __iter__(self) -> Iterable[tuple[str, MMOmeTiffFOV]]:
        for key in self.xdata:
            yield key, self[key]

    def __enter__(self) -> MMStack:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close file handles"""
        self._first_tif.close()

    def _set_mm_meta(self) -> None:
        """Assign image metadata from summary metadata."""
        self.mm_meta = self._first_tif.micromanager_metadata
        mm_version = self.mm_meta["Summary"]["MicroManagerVersion"]
        if "beta" in mm_version:
            if self.mm_meta["Summary"]["Positions"] > 1:
                self._stage_positions = []

                for p in range(
                    len(self.mm_meta["Summary"]["StagePositions"])
                ):
                    pos = self._simplify_stage_position_beta(
                        self.mm_meta["Summary"]["StagePositions"][p]
                    )
                    self._stage_positions.append(pos)

            # MM beta versions sometimes don't have 'ChNames',
            # so I'm wrapping in a try-except and setting the
            # channel names to empty strings if it fails.
            try:
                for ch in self.mm_meta["Summary"]["ChNames"]:
                    self.channel_names.append(ch)
            except KeyError:
                self.channel_names = self.mm_meta["Summary"]["Channels"] * [
                    ""
                ]  # empty strings

        elif mm_version == "1.4.22":
            for ch in self.mm_meta["Summary"]["ChNames"]:
                self.channel_names.append(ch)

            else:
                if self.mm_meta["Summary"]["Positions"] > 1:
                    self._stage_positions = []

                    for p in range(self.mm_meta["Summary"]["Positions"]):
                        pos = self._simplify_stage_position(
                            self.mm_meta["Summary"]["StagePositions"][p]
                        )
                        self._stage_positions.append(pos)

            for ch in self.mm_meta["Summary"]["ChNames"]:
                self.channel_names.append(ch)

        self._z_step_size = self.mm_meta["Summary"]["z-step_um"]

    def _simplify_stage_position(self, stage_pos: dict) -> dict:
        """
        flattens the nested dictionary structure of stage_pos
        and removes superfluous keys

        Parameters
        ----------
        stage_pos:      (dict)
            dictionary containing a single position's device info

        Returns
        -------
        out:            (dict)
            flattened dictionary
        """

        out = copy(stage_pos)
        out.pop("DevicePositions")
        for dev_pos in stage_pos["DevicePositions"]:
            out.update({dev_pos["Device"]: dev_pos["Position_um"]})
        return out

    def _simplify_stage_position_beta(self, stage_pos: dict) -> dict:
        """
        flattens the nested dictionary structure of stage_pos
        and removes superfluous keys
        for MM2.0 Beta versions

        Parameters
        ----------
        stage_pos:      (dict)
            dictionary containing a single position's device info

        Returns
        -------
        new_dict:       (dict)
            flattened dictionary

        """

        new_dict = {}
        new_dict["Label"] = stage_pos["label"]
        new_dict["GridRow"] = stage_pos["gridRow"]
        new_dict["GridCol"] = stage_pos["gridCol"]

        for sub in stage_pos["subpositions"]:
            values = []
            for field in ["x", "y", "z"]:
                if sub[field] != 0:
                    values.append(sub[field])
            if len(values) == 1:
                new_dict[sub["stageName"]] = values[0]
            else:
                new_dict[sub["stageName"]] = values

        return new_dict

    def _infer_image_meta(self) -> None:
        """
        Infer data type and pixel size from the first image plane metadata.
        """
        page = self._first_tif.pages[0]
        self.dtype = page.dtype
        for tag in page.tags.values():
            if tag.name == "MicroManagerMetadata":
                # assuming X and Y pixel sizes are the same
                xy_size = tag.value.get("PixelSizeUm")
                self._xy_pixel_size = xy_size if xy_size else None
                return
            else:
                continue
        logging.warning(
            "Micro-Manager image plane metadata cannot be loaded."
        )
        self._xy_pixel_size = None

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        """ZXY pixel size in micrometers."""
        if self._xy_pixel_size is None:
            raise AttributeError("XY pixel size cannot be determined.")
        return (
            float(v)
            for v in (
                self._z_step_size,
                self._xy_pixel_size,
                self._xy_pixel_size,
            )
        )


class MicromanagerOmeTiffReader:
    # FIXME: delete this. It's kept for now to avoid import error.
    pass
