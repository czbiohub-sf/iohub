from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from natsort import natsorted
from ndtiff import Dataset
from numpy.typing import ArrayLike
from xarray import DataArray
from xarray import Dataset as XDataset

from iohub.mm_fov import MicroManagerFOV, MicroManagerFOVMapping


class NDTiffFOV(MicroManagerFOV):
    def __init__(self, parent: NDTiffDataset, key: int) -> None:
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

    @property
    def t_scale(self) -> float:
        return 1.0

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> ArrayLike:
        return self._xdata[key]

    @property
    def xdata(self) -> DataArray:
        return self._xdata

    def frame_metadata(self, t: int, c: int, z: int) -> dict[str, Any]:
        return self.parent.get_image_metadata(self._position, t, c, z)


class NDTiffDataset(MicroManagerFOVMapping):
    """Reader for ND-TIFF datasets acquired with Micro/Pycro-Manager,
    effectively a wrapper of the `ndtiff.Dataset` class.
    """

    def __init__(self, data_path: Path | str):
        super().__init__()
        data_path = Path(data_path)
        if not data_path.is_dir():
            raise FileNotFoundError(
                f"{data_path} is not a valid NDTiff dataset."
            )
        self.dataset = Dataset(str(data_path))
        self.root = data_path
        self.dirname = data_path.name
        self._axes = self.dataset.axes
        self._str_posistion_axis = self._check_str_axis("position")
        self._str_channel_axis = self._check_str_axis("channel")
        self.frames = (
            len(self._axes["time"]) if "time" in self._axes.keys() else 1
        )
        self.channels = (
            len(self._axes["channel"]) if "channel" in self._axes.keys() else 1
        )
        self.slices = len(self._axes["z"]) if "z" in self._axes.keys() else 1
        self.height = self.dataset.image_height
        self.width = self.dataset.image_width
        self.dtype = self.dataset.dtype

        self._all_position_keys = self._parse_all_position_keys()
        self._mm_meta = self._get_summary_metadata()
        self.channel_names = list(self.dataset.get_channel_names())
        self.stage_positions = self._mm_meta["Summary"]["StagePositions"]
        z_step_size = float(self._mm_meta["Summary"]["z-step_um"] or 1.0)
        xy_pixel_size = float(self._mm_meta["Summary"]["PixelSize_um"] or 1.0)
        self.zyx_scale = (z_step_size, xy_pixel_size, xy_pixel_size)
        self._gather_xdata()

    def __enter__(self) -> NDTiffDataset:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __iter__(self) -> Iterable[tuple[str, NDTiffFOV]]:
        for key in self.xdata.keys():
            key = str(key)
            yield key, NDTiffFOV(self, key)

    def __contains__(self, key: str | int) -> bool:
        return str(key) in self._xdata

    def __len__(self) -> int:
        return len(self._all_position_keys)

    def __getitem__(self, key: int | str) -> NDTiffFOV:
        return NDTiffFOV(self, str(key))

    def close(self) -> None:
        self.dataset.close()

    def _get_summary_metadata(self):
        pm_metadata = self.dataset.summary_metadata
        pm_metadata["MicroManagerVersion"] = "pycromanager"
        pm_metadata["Positions"] = len(self)
        img_metadata = self.get_image_metadata(
            self._all_position_keys[0], 0, 0, 0
        )

        pm_metadata["z-step_um"] = None
        if "ZPosition_um_Intended" in img_metadata.keys():
            pm_metadata["z-step_um"] = np.around(
                abs(
                    self.get_image_metadata(0, 0, 0, 1)[
                        "ZPosition_um_Intended"
                    ]
                    - self.get_image_metadata(0, 0, 0, 0)[
                        "ZPosition_um_Intended"
                    ]
                ),
                decimals=3,
            ).astype(float)

        pm_metadata["StagePositions"] = []
        if "position" in self._axes:
            for position in self._axes["position"]:
                position_metadata = {}
                img_metadata = self.get_image_metadata(position, 0, 0, 0)

                if img_metadata is not None and all(
                    key in img_metadata.keys()
                    for key in [
                        "XPosition_um_Intended",
                        "YPosition_um_Intended",
                    ]
                ):
                    position_metadata[img_metadata["Core-XYStage"]] = (
                        img_metadata["XPosition_um_Intended"],
                        img_metadata["YPosition_um_Intended"],
                    )

                # Position label may also be obtained from "PositionName"
                # metadata key with AcqEngJ >= 0.29.0
                if isinstance(position, str):
                    position_metadata["Label"] = position

                pm_metadata["StagePositions"].append(position_metadata)

        return {"Summary": pm_metadata}

    def _check_str_axis(self, axis: Literal["position", "channel"]) -> bool:
        if axis in self._axes:
            coord_sample = next(iter(self._axes[axis]))
            return isinstance(coord_sample, str)
        else:
            return False

    @property
    def str_position_axis(self) -> bool:
        """Position axis is string-valued"""
        return self._str_posistion_axis

    @property
    def str_channel_axis(self) -> bool:
        """Channel axis is string-valued"""
        return self._str_channel_axis

    @property
    def ndtiff_axes(self) -> tuple[str]:
        return ("position", "time", "channel", "z", "y", "x")

    def _check_coordinates(self, p: int | str, t: int, c: int | str, z: int):
        """
        Check that the (p, t, c, z) coordinates are part of the ndtiff dataset.
        Replace coordinates with None or string values in specific cases - see
        below
        """
        coords = [p, t, c, z]
        axes = self.ndtiff_axes[:4]

        for i, axis in enumerate(axes):
            coord = coords[i]

            # Check if the axis is part of the dataset axes
            if axis in self._axes.keys():
                # Check if coordinate is part of the dataset axis
                if coord in self._axes[axis]:
                    # all good
                    pass

                # The requested coordinate is not part of the axis
                else:
                    # If coord=0 is requested and the coordinate axis exists,
                    # but is string valued (e.g. {'Pos0', 'Pos1'}), a warning
                    # will be raised and the coordinate will be replaced by a
                    # random sample.

                    # Coordinates are in sets, here we get one sample from the
                    # set without removing it:
                    # https://stackoverflow.com/questions/59825
                    coord_sample = next(iter(self._axes[axis]))
                    if coord == 0 and isinstance(coord_sample, str):
                        coords[i] = coord_sample
                        warnings.warn(
                            f"Indices of {axis} are string-valued. "
                            f"Returning data at {axis} = {coord}"
                        )
                    else:
                        # If the coordinate is not part of the axis and
                        # nonzero, a ValueError will be raised
                        raise ValueError(
                            f"Image coordinate {axis} = {coord} is not "
                            "part of this dataset."
                        )

            # The axis is not part of the dataset axes
            else:
                # Nothing to do if coord == None
                if coord is not None:
                    # If coord = 0 is requested, the coordinate will be
                    # replaced with None
                    if coord == 0:
                        coords[i] = None
                    # If coord != 0 is requested and the axis is not part of
                    # the dataset, ValueError will be raised
                    else:
                        raise ValueError(
                            f"Axis {axis} is not part of this dataset"
                        )

        return tuple(coords)

    def _parse_all_position_keys(self) -> list[int | str | None]:
        return natsorted(list(self._axes.get("position", [None])))

    def _check_position_key(self, key: int | str) -> bool:
        if "position" in self._axes.keys():
            if key not in self._axes["position"]:
                raise ValueError(
                    f"Position index {key} is not part of this dataset. "
                    f'Valid positions are: {self._axes["position"]}'
                )
        else:
            if key not in (0, None):
                warnings.warn(
                    f"Position index {key} is not part of this dataset. "
                    "Returning data at the default position."
                )
                key = None
        return key

    def _gather_xdata(self) -> None:
        shape = (
            len(self),
            self.frames,
            self.channels,
            self.slices,
            self.height,
            self.width,
        )
        if self._all_position_keys == [None]:
            pkeys = ["0"]
        else:
            pkeys = [str(k) for k in self._all_position_keys]
        # add singleton axes so output is always (p, t, c, z, y, x)
        da = DataArray(
            self.dataset.as_array().reshape(shape),
            dims=self.ndtiff_axes,
            name=self.dirname,
        ).assign_coords(position=pkeys)
        self._xdata = da.to_dataset(dim="position")

    @property
    def xdata(self) -> XDataset:
        return self._xdata

    def get_image_metadata(
        self, p: int | str, t: int, c: int | str, z: int
    ) -> dict:
        """Return image plane metadata at the requested PTCZ coordinates

        Parameters
        ----------
        p : int or str
            position index
        t : int
            time index
        c : int or str
            channel index
        z : int
            slice/z index

        Returns
        -------
        dict
            image plane metadata
        """
        metadata = None
        if not self.str_position_axis and isinstance(p, str):
            if p.isdigit():
                p = int(p)
        p, t, c, z = self._check_coordinates(p, t, c, z)
        if self.dataset.has_image(position=p, time=t, channel=c, z=z):
            metadata = self.dataset.read_metadata(
                position=p, time=t, channel=c, z=z
            )
        return metadata
