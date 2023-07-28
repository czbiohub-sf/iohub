import warnings

import numpy as np
import zarr
from ndtiff import Dataset

from iohub.reader_base import ReaderBase


class NDTiffReader(ReaderBase):
    """Reader for ND-TIFF datasets acquired with Micro/Pycro-Manager,
    effectively a wrapper of the `ndtiff.Dataset` class.
    """

    def __init__(self, data_path: str):
        super().__init__()

        self.dataset = Dataset(data_path)
        self._axes = self.dataset.axes

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

        self.mm_meta = self._get_summary_metadata()
        self.channel_names = list(self.dataset.get_channel_names())
        self.stage_positions = self.mm_meta["Summary"]["StagePositions"]
        self.z_step_size = self.mm_meta["Summary"]["z-step_um"]
        self.xy_pixel_size = self.mm_meta["Summary"]["PixelSize_um"]

    def _get_summary_metadata(self):
        pm_metadata = self.dataset.summary_metadata
        pm_metadata["MicroManagerVersion"] = "pycromanager"
        pm_metadata["Positions"] = self.get_num_positions()
        img_metadata = self.get_image_metadata(0, 0, 0, 0)

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
            position_axis = self._axes["position"]

            for position in position_axis:
                position_metadata = {}
                img_metadata = self.get_image_metadata(position, 0, 0, 0)

                if all(
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

                if "PositionName" in img_metadata.keys():
                    position_metadata["Label"] = img_metadata["PositionName"]

                pm_metadata["StagePositions"].append(position_metadata)

        return {"Summary": pm_metadata}

    def _check_coordinates(self, p, t, c, z):
        coord_names = ("position", "time", "channel", "z")
        for coord, coord_name in zip((p, t, c, z), coord_names):
            if coord == 0 and coord_name not in self._axes.keys():
                coord = None
            elif coord not in self._axes[coord_name]:
                raise ValueError(
                    f"Image coordinate {coord_name} = {coord} is not part of"
                    "this dataset."
                )

        return p, t, c, z

    def get_num_positions(self) -> int:
        return (
            len(self._axes["position"])
            if "position" in self._axes.keys()
            else 1
        )

    def get_image(
        self, p: int or str, t: int, c: int or str, z: int
    ) -> np.ndarray:
        """return the image at the provided PTCZ coordinates

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
        np.ndarray
            numpy array of shape (Y, X) at given coordinate
        """

        image = None
        p, t, c, z = self._check_coordinates(p, t, c, z)

        if self.dataset.has_image(position=p, time=t, channel=c, z=z):
            image = self.dataset.read_image(position=p, time=t, channel=c, z=z)

        return image

    def get_zarr(self, position: int or str) -> zarr.array:
        """.. danger::
            The behavior of this function is different from other
            ReaderBase children as it return a Dask array
            rather than a zarr array.

        Return a lazy-loaded dask array with shape TCZYX at the given position.
        Data is not loaded into memory.


        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        position:       (zarr.array)

        """
        # TODO: try casting the dask array into a zarr array
        # using `dask.array.to_zarr()`.
        # Currently this call brings the data into memory
        if "position" in self._axes.keys():
            if position not in self._axes["position"]:
                raise ValueError(
                    f"Position index {position} is not part of this dataset."
                    f'Valid positions are: {self._axes["position"]}'
                )
        else:
            if position not in (0, None):
                warnings.warn(
                    f"Position index {position} is not part of this dataset. "
                    "Returning data at the default position."
                )
                position = None

        da = self.dataset.as_array(position=position)
        shape = (
            self.frames,
            self.channels,
            self.slices,
            self.height,
            self.width,
        )
        # add singleton axes so output is 5D
        return da.reshape(shape)

    def get_array(self, position: int or str) -> np.ndarray:
        """
        return a numpy array with shape TCZYX at the given position

        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        position:       (np.ndarray)

        """

        return np.asarray(self.get_zarr(position))

    def get_image_metadata(
        self, p: int or str, t: int, c: int or str, z: int
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
        p, t, c, z = self._check_coordinates(p, t, c, z)

        if self.dataset.has_image(position=p, time=t, channel=c, z=z):
            metadata = self.dataset.read_metadata(
                position=p, time=t, channel=c, z=z
            )

        return metadata
