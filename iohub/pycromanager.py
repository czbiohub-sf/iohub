import warnings
import numpy as np
import zarr
from iohub.reader_base import ReaderBase
from pycromanager import Dataset


class PycromanagerReader(ReaderBase):
    def __init__(self, data_path: str):
        super().__init__()

        """
        Reader for data acquired with pycromanager, effectively a wrapper of the pycromanager.Dataset class

        """

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

    def _get_summary_metadata(self):
        pm_metadata = self.dataset.summary_metadata
        pm_metadata["MicroManagerVersion"] = "pycromanager"
        pm_metadata["Positions"] = self.get_num_positions()

        img_metadata = self.get_image_metadata(0, 0, 0, 0)
        pm_metadata["z-step_um"] = None
        pm_metadata["StagePositions"] = []

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

        if "XPosition_um_Intended" in img_metadata.keys():
            for p in range(self.get_num_positions()):
                img_metadata = self.get_image_metadata(p, 0, 0, 0)
                pm_metadata["StagePositions"].append(
                    {
                        img_metadata["Core-XYStage"]: (
                            img_metadata["XPosition_um_Intended"],
                            img_metadata["YPosition_um_Intended"],
                        )
                    }
                )

        return {"Summary": pm_metadata}

    def _check_coordinates(self, p, t, c, z):
        if p == 0 and "position" not in self._axes.keys():
            p = None
        if t == 0 and "time" not in self._axes.keys():
            t = None
        if c == 0 and "channel" not in self._axes.keys():
            c = None
        if z == 0 and "z" not in self._axes.keys():
            z = None

        return p, t, c, z

    def get_num_positions(self) -> int:
        return (
            len(self._axes["position"])
            if "position" in self._axes.keys()
            else 1
        )

    def get_image(self, p, t, c, z) -> np.ndarray:
        """
        return the image at the provided PTCZ coordinates

        Parameters
        ----------
        p:              (int) position index
        t:              (int) time index
        c:              (int) channel index
        z:              (int) slice/z index

        Returns
        -------
        image:          (np-array) numpy array of shape (Y, X) at given coordinate

        """

        image = None
        p, t, c, z = self._check_coordinates(p, t, c, z)

        if self.dataset.has_image(position=p, time=t, channel=c, z=z):
            image = self.dataset.read_image(position=p, time=t, channel=c, z=z)

        return image

    def get_zarr(self, position: int) -> zarr.array:
        """
        return a lazy-loaded dask array with shape TCZYX at the given position.
        Data is not loaded into memory.

        Note: The behavior of this function is different from other WaveorderReaders
         as it return a Dask array rather than a zarr array.

        # TODO: try casting the dask array into a zarr array using dask.array.to_zarr().
        # Currently this call brings the data into memory

        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        position:       (zarr.array)

        """

        ax = [
            ax_
            for ax_ in ["position", "time", "channel", "z"]
            if ax_ in self._axes
        ]

        if "position" in self._axes.keys():
            # da is Dask array
            da = self.dataset.as_array(axes=ax, position=position)
        else:
            if position not in (0, None):
                warnings.warn(
                    f"Position index {position} is not part of this dataset."
                    f" Returning data at default position."
                )
            da = self.dataset.as_array(axes=ax)

        shape = (
            self.frames,
            self.channels,
            self.slices,
            self.height,
            self.width,
        )

        return da.reshape(shape)

    def get_array(self, position: int) -> np.ndarray:
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

    def get_image_metadata(self, p, t, c, z) -> dict:
        """
        return image plane metadata at the requested PTCZ coordinates

        Parameters
        ----------
        p:              (int) position index
        t:              (int) time index
        c:              (int) channel index
        z:              (int) slice/z index

        Returns
        -------
        metadata:       (dict) image plane metadata dictionary

        """
        metadata = None
        p, t, c, z = self._check_coordinates(p, t, c, z)

        if self.dataset.has_image(position=p, time=t, channel=c, z=z):
            metadata = self.dataset.read_metadata(
                position=p, time=t, channel=c, z=z
            )

        return metadata
