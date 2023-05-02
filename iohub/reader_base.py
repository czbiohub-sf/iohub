import zarr
from numpy.typing import DTypeLike, NDArray


class ReaderBase:
    def __init__(self):
        self.frames: int = None
        self.channels: int = None
        self.slices: int = None
        self.height: int = None
        self.width: int = None
        self.dtype: DTypeLike = None
        self._mm_meta: dict = None
        self._stage_positions: list[dict] = []
        self.z_step_size: float = None
        self.channel_names: list[str] = None

    @property
    def shape(self):
        """Get the underlying data shape as a tuple.

        Returns
        -------
        tuple
            (frames, slices, channels, height, width)

        """
        return self.frames, self.channels, self.slices, self.height, self.width

    @property
    def mm_meta(self):
        return self._mm_meta

    @mm_meta.setter
    def mm_meta(self, value):
        if not isinstance(value, dict):
            raise TypeError(
                f"Type of `mm_meta` should be `dict`, got `{type(value)}`."
            )
        self._mm_meta = value

    @property
    def stage_positions(self):
        return self._stage_positions

    @stage_positions.setter
    def stage_positions(self, value):
        if not isinstance(value, list):
            raise TypeError(
                f"Type of `mm_meta` should be `dict`, got `{type(value)}`."
            )
        self._stage_positions = value

    def get_zarr(self, position: int) -> zarr.Array:
        """Get a zarr array for a given position.

        Parameters
        ----------
        position : int
            position (aka ome-tiff scene)

        Returns
        -------
        zarr.Array
        """
        raise NotImplementedError

    def get_array(self, position: int) -> NDArray:
        """Get a numpy array for a given position.

        Parameters
        ----------
        position : int
            position (aka ome-tiff scene)

        Returns
        -------
        NDArray
        """

    def get_image(self, p: int, t: int, c: int, z: int) -> NDArray:
        """Get the image slice at dimension P, T, C, Z.

        Parameters
        ----------
        p : int
            index of the position dimension
        t : int
            index of the time dimension
        c : int
            index of the channel dimension
        z : int
            index of the z dimension

        Returns
        -------
        NDArray
            2D image frame
        """
        raise NotImplementedError

    def get_num_positions(self) -> int:
        """Get total number of scenes referenced in ome-tiff metadata.

        Returns
        -------
        int
            number of positions
        """
        raise NotImplementedError
