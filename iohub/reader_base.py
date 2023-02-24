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
        self.mm_meta: dict = None
        self.stage_positions: list[dict] = None
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
        pass

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

        Returns:
        -------
        int
            number of positions
        """
        raise NotImplementedError
