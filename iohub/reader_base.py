import zarr
import numpy as np


class ReaderBase:

    frames = None
    channels = None
    slices = None
    height = None
    width = None
    mm_meta = None
    stage_positions = None
    z_step_size = None

    def get_zarr(self, position: int) -> zarr.array:
        pass

    def get_array(self, position: int) -> np.ndarray:
        pass

    def get_image(self, p, t, c, z) -> np.ndarray:
        pass

    def get_num_positions(self) -> int:
        pass
