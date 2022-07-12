import zarr
import numpy as np


class ReaderBase:

    def __init__(self):
        self.frames = None
        self.channels = None
        self.slices = None
        self.height = None
        self.width = None
        self.dtype = None
        self.mm_meta = None
        self.stage_positions = None
        self.z_step_size = None
        self.channel_names = None

    @property
    def shape(self):
        return self.frames, self.channels, self.slices, self.height, self.width

    def get_zarr(self, position: int) -> zarr.array:
        pass

    def get_array(self, position: int) -> np.ndarray:
        pass

    def get_image(self, p, t, c, z) -> np.ndarray:
        pass

    def get_num_positions(self) -> int:
        pass
