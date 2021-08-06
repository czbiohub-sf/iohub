import zarr
import numpy as np

class ReaderInterface:

    def get_zarr(self, position: int) -> zarr.array:
        pass

    def get_array(self, position: int) -> np.ndarray:
        pass

    def get_num_positions(self) -> int:
        pass

    # def mm_meta(self) -> int:
    #     return
    #
    # def stage_positions(self):
    #     pass
    #
    # def z_step_size(self):
    #     pass
    #
    # def height(self):
    #     pass
    #
    # def width(self):
    #     pass
    #
    # def frames(self):
    #     pass
    #
    # def slices(self):
    #     pass
    #
    # def channels(self):
    #     pass
    #
    # def channel_names(self):
    #     pass