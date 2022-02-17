import zarr
import numpy as np

class ReaderBase:

    def get_zarr(self, position: int) -> zarr.array:
        pass

    def get_array(self, position: int) -> np.ndarray:
        pass

    def get_image(self, p, t, c, z) -> np.ndarray:
        pass

    def get_num_positions(self) -> int:
        pass
