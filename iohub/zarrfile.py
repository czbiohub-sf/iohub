import numpy as np
import os
import zarr

from waveorder.io.reader_interface import ReaderInterface


"""
This reader is written to load data that has been directly streamed from micro-manager
    to a zarr array.  The array dims order is based on the micro-manager order
    
by default supports READ ONLY MODE
"""


class ZarrReader(ReaderInterface):

    def __init__(self,
                 zarrfile: str):

        # zarr files (.zarr) are directories
        if not os.path.isdir(zarrfile):
            raise ValueError("file is not a .zarr file")
        # if not '.zarr' in zarrfile:
        #     raise ValueError("file is not a .zarr file")

        self.zf = zarrfile
        self.z = zarr.open(self.zf, 'r', chunks=(1, 1, 1, 1, 2048, 2048))

        # structure of zarr array
        (self.pos,
         self.frames,
         self.channels,
         self.slices,
         self.height,
         self.width) = self.z.shape

        # self.positions = {}
        # for p in range(self.pos):
        #     self.positions[p] = self.z[p]

    def get_zarr(self, pt: tuple) -> zarr.array:
        # return self.positions[position]
        return self.z[pt[0], pt[1]].astype(np.uint16)

    def get_array(self, pt: tuple) -> np.ndarray:
        # return np.array(self.positions[position])
        return np.array(self.z[pt[0], pt[1]]).astype(np.uint16)

    def get_num_positions(self) -> int:
        # return len(self.positions)
        return len(self.z)

    @property
    def shape(self):
        return self.frames, self.channels, self.slices, self.height, self.width

