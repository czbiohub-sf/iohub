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

        if not os.path.isfile(zarrfile):
            raise ValueError("file is not a .zarr file")
        if not '.zarr' in zarrfile:
            raise ValueError("file is not a .zarr file")

        self.zf = zarrfile
        self.z = zarr.open(self.zf, 'r')

        # structure of zarr array
        (self.pos,
         self.frames,
         self.channels,
         self.slices,
         self.height,
         self.width) = self.z.shape

        self.positions = {}
        for p in range(self.pos):
            self.positions[p] = self.z[p]

    def get_zarr(self, position: int) -> zarr.array:
        return self.positions[position]

    def get_array(self, position: int) -> np.ndarray:
        return np.array(self.positions[position])

    def get_num_positions(self) -> int:
        return len(self.positions)

    @property
    def shape(self):
        return self.frames, self.channels, self.slices, self.height, self.width

