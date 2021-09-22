import numpy as np
import os
import re
from copy import copy
import tifffile as tiff
import glob
from waveorder.io.reader_interface import ReaderInterface


class UPTIReader(ReaderInterface):

    """
    Reader for HCS ome-zarr arrays.  OME-zarr structure can be found here: https://ngff.openmicroscopy.org/0.1/
    Also collects the HCS metadata so it can be later copied.
    """

    def __init__(self, folder: str, extract_data: bool = False):

        # zarr files (.zarr) are directories
        if not os.path.isdir(folder):
            raise ValueError("folder does not exist")

        self.data_folder = folder
        self.files = glob.glob(os.path.join(folder, '*.tif'))
        info_img = tiff.imread(self.files[0]).dtype
        self.dtype = info_img.dtype
        (self.height, self.width) = info_img.shape
        self.positions = 1
        self.frames = 1
        self.patterns = 0
        self.states = 0
        self.slices = 0
        self.channel_names = []
        self._map_files()
        self.channels = len(self.channel_names)

        # structure of zarr array
        self.stage_positions = 1
        self.z_step_size = None
        self.position_map = ()

        # initialize metadata
        self.mm_meta = None
        if extract_data:
            self._create


    def _map_files(self):
        self.file_map = dict()

        states = True if 'State' in self.files[0] else False
        for file in self.files:
            pattern = re.search('pattern_\d\d\d', file).group(0)
            pattern_idx = int(pattern.strip('pattern_'))
            z = int(re.search('z_\d\d\d', file).group(0).strip('z_'))
            state_idx = 0

            if pattern_idx > self.patterns:
                self.patterns = pattern_idx

            if z > self.slices:
                self.slices = z

            if states:
                state = re.search('State_\d\d\d', file).group(0)
                state_idx = int(state.strip('State_'))

                if f'{pattern}_{state}' not in self.channel_names:
                    self.channel_names.append(pattern)
            else:
                if pattern not in self.channel_names:
                    self.channel_names.append(pattern)

            self.file_map[(pattern_idx, state_idx, z)] = file

            self.channel_names.sort(key=lambda x: re.sub('State_\d\d\d', '', x)) # sorts pattern first

    def get_zarr(self, position=0):
        """
        Returns the position-level zarr group

        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        (ZarrGroup) Position subgroup containing the array group+data

        """
        if position > self.positions - 1:
            raise ValueError('Entered position is greater than the number of positions in the data')

        pos_info = self.position_map[position]
        well = pos_info['well']
        pos = pos_info['name']
        return self.store[well][pos]

    def get_array(self, position):
        """
        Gets the (T, C, Z, Y, X) array at given position

        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        (ZarrArray) Zarr array of size (T, C, Z, Y, X) at specified position

        """
        pos = self.get_zarr(position)
        return pos['array']

    def get_num_positions(self) -> int:
        return self.positions

    @property
    def shape(self):
        return self.frames, self.channels, self.slices, self.height, self.width

