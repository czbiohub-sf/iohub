import numpy as np
import os
import re
import zarr
import tifffile as tiff
import glob
from waveorder.io.reader_interface import ReaderInterface


class UPTIReader(ReaderInterface):

    """
    Reader for UPTI raw data.  Accepts both new live UPTI and older UPTI format.
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
        self.z_step_size = None

        # initialize metadata
        self.mm_meta = None
        self.position_arrays = dict()
        if extract_data:
            for i in range(self.positions):
                self._create_position_array(i)

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

    def _create_position_array(self, pos):
        """
        maps all of the tiff data into a virtual zarr store in memory for a given position

        Parameters
        ----------
        pos:            (int) index of the position to create array under

        Returns
        -------

        """

        self.position_arrays[pos] = zarr.empty(shape=(self.frames, self.channels, self.slices, self.height, self.width),
                                               chunks=(1, 1, 1, self.height, self.width),
                                               dtype=self.dtype)
        # add all the images with this specific dimension.  Will be blank images if dataset
        # is incomplete
        for t in range(self.frames):
            for c in range(self.channels):
                for z in range(self.slices):
                    self.position_arrays[pos][t, c, z] = self._get_image(c, z)


    def _get_image(self, c, z):

        chan_name = self.channel_names[c]
        pattern = int(re.search('pattern_\d\d\d', chan_name).group(0).strip('pattern_'))
        state = re.search('State_\d\d\d', chan_name)
        state_idx = 0 if not state else int(state.group(0).strip('State_'))

        fn = self.file_map[(pattern, state_idx, z)]
        img = zarr.open(tiff.imread(fn, aszarr=True), 'r')

        return img

    def get_zarr(self, position=0):
        """
        return a zarr array for a given position

        Parameters
        ----------
        position:       (int) position (aka ome-tiff scene)

        Returns
        -------
        position:       (zarr.array)

        """
        if position > self.positions - 1:
            raise ValueError('Entered position is greater than the number of positions in the data')

        if position not in self.position_arrays.keys():
            self._create_position_array(position)
        return self.position_arrays[position]

    def get_array(self, position=0):
        """
        return a numpy array for a given position

        Parameters
        ----------
        position:   (int) position (aka ome-tiff scene)

        Returns
        -------
        position:   (np.ndarray)

        """

        if position > self.positions - 1:
            raise ValueError('Entered position is greater than the number of positions in the data')

        # if position hasn't been initialized in memory, do that.
        if position not in self.position_arrays.keys():
            self._create_position_array(position)

        return np.array(self.position_arrays[position])

    def get_num_positions(self) -> int:
        return self.positions

    @property
    def shape(self):
        return self.frames, self.channels, self.slices, self.height, self.width

