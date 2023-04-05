import glob
import os
import re

import numpy as np
import tifffile as tiff
import zarr

from iohub.reader_base import ReaderBase


class UPTIReader(ReaderBase):

    """
    Reader for UPTI raw data.
    Accepts both new live UPTI and older UPTI format.
    """

    def __init__(self, folder: str, extract_data: bool = False):
        # check if folder exists
        if not os.path.isdir(folder):
            raise ValueError("folder does not exist")

        # Initializate data parameters
        self.data_folder = folder
        if glob.glob(os.path.join(folder, "*.tif")) == []:
            self.files = glob.glob(os.path.join(folder, "*.tiff"))
        else:
            self.files = glob.glob(os.path.join(folder, "*.tif"))
        info_img = tiff.imread(self.files[0])
        self.dtype = info_img.dtype
        (self.height, self.width) = info_img.shape
        self.positions = 1
        self.frames = 1
        self.patterns = 0
        self.states = 0
        self.slices = 0
        self.channel_names = []

        # map files and update data parameters
        self._map_files()
        self.channels = len(self.channel_names)
        self.z_step_size = None

        # initialize metadata
        self.position_arrays = dict()

        if extract_data:
            for i in range(self.positions):
                self._create_position_array(i)

    def _map_files(self):
        """
        Creates a map of (pattern, state, slice) coordinate
        with corresponding file.
        Uses regex parsing and assumes a consistent file naming structure.

        Returns
        -------

        """

        self.file_map = dict()

        states = True if "State" in self.files[0] else False

        # Loop through the files and use regex to grab patterns and states,
        # **hardcoded file name structure**
        for file in self.files:
            pattern = re.search(r"pattern_\d\d\d", file).group(0)
            pattern_idx = int(pattern.strip("pattern_"))
            if re.search(r"z_\d\d\d", file) is None:
                z = 0
            else:
                z = int(re.search(r"z_\d\d\d", file).group(0).strip("z_"))
            state_idx = 0

            # update dimensionality as we read the file names
            if pattern_idx + 1 > self.patterns:
                self.patterns = pattern_idx + 1

            if z + 1 > self.slices:
                self.slices = z + 1

            # if live upti, grab the states, otherwise just use patterns
            if states:
                state = re.search(r"State_\d\d\d", file).group(0)
                state_idx = int(state.strip("State_"))

                if f"{pattern}_{state}" not in self.channel_names:
                    self.channel_names.append(f"{pattern}_{state}")
            else:
                if pattern not in self.channel_names:
                    self.channel_names.append(pattern)

            self.file_map[(pattern_idx, state_idx, z)] = file

            # sorts list by pattern 0 --> state_0, state_1, state_2, state_3
            self.channel_names.sort(
                key=lambda x: re.sub(r"pattern_\d\d\d", "", x)
            )
            self.channel_names.sort(
                key=lambda x: re.sub(r"State_\d\d\d", "", x)
            )

    def _create_position_array(self, pos):
        """
        maps all of the tiff data into a virtual zarr store in memory
        for a given position

        Parameters
        ----------
        pos:            (int) index of the position to create array under

        Returns
        -------

        """

        self.position_arrays[pos] = zarr.zeros(
            shape=(
                self.frames,
                self.channels,
                self.slices,
                self.height,
                self.width,
            ),
            chunks=(1, 1, 1, self.height, self.width),
            dtype=self.dtype,
        )
        # add all the images with this specific dimension.
        # Will be blank images if dataset
        # is incomplete
        for t in range(self.frames):
            for c in range(self.channels):
                for z in range(self.slices):
                    self.position_arrays[pos][t, c, z] = self._get_image(c, z)

    def _get_image(self, c, z):
        """
        Gets image at specific channel, z coordinate.
        Makes sure that the channel index
        specified corresponds to the channel_name index.

        Parameters
        ----------
        c:              (int) channel index.  Maps to channel_name index
        z:              (int) z/slice index.

        Returns
        -------
        img             (nd-array) numpy array of dimensions (Y, X)
        """

        chan_name = self.channel_names[c]
        pattern = int(
            re.search(r"pattern_\d\d\d", chan_name).group(0).strip("pattern_")
        )
        state = re.search(r"State_\d\d\d", chan_name)
        state_idx = 0 if not state else int(state.group(0).strip("State_"))

        fn = self.file_map[(pattern, state_idx, z)]
        img = zarr.open(tiff.imread(fn, aszarr=True), "r")

        return img

    def get_zarr(self, position):
        """
        return a zarr array for a given position.  Allows for only one position

        Parameters
        ----------
        position:       (int) position

        Returns
        -------
        position:       (zarr.array)

        """
        if position > self.positions - 1:
            raise ValueError(
                "Entered position is greater than "
                "the number of positions in the data"
            )

        if position not in self.position_arrays.keys():
            self._create_position_array(position)
        return self.position_arrays[position]

    def get_array(self, position):
        """
        return a numpy array for a given position.
        Allows for only one position

        Parameters
        ----------
        position:   (int) position

        Returns
        -------
        position:   (np.ndarray)

        """

        if position > self.positions - 1:
            raise ValueError(
                "Entered position is greater than "
                "the number of positions in the data"
            )

        # if position hasn't been initialized in memory, do that.
        if position not in self.position_arrays.keys():
            self._create_position_array(position)

        return np.array(self.position_arrays[position])

    def get_num_positions(self) -> int:
        return self.positions

    @property
    def shape(self):
        return self.frames, self.channels, self.slices, self.height, self.width
