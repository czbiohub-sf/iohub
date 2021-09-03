import zarr
import os
from waveorder.io.writer_builders import PhysicalZarr, StokesZarr, RawZarr


class WaveorderWriter:
    """
    given stokes or physical data, construct a standard hierarchy in zarr for output
        should conform to the ome-zarr standard as much as possible

    """
    __builder = None
    __save_dir = None
    __root_store_path = None
    __builder_name = None
    __current_zarr_group = None
    __current_zarr_dir = None
    store = None

    current_group_name = None
    current_position = None

    def __init__(self,
                 save_dir: str = None,
                 datatype: str = None,
                 alt_name: str = None,
                 silence: bool = False):
        """
        datatype is one of "stokes" or "physical".
        Alt name specifies alternative name for the builder. i.e.'stokes denoised'

        :param save_dir:
        :param datatype:
        :param alt_name:
        """

        self.datatype = datatype
        self.__builder_name = alt_name
        self.silence = silence

        if os.path.exists(save_dir) and save_dir.endswith('.zarr'):
            print(f'Opening existing store at {save_dir}')
            self._open_zarr_root(save_dir)

        else:
            self._check_is_dir(save_dir)

    def _check_is_dir(self, path):
        """
        directory verification
        assigns self.__save_dir

        Parameters
        ----------
        path (str):

        Returns
        -------

        """
        if os.path.isdir(path) and os.path.exists(path):
            self.__save_dir = path
        else:
            print(f'No existing directory found. Creating new directory at {path}')
            os.mkdir(path)
            self.__save_dir = path

    def get_current_group(self):
        return self.current_group_name

    def set_type(self, datatype, alt_name=None):
        """
        one of "physical" or "stokes" or "raw"

        Parameters
        ----------
        datatype:   (str) one of "physical" or "stokes"
        alt_name:   (str) alternative name for the builder.  Changes the name of the directory below pos directory

        Returns
        -------

        """
        self.__builder_name = alt_name
        self.datatype = datatype

    def create_position(self, position, prefix=None):
        """
        append file paths for zarr store

        Parameters
        ----------
        position: (int) position subfolder that will contain .zarr array
        prefix:   (str) prefix to put before the name of position subdirectory i.e. {Prefix}_Pos_000

        Returns
        -------

        """

        grp_name = f'{prefix}_Pos_{position:03d}.zarr' if prefix else f'Pos_{position:03d}.zarr'
        grp_path = os.path.join(self.__root_store_path, grp_name)

        if os.path.exists(grp_path):
            raise FileExistsError('A position subgroup with this name already exists')
        else:
            if not self.silence: print(f'Creating and opening subgroup {grp_name}')
            self.store.create_group(grp_name)
            self.__current_zarr_group = self.store[grp_name]
            self.current_group_name = grp_name
            self.current_position = position

    def open_position(self, position, prefix=None):

        """
        Opens existing position sub-group.  Prefix must be specified to find the subgroup correctly

        Parameters
        ----------
        position: (int) position subfolder that will contain .zarr array
        prefix:   (str) prefix to put before the name of position subdirectory i.e. {Prefix}_Pos_000

        Returns
        -------

        """

        grp_name = f'{prefix}_Pos_{position:03d}.zarr' if prefix else f'Pos_{position:03d}.zarr'
        grp_path = os.path.join(self.__root_store_path, grp_name)

        #TODO: Find fancy way to search for prefix
        if os.path.exists(grp_path):
            if not self.silence: print(f'Opening subgroup {grp_name}')
            self.group_name = grp_name
            self.__current_zarr_group = self.store[grp_name]
            self.current_group_name = grp_name
            self.current_position = position

            groups = list(self.__current_zarr_group.group_keys())
            if len(groups) != 0 and self.__builder.name in groups:
                self.__current_zarr_dir = self.__current_zarr_group[self.__builder.name]
            else:
                self.__current_zarr_dir = self.__current_zarr_group

        else:
            raise FileNotFoundError(f'Could not find zarr position subgroup at {grp_path}\
                                    Check spelling or create position subgroup with create_position')

    def create_zarr_root(self, name):
        """
        Method for creating the root zarr store.
        If the store already exists, it will raise an error.
        Name corresponds to the root directory name (highest level) zarr store.

        Parameters
        ----------
        name:       (string) Name of the zarr store.

        """

        if self.datatype is None:
            raise AttributeError("datatype is not set.  Must be one of 'stokes' or 'physical'")
        elif self.datatype == 'stokes':
            self.__builder = StokesZarr(self.__builder_name)
        elif self.datatype == 'physical':
            self.__builder = PhysicalZarr(self.__builder_name)
        elif self.datatype == 'raw':
            self.__builder = RawZarr(self.__builder_name)
        else:
            raise NotImplementedError("datatype must be one of 'stokes' or 'physical' or 'raw'")

        if not name.endswith('.zarr'):
            name = name+'.zarr'

        zarr_path = os.path.join(self.__save_dir, name)
        if os.path.exists(zarr_path):
            raise FileExistsError('A zarr store with this name already exists')

        print(f'Creating new zarr store at {zarr_path}')
        self.store = zarr.open(zarr_path)
        self.__root_store_path = zarr_path

    def _open_zarr_root(self, path):
        """
        Change current zarr to an existing store
        if zarr doesn't exist, raise error

        Parameters
        ----------
        path:       (str) path to store. Must end in .zarr

        Returns
        -------

        """

        if self.datatype is None:
            raise AttributeError("datatype is not set.  Must be one of 'stokes' or 'physical' or 'raw'")
        elif self.datatype == 'stokes':
            self.__builder = StokesZarr(self.__builder_name)
        elif self.datatype == 'physical':
            self.__builder = PhysicalZarr(self.__builder_name)
        elif self.datatype == 'raw':
            self.__builder = RawZarr(self.__builder_name)
        else:
            raise NotImplementedError("datatype must be one of 'stokes' or 'physical' or 'raw'")

        if os.path.exists(path):
            self.store = zarr.open(path)
            self.__root_store_path = path
        else:
            raise FileNotFoundError(f'No store found at {path}, check spelling or create new store with create_zarr')

    def init_array(self, data_shape: tuple, chunk_size: tuple, chan_names, clims = None, dtype='float32', overwrite=False):
        """
        Initializes the array and metadata for the array.  It will create the builder subgroup (ie. 'physical_data')
        under the current position subgroup.  The metadata lives in the builder subgroup with the array underneath

        Parameters
        ----------
        data_shape:     (tuple) Shape of the position dataset (T, C, Z, Y, X)
        chunk_size:     (tuple) Chunks to save, (T, C, Z, Y, X)
                                i.e. to chunk along z chunk_size = (1, 1, 1, Y, X)
        chan_names:     (list) List of channel names in the order of the channel dimensions
                                i.e. if 3D Phase is C = 0, list '3DPhase' first.
        clims:          (list of tuples) contrast limits to display for each channel
                                in visualization tools
        dtype:          (string) data type of the array

        Returns
        -------

        """
        if len(chan_names) != data_shape[1]:
            raise ValueError('Number of Channel Names does not equal number of channels \
                                in the array')

        self.__builder.init_array(self.__current_zarr_group, data_shape, chunk_size, dtype, chan_names, clims, overwrite)
        self.__current_zarr_dir = self.__current_zarr_group[self.__builder.name]

    def set_compressor(self, compressor):
        """
        Placeholder function for future user-specified compressors.

        Parameters
        ----------
        compressor: (object) compressor to use for data saving.

        Returns
        -------

        """
        self.__builder.init_compressor(compressor)

    def write(self, data, t=None, c=None, z=None):
        """
        Wrapper that calls the builder's write function.
        Will write to existing array of zeros and place
        data over the specified indicies.

        Parameters
        ----------
        data:   (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        t:      (list or value), index or index range of the time dimension
        c:      (list or value), index or index range of the channel dimension
        z:      (list or value), index or index range of the Z-slice dimension

        Returns
        -------

        """
        if t is None:
            t = [0,data.shape[0]]

        if c is None:
            c = [0,data.shape[1]]

        if z is None:
            z = [0,data.shape[2]]

        if isinstance(t, int):
            t = [t]

        if isinstance(c, int):
            c = [c]

        if isinstance(z, int):
            z = [z]

        self.__builder.set_zarr(self.__current_zarr_dir)
        self.__builder.write(data, t, c, z)
