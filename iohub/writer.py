import zarr
import os
from waveorder.io.writer_structures import HCSZarr, DefaultZarr


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
    store = None

    current_group_name = None
    current_position = None

    def __init__(self,
                 save_dir: str = None,
                 hcs: bool = False,
                 hcs_meta: dict = None,
                 verbose: bool = False):

        self.verbose = verbose
        self.use_hcs = hcs
        self.hcs_meta = hcs_meta

        if os.path.exists(save_dir) and save_dir.endswith('.zarr'):
            print(f'Opening existing store at {save_dir}')
            self._open_zarr_root(save_dir)
        else:
            self._check_is_dir(save_dir)

        # initialize the subwriter based upon HCS or Default
        if self.use_hcs:
            if not self.hcs_meta:
                raise ValueError('No HCS Metadata provided. If HCS format is to be used you must specify the HCS Metadata')

            self.sub_writer = HCSZarr(self.store, self.__root_store_path, self.hcs_meta)
        else:
            self.sub_writer = DefaultZarr(self.store, self.__root_store_path)

        if self.verbose:
            self.sub_writer.set_verbosity(self.verbose)

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

    def _open_zarr_root(self, path):

        #TODO: Use case where user opens an already HCS-store?
        """
        Change current zarr to an existing store
        if zarr doesn't exist, raise error

        Parameters
        ----------
        path:       (str) path to store. Must end in .zarr

        Returns
        -------

        """

        if os.path.exists(path):
            self.store = zarr.open(path)
            self.__root_store_path = path
        else:
            raise FileNotFoundError(f'No store found at {path}, check spelling or create new store with create_zarr')

    def create_zarr_root(self, name):
        """
        Method for creating the root zarr store.
        If the store already exists, it will raise an error.
        Name corresponds to the root directory name (highest level) zarr store.

        Parameters
        ----------
        name:       (string) Name of the zarr store.

        """

        if not name.endswith('.zarr'):
            name = name+'.zarr'

        zarr_path = os.path.join(self.__save_dir, name)
        if os.path.exists(zarr_path):
            raise FileExistsError('A zarr store with this name already exists')

        print(f'Creating new zarr store at {zarr_path}')
        self.store = zarr.open(zarr_path)
        self.__root_store_path = zarr_path
        self.sub_writer.set_store(self.store)
        self.sub_writer.set_root(self.__root_store_path)
        self.sub_writer.init_hierarchy()

    def init_array(self, position, data_shape, chunk_size, chan_names, dtype='float32',
                   clims=None, position_name=None, overwrite=False):
        """

        Creates a subgroup structure based on position index.  Then initializes the zarr array under the
        current position subgroup.  Array level is called 'array' in the hierarchy.

        Parameters
        ----------
        position:           (int) Position index upon which to initialize array
        data_shape:         (tuple)  Desired Shape of your data (T, C, Z, Y, X).  Must match data
        chunk_size:         (tuple) Desired Chunk Size (T, C, Z, Y, X).  Chunking each image would be (1, 1, 1, Y, X)
        dtype:              (str) Data Type, i.e. 'uint16'
        chan_names:         (list) List of strings corresponding to your channel names.  Used for OME-zarr metadata
        clims:              (list) list of tuples corresponding to contrast limtis for channel.  OME-Zarr metadata
        overwrite:          (bool) Whether or not to overwrite the existing data that may be present.

        Returns
        -------

        """

        pos_name = position_name if position_name else f'Pos_{position:03d}'

        # Make sure data matches OME zarr structure
        if len(data_shape) != 5:
            raise ValueError('Data shape must be (T, C, Z, Y, X)')

        self.sub_writer.create_position(position, pos_name)
        self.sub_writer.init_array(data_shape, chunk_size, dtype, chan_names, clims, overwrite)

    def write(self, data, p, t=None, c=None, z=None):
        """
        Wrapper that calls the builder's write function.
        Will write to existing array of zeros and place
        data over the specified indicies.

        Parameters
        ----------
        data:   (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        p:      (int), Position index in which to write the data into
        t:      (slice or int), index or index range of the time dimension
        c:      (slice or int), index or index range of the channel dimension
        z:      (slice or int), index or index range of the Z-slice dimension
        Returns
        -------
        """
        self.sub_writer.open_position(p)

        if t is None:
            t = slice(0, data.shape[0])

        if c is None:
            c = slice(0, data.shape[1])

        if z is None:
            z = slice(0, data.shape[2])

        self.sub_writer.write(data, t, c, z)
