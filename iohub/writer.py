import numpy as np
import zarr
import os
from numcodecs import Blosc
from typing import Union
from zarr import Array, Group

# todo: add checks for inconsistent data shapes
# todo: change to persistent arrays?


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
                 alt_name: str = None):
        """
        datatype is one of "stokes" or "physical".
        Alt name specifies alternative name for the builder. i.e.'stokes denoised'

        :param save_dir:
        :param datatype:
        :param alt_name:
        """

        self.datatype = datatype
        self.__builder_name = alt_name

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
        one of "physical" or "stokes"

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
            print(f'Creating and opening subgroup {grp_name}')
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
            print(f'Opening subgroup {grp_name}')
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
        else:
            raise NotImplementedError("datatype must be one of 'stokes' or 'physical'")

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
            raise AttributeError("datatype is not set.  Must be one of 'stokes' or 'physical'")
        elif self.datatype == 'stokes':
            self.__builder = StokesZarr(self.__builder_name)
        elif self.datatype == 'physical':
            self.__builder = PhysicalZarr(self.__builder_name)
        else:
            raise NotImplementedError("datatype must be one of 'stokes' or 'physical'")

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

class Builder:
    """
    ABC for all builders
    """

    # create zarr memory store
    def init_zarr(self,
                  directory: str,
                  name: str): pass

    # create subarrays named after the channel
    def init_array(self,
                   store: Union[Array, Group],
                   data_shape: tuple,
                   chunk_size: tuple,
                   dtype: str,
                   pos: int,
                   prefix: str,
                   overwrite: bool
                   ): pass

    # assign group and array attributes
    def init_meta(self): pass

    # write data to t, c, z slice
    def write(self,
              data,
              t,
              c,
              z): pass


class PhysicalZarr(Builder):
    name = None

    def __init__(self, name=None):
        """

        """
        self.__pzarr = None
        self.__compressor = None

        if name:
            self.name = name
        else:
            self.name = 'physical_data'


    def write(self, data, t, c, z):
        """
        Write data to specified index of initialized zarr array

        :param data: (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        :param t: (list), index or index range of the time dimension
        :param c: (list), index or index range of the channel dimension
        :param z: (list), index or index range of the z dimension

        """

        shape = np.shape(data)

        if self.__pzarr.__len__() == 0:
            raise ValueError('Array not initialized')

        if len(c) == 1 and len(t) == 1 and len(z) == 1:

            if len(shape) > 2:
                raise ValueError('Index dimensions do not match data dimensions')
            else:
                self.__pzarr['array'][c[0], t[0], z[0]] = data

        elif len(c) == 1 and len(t) == 2 and len(z) == 1:
            self.__pzarr['array'][t[0]:t[1], c[0], z[0]] = data

        elif len(c) == 1 and len(t) == 1 and len(z) == 2:
            self.__pzarr['array'][t[0], c[0], z[0]:z[1]] = data

        elif len(c) == 1 and len(t) == 2 and len(z) == 2:
            self.__pzarr['array'][t[0]:t[1], c[0], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 2 and len(z) == 2:
            self.__pzarr['array'][t[0]:t[1], c[0]:c[1], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 1 and len(z) == 2:
            self.__pzarr['array'][t[0], c[0]:c[1], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 2 and len(z) == 1:
            self.__pzarr['array'][t[0]:t[1], c[0]:c[1], z[0]] = data

        elif len(c) == 2 and len(t) == 1 and len(z) == 1:
            self.__pzarr['array'][t[0], c[0]:c[1], z[0]] = data

        else:
            raise ValueError('Did not understand data formatting')

    def create_channel_dict(self, chan_name, clim=None):

        if chan_name == 'Retardance':
            min = 0.0
            max = 1000.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 100.0
        elif chan_name == 'Orientation':
            min = 0.0
            max = 180.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 180.0

        elif chan_name == 'Phase3D':
            min = -10.0
            max = 10.0
            start = clim[0] if clim else -0.2
            end = clim[1] if clim else 0.2


        elif chan_name == 'BF':
            min = 0.0
            max = 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 5.0

        else:
            min = 0.0
            max = 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 65535.0

        dict_ = {'active': True,
                'coefficient': 1.0,
                'color': '808080',
                'family': 'linear',
                'inverted': False,
                'label': chan_name,
                'window': {'end': end, 'max': max, 'min': min, 'start': start}
                }

        return dict_

    def set_channel_attributes(self, chan_names: list, clims: list):
        """
        A method for creating ome-zarr metadata dictionary.
        Channel names are defined by the user, everything else
        is pre-defined.

        Parameters
        ----------
        chan_names:     (list) List of channel names in the order of the channel dimensions
                                i.e. if 3D Phase is C = 0, list '3DPhase' first.

        clims:          (list of tuples) contrast limits to display for every channel

        """

        rdefs = {'defaultT': 0,
                 'model': 'color',
                 'projection': 'normal',
                 'defaultZ': 0}

        multiscale_dict = [{'datasets': [{'path': "array"}],
                            'version': '0.1'}]
        dict_list = []

        for i in range(len(chan_names)):
            if clims == None:
                dict_list.append(self.create_channel_dict(chan_names[i]))
            else:
                #TODO: Check if clims length matches channel length
                dict_list.append(self.create_channel_dict(chan_names[i], clims[i]))

        full_dict = {'multiscales': multiscale_dict,
                     'omero': {
                         'channels': dict_list,
                         'rdefs': rdefs,
                         'version': 0.1}
                     }

        self.__pzarr.attrs.put(full_dict)

    def _zarr_exists(self, path):
        if os.path.exists(path):
            print(f'Found existing store at {path}')
            return True
        else:
            # print(f'Creating new store at {path}')
            return False

    # Placeholder function for future compressor customization
    def init_compressor(self, compressor_: str):
        """
        Zarr supports a variety of compressor libraries:
            from NumCodecs:
                Blosc, Zstandard, LZ4, Zlib, BZ2, LZMA
        Blosc library supports many algorithms:
                zstd, blosclz, lz4, lz4hc, zlib, snappy

        Parameters
        ----------
        compressor_:    (str) one of the compressor libraries from NumCodecs

        Returns
        -------

        """
        if compressor_ is "Blosc":
            self.__compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
        else:
            raise NotImplementedError("only Blosc library with zstd algorithm is supported")

    # Initialize zero array
    def init_array(self, store, data_shape, chunk_size, dtype, chan_names, clims, overwrite):
        """

        Parameters
        ----------
        store
        data_shape
        chunk_size
        dtype
        chan_names
        clims
        overwrite

        Returns
        -------

        """
        if self.__compressor is None:
            self.init_compressor('Blosc')

        # Make sure data matches OME zarr structure
        if len(data_shape) != 5:
            raise ValueError('Data shape must be (T, C, Z, Y, X)')

        #TODO: GET RID OF THIS?
        try:
            self.set_zarr(store[self.name])
        except:
            store.create_group(self.name)
            self.set_zarr(store[self.name])

        self.set_channel_attributes(chan_names, clims)
        self.__pzarr.zeros('array', shape=data_shape, chunks=chunk_size, dtype=dtype,
                           compressor=self.__compressor, overwrite=overwrite)

    def init_zarr(self, directory, name=None):
        """
        create a zarr.DirectoryStore at the supplied directory

        Parameters
        ----------
        directory:  (str) path to directory store
        name:   (str) name of zarr array, with extension '.zarr'.  Defaults to 'physical_data.zarr'

        Returns
        -------

        """
        if name is None:
            name = 'physical_data.zarr'
        if not name.endswith('.zarr'):
            name += '.zarr'
        path = os.path.join(directory, name)

        # if name is None:
        #     path = os.path.join(directory, 'physical_data.zarr')
        # else:
        #     path = os.path.join(directory, name)
        #     if not path.endswith('.zarr'):
        #         path += '.zarr'

        if not os.path.exists(path):
            store = zarr.open(path)
            self.set_zarr(store)
        else:
            raise FileExistsError(f"zarr already exists at {path}")

    def set_zarr(self, store):
        """
        set this object's zarr store
        Parameters
        ----------
        store

        Returns
        -------

        """
        self.__pzarr = store

    def get_zarr(self):
        return self.__pzarr


class StokesZarr(Builder):
    name = None

    def __init__(self, name=None):
        """

        """
        self.__szarr = None
        self.__compressor = None

        if name:
            self.name = name
        else:
            self.name = 'stokes_data'


    def write(self, data, t, c, z):
        """
        Write data to specified index of initialized zarr array

        :param data: (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        :param t: (list), index or index range of the time dimension
        :param c: (list), index or index range of the channel dimension
        :param z: (list), index or index range of the z dimension

        """

        shape = np.shape(data)

        if self.__szarr.__len__() == 0:
            raise ValueError('Array not initialized')

        if len(c) == 1 and len(t) == 1 and len(z) == 1:

            if len(shape) > 2:
                raise ValueError('Index dimensions do not match data dimensions')
            else:
                self.__szarr['array'][c[0], t[0], z[0]] = data

        elif len(c) == 1 and len(t) == 2 and len(z) == 1:
            self.__szarr['array'][t[0]:t[1], c[0], z[0]] = data

        elif len(c) == 1 and len(t) == 1 and len(z) == 2:
            self.__szarr['array'][t[0], c[0], z[0]:z[1]] = data

        elif len(c) == 1 and len(t) == 2 and len(z) == 2:
            self.__szarr['array'][t[0]:t[1], c[0], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 2 and len(z) == 2:
            self.__szarr['array'][t[0]:t[1], c[0]:c[1], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 1 and len(z) == 2:
            self.__szarr['array'][t[0], c[0]:c[1], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 2 and len(z) == 1:
            self.__szarr['array'][t[0]:t[1], c[0]:c[1], z[0]] = data

        elif len(c) == 2 and len(t) == 1 and len(z) == 1:
            self.__szarr['array'][t[0], c[0]:c[1], z[0]] = data

        else:
            raise ValueError('Did not understand data formatting')

    def create_channel_dict(self, chan_name, clim=None):

        if chan_name == 'S0':
            min = 0.0
            max = 65535
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 1.0

        elif chan_name == 'S1':
            min = 10.0
            max = -10.0
            start = clim[0] if clim else -0.5
            end = clim[1] if clim else 0.5

        elif chan_name == 'S2':
            min = -10.0
            max = 10.0
            start = clim[0] if clim else -0.5
            end = clim[1] if clim else 0.5

        elif chan_name == 'S3':
            min = -10
            max = 10
            start = clim[0] if clim else -1.0
            end = clim[1] if clim else 1.0

        else:
            min = 0.0
            max = 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 65535.0

        dict_ = {'active': True,
                'coefficient': 1.0,
                'color': '808080',
                'family': 'linear',
                'inverted': False,
                'label': chan_name,
                'window': {'end': end, 'max': max, 'min': min, 'start': start}
                }

        return dict_

    def set_channel_attributes(self, chan_names: list, clims: list):
        """
        A method for creating ome-zarr metadata dictionary.
        Channel names are defined by the user, everything else
        is pre-defined.

        Parameters
        ----------
        chan_names:     (list) List of channel names in the order of the channel dimensions
                                i.e. if 3D Phase is C = 0, list '3DPhase' first.

        clims:          (list of tuples) contrast limits to display for every channel

        """

        rdefs = {'defaultT': 0,
                 'model': 'color',
                 'projection': 'normal',
                 'defaultZ': 0}

        multiscale_dict = [{'datasets': [{'path': "array"}],
                            'version': '0.1'}]
        dict_list = []

        for i in range(len(chan_names)):
            if clims == None:
                dict_list.append(self.create_channel_dict(chan_names[i]))
            else:
                #TODO: Check if clims length matches channel length
                dict_list.append(self.create_channel_dict(chan_names[i], clims[i]))

        full_dict = {'multiscales': multiscale_dict,
                     'omero': {
                         'channels': dict_list,
                         'rdefs': rdefs,
                         'version': 0.1}
                     }

        self.__szarr.attrs.put(full_dict)

    def _zarr_exists(self, path):
        if os.path.exists(path):
            print(f'Found existing store at {path}')
            return True
        else:
            # print(f'Creating new store at {path}')
            return False

    # Placeholder function for future compressor customization
    def init_compressor(self, compressor_: str):
        """
        Zarr supports a variety of compressor libraries:
            from NumCodecs:
                Blosc, Zstandard, LZ4, Zlib, BZ2, LZMA
        Blosc library supports many algorithms:
                zstd, blosclz, lz4, lz4hc, zlib, snappy

        Parameters
        ----------
        compressor_:    (str) one of the compressor libraries from NumCodecs

        Returns
        -------

        """
        if compressor_ is "Blosc":
            self.__compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
        else:
            raise NotImplementedError("only Blosc library with zstd algorithm is supported")

    # Initialize zero array
    def init_array(self, store, data_shape, chunk_size, dtype, chan_names, clims, overwrite):
        """

        Parameters
        ----------
        store
        data_shape
        chunk_size
        dtype
        chan_names
        clims
        overwrite

        Returns
        -------

        """
        if self.__compressor is None:
            self.init_compressor('Blosc')

        # Make sure data matches OME zarr structure
        if len(data_shape) != 5:
            raise ValueError('Data shape must be (T, C, Z, Y, X)')

        try:
            self.set_zarr(store[self.name])
        except:
            store.create_group(self.name)
            self.set_zarr(store[self.name])

        self.set_channel_attributes(chan_names, clims)
        self.__szarr.zeros('array', shape=data_shape, chunks=chunk_size, dtype=dtype,
                           compressor=self.__compressor, overwrite=overwrite)

    def init_zarr(self, directory, name=None):
        """
        create a zarr.DirectoryStore at the supplied directory

        Parameters
        ----------
        directory:  (str) path to directory store
        name:   (str) name of zarr array, with extension '.zarr'.  Defaults to 'physical_data.zarr'

        Returns
        -------

        """
        if name is None:
            name = 'physical_data.zarr'
        if not name.endswith('.zarr'):
            name += '.zarr'
        path = os.path.join(directory, name)

        if not os.path.exists(path):
            store = zarr.open(path)
            self.set_zarr(store)
        else:
            raise FileExistsError(f"zarr already exists at {path}")

    def set_zarr(self, store):
        """
        set this object's zarr store
        Parameters
        ----------
        store

        Returns
        -------

        """
        self.__szarr = store

    def get_zarr(self):
        return self.__szarr