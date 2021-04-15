import numpy as np
import zarr
import os
from numcodecs import Blosc
from typing import Union
from zarr import Array, Group

# todo: implement hierarchies / groups
# todo: add checks for inconsistent data shapes
# todo: change to persistent arrays?


# initialize upfront -- with file path and dimensions -- use zeros and replace later
#       initial dimensions are just (p, t, z, y, x) = (1, 1, 1, 512, 512) ==> (file) 2.0.0._._
#       Initial chunks are important, must determine this first.
# as data is reconstructed, replace zeros using same zarr reference.
#


class WaveorderWriter:
    """
    given stokes or physical data, construct a standard hierarchy in zarr for output
        should conform to the ome-zarr standard as much as possible

    """
    __builder = None
    __save_dir = None
    __store_path = None
    store = None
    current_position = None
    position_dir = None

    def __init__(self,
                 save_dir: str,
                 datatype: str = None):
        """
        datatype is one of "stokes" or "physical"
        :param save_dir:
        :param datatype:
        """

        self.datatype = datatype

        self._check_is_dir(save_dir)

        # if datatype == 'stokes':
        #     self.__builder = StokesZarr()
        # elif datatype == 'physical':
        #     self.__builder = PhysicalZarr()
        # else:
        #     raise NotImplementedError()

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
        if os.path.isdir(path):
            self.__save_dir = path
        else:
            os.mkdir(path)
            self.__save_dir = path

    def _check_and_create_position(self, pos):
        """
        modifies the underlying zarr path based on position subfolder

        Parameters
        ----------
        pos:    (int):

        Returns
        -------

        """

        position_path = os.path.join(self.__save_dir, f'Pos_{pos:03d}')
        if os.path.exists(position_path):
            self.position_dir = position_path
        else:
            os.mkdir(position_path)
            self.position_dir = position_path

    def set_type(self, datatype):
        """
        one of "physical" or "stokes"

        Parameters
        ----------
        datatype:   (str) one of "physical" or "stokes"

        Returns
        -------

        """
        self.datatype = datatype

    def set_position(self, position):
        """
        append file paths for zarr store

        Parameters
        ----------
        position: (int) position subfolder that will contain .zarr array

        Returns
        -------

        """

        self._check_and_create_position(position)
        self.current_position = position

    def create_zarr(self, name=None):
        """
        Method for creating a zarr store.
        If the store already exists, it will open that store.
        If no name is supplied, the default builder name will
        be used ('stokes_data' or 'physical_data')

        Parameters
        ----------
        name:       (string) Optional. Name of the zarr store.

        """

        if self.datatype is None:
            raise AttributeError("datatype is not set.  Must be one of 'stokes' or 'physical'")
        elif self.datatype == 'stokes':
            self.__builder = StokesZarr()
        elif self.datatype == 'physical':
            self.__builder = PhysicalZarr()
        else:
            raise NotImplementedError("datatype must be one of 'stokes' or 'physical'")

        self.__builder.init_zarr(self.position_dir, name)
        self.store = self.__builder.get_zarr()

    def set_zarr(self, path):
        """
        Change current zarr to an existing store
        if zarr doesn't exist, raise error

        Parameters
        ----------
        path:       (str) path to

        Returns
        -------

        """

        if self.datatype is None:
            raise AttributeError("datatype is not set.  Must be one of 'stokes' or 'physical'")
        elif self.datatype == 'stokes':
            self.__builder = StokesZarr()
        elif self.datatype == 'physical':
            self.__builder = PhysicalZarr()
        else:
            raise NotImplementedError("datatype must be one of 'stokes' or 'physical'")

        if os.path.exists(path):
            print(f'Opening existing store at {path}')
            self.__store_path = path
            self.store = zarr.open(path)
            self.__builder.set_zarr(self.store)
        else:
            raise FileNotFoundError(f'No store found at {path}, check spelling or create new store with create_zarr')

    def init_array(self, data_shape: tuple, chunk_size: tuple, dtype='float32', overwrite=False):
        """
        Parameters
        ----------
        data_shape:     (tuple) Shape of the position dataset (T, C, Z, Y, X)
        chunk_size:     (tuple) Chunks to save, (T, C, Z, Y, X)
                                i.e. to chunk along z chunk_size = (1, 1, 1, Y, X)
        dtype:          (string) data type of the array

        Returns
        -------

        """
        self.__builder.init_array(self.store, data_shape, chunk_size, dtype, overwrite)

    #TODO: Add user-defined contrast limits
    def set_channel_attributes(self, chan_names: list):
        """
        A method for creating ome-zarr metadata dictionary.
        Channel names are defined by the user, everything else
        is pre-defined.

        Parameters
        ----------
        chan_names:     (list) List of channel names in the order of the channel dimensions
                                i.e. if 3D Phase is C = 0, list '3DPhase' first.

        """

        if len(chan_names) != self.store['array'].shape[1]:
            raise ValueError('Number of Channel Names does not equal number of channels \
                                in the array')
        else:

            rdefs = {'defaultT': 0,
                     'model': 'color',
                     'projection': 'normal',
                     'defaultZ': 0}

            multiscale_dict = [{'datasets': [{'path': "array"}],
                                'version': '0.1'}]
            dict_list = []
            for i in range(len(chan_names)):
                dict_list.append(self.__builder.create_channel_dict(chan_names[i]))

            full_dict = {'multiscales': multiscale_dict,
                         'omero': {'channels': dict_list},
                         'rdefs': rdefs}

            self.store.attrs.put(full_dict)

    def set_compressor(self, compressor):
        self.__builder.init_compressor(compressor)

    def write(self, data, t, c, z):
        """
        Wrapper that calls the builder's write function.
        Will write to existing array of zeros and place
        data over the specified indicies

        :param data: (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        :param t: (list or value), index or index range of the time dimension
        :param c: (list or value), index or index range of the channel dimension
        :param z: (list or value), index or index range of the Z-slice dimension

        """

        if isinstance(t, int):
            t = [t]

        if isinstance(c, int):
            c = [c]

        if isinstance(z, int):
            z = [z]

        self.__builder.set_zarr(self.store)
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
    # __pzarr = None
    # __compressor = None

    def __init__(self):
        """

        """
        self.__pzarr = None
        self.__compressor = None

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

    # TODO: adjust min/max contrast limits based on data type
    def create_channel_dict(self, chan_name):

        if chan_name == 'Retardance':
            max = 100.0
            min = 0.0
            start = 0.0
            end = 8.0
        elif chan_name == 'Orientation':
            max = 4.0
            min = 0.0
            start = 0.0
            end = 3.141593

        elif chan_name == 'Phase3D':
            min = -1.0
            max = 1.0
            start = -0.2
            end = 0.2

        elif chan_name == 'BF':
            min = 0.0
            max = 100
            start = 0.0
            end = 5.0

        else:
            min = 0.0
            max = 10000.0
            start = 0.0
            end = 1000.0

        dict = {'active': True,
                'coefficient': 1.0,
                'color': '808080',
                'family': 'linear',
                'inverted': False,
                'label': chan_name,
                'window': {'end': end, 'max': max, 'min': min, 'start': start}
                }

        return dict

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
    def init_array(self, store, data_shape, chunk_size, dtype, overwrite):
        """

        Parameters
        ----------
        store
        data_shape
        chunk_size
        dtype
        overwrite

        Returns
        -------

        """
        if self.__compressor is None:
            self.init_compressor('Blosc')

        # Make sure data matches OME zarr structure
        if len(data_shape) != 5:
            raise ValueError('Data shape must be (T, C, Z, Y, X)')

        self.set_zarr(store)
        self.__pzarr.zeros('array',shape=data_shape, chunks=chunk_size, dtype=dtype,
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
    __szarr = None
    __compressor = None

    def write(self, data, t, c, z):
        """
        Write data to specified index of initialized zarr array

        :param data: (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        :param t: (list), index or index range of the time dimension
        :param c: (list), index or index range of the channel dimension
        :param z: (list), index or index range of the Z dimension

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
            self.__szarr['array'][t[0], c[0]:c[1],  z[0]] = data

    def create_channel_dict(self, chan_name):

        if chan_name == 'S0':
            max = 10
            min = 0.0
            start = 0.0
            end = 1.0
        elif chan_name == 'S1':
            max = 1.0
            min = -1.0
            start = -0.5
            end = 0.5

        elif chan_name == 'S2':
            max = 1.0
            min = -1.0
            start = -0.5
            end = 0.5

        elif chan_name == 'S3':
            max = 1.0
            min = -1.0
            start = -1.0
            end = 1.0

        else:
            min = 0.0
            max = 1.0
            start = 0.0
            end = 1.0

        dict = {'active': True,
                'coefficient': 1.0,
                'color': '808080',
                'family': 'linear',
                'inverted': False,
                'label': chan_name,
                'window': {'end': end, 'max': max, 'min': min, 'start': start}
                }

        return dict

    def _zarr_exists(self, path):
        if os.path.exists(path):
            print(f'Found existing store at {path}')
            # return True
        else:
            print(f'Creating new store at {path}')

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

    def init_array(self, store, data_shape, chunk_size, dtype, overwrite):
        """

        Parameters
        ----------
        store
        data_shape
        chunk_size
        dtype
        overwrite

        Returns
        -------

        """
        if self.__compressor is None:
            self.init_compressor("Blosc")

        if len(data_shape) != 5:
            raise ValueError('Data shape must be (T, C, Z, Y, X)')

        self.set_zarr(store)
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
            name = 'stokes_data.zarr'
        if not name.endswith('.zarr'):
            name += '.zarr'
        path = os.path.join(directory, name)

        if not os.path.exists(path):
            store = zarr.open(path)
            self.set_zarr(store)
        else:
            raise FileExistsError(f"zarr already exists at {path}")

        # if name == None:
        #     path = os.path.join(directory, 'stokes_data.zarr')
        # else:
        #     path = os.path.join(directory, name)
        #     if not path.endswith('.zarr'):
        #         path += '.zarr'
        # self._zarr_exists(path)
        # store = zarr.open(path)
        # self.set_zarr(store)

    def set_zarr(self, store):
        self.__szarr = store

    def get_zarr(self):
        return self.__szarr
