import numpy as np
import zarr
import os
from numcodecs import Blosc

"""
Method:

create zarr memorystore
construct hierarchy there
place arrays into this memory store

at save time, create a target store that is zarr directoryStore
use zarr.copy_store to copy memory store to directory store

"""

# todo: how should we supply data?  as individual arrays?
# todo: don't forget that the user should be allowed to supply np arrays and zarr arrays
# todo: change to persistent arrays


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
    save_dir = None
    store_path = None
    store = None

    def __init__(self, save_dir: str, datatype: str):
        """
        datatype is one of "stokes" or "physical"
        :param save_dir:
        :param datatype:
        :param chunksize:
        """

        self._check_is_dir(save_dir)

        if datatype == 'stokes':
            self.__builder = StokesZarr()
        elif datatype == 'physical':
            self.__builder = PhysicalZarr()
        else:
            raise NotImplementedError()

    def _check_is_dir(self, path):
        # Upon init, check to make sure save path is a directory,
        # if not, create that directory
        if os.path.isdir(path):
            self.save_dir = path
        else:
            os.mkdir(path)
            self.save_dir = path

    def add_store(self, name: str):
        # Add Zarr store to existing save directory

        path = os.path.join(self.save_dir, name)

        if not path.endswith('.zarr'):
            path += '.zarr'
        try:
            self.set_store(path)
        except:
            print(f'Opening New Zarr Store at {path}')
            self.store = zarr.open(path)
            self.store_path = path

    def set_zarr_parameters(self, shape, chunks, **kwargs):
        # use kwargs to assign zarr array parameters
        self.__builder.init_zarr()
        pass

    def set_mm_metadata(self, meta):
        self.__builder.init_meta()
        pass

    def set_save_dir(self, path):
        self._check_is_dir(path)

    def set_store(self, path):

        if os.path.exists(path):
            print(f'Opening existing store at {path}')
            self.store_path = path
            self.store = zarr.open(path)
        else:
            raise ValueError(f'No store found at {path}, check spelling or create new store with add_store')

    def init_array(self, data_shape, chunk_size, dtype):
        self.__builder.init_array(self.store, data_shape, chunk_size, dtype)

    # def set_stokes(self, stokes_data: np.ndarray):
    #     # assign class attribute data
    #     # call self.__builder attributes
    #     self.__builder.init_arrays()
    #     pass
    #
    # def set_physical(self, physical_data: np.ndarray):
    #     # assign class attribute data
    #     # desired keyword args:
    #     #   data, channel, chunks, compressor
    #     # can be learned from supplied keywords:
    #     #   data.shape, chunk compatibility,
    #     self.__builder.init_arrays()

    # def write_stokes(self):
    #     pass
    #
    # def write_physical(self):
    #     pass

    def write(self, data, P, T, Z):
        self.__builder.init_zarr(self.store)
        self.__builder.write(data, P, T, Z)
        pass


class Builder:
    # interface for all builders

    # create zarr memory store
    def init_zarr(self): pass

    # create subarrays named after the channel
    def init_arrays(self): pass

    # assign group and array attributes
    def init_meta(self): pass

    # create a directory store and write the memory store to that
    def to_disk(self): pass


class PhysicalZarr(Builder):
    __pzarr = None
    __compressor = None

    def write(self, data, P, T, Z):
        """
        Write data to specified index of initialized zarr array

        :param data: (nd-array), data to be saved. Must be the shape that matches indices (P, T, Z, Y, X)
        :param P: (tuple or value), index or index range of the position dimension
        :param T: (tuple or value), index or index range of the time dimension
        :param Z: (tuple or value), index or index range of the Z dimension

        """

        shape = np.shape(data)

        if self.__pzarr.__len__() == 0:
            raise ValueError('Array not initialized')

        if len(P) == 1 and len(T) == 1 and len(Z) == 1:

            if len(shape) > 2:
                raise ValueError('Index dimensions do not match data dimensions')
            else:
                self.__pzarr['array'][P[0], T[0], Z[0]] = data

        elif len(P) == 1 and len(T) == 2 and len(Z) == 1:
            self.__pzarr['array'][P[0], T[0]:T[1], Z[0]] = data

        elif len(P) == 1 and len(T) == 1 and len(Z) == 2:
            self.__pzarr['array'][P[0], T[0], Z[0]:Z[1]] = data

        elif len(P) == 1 and len(T) == 2 and len(Z) == 2:
            self.__pzarr['array'][P[0], T[0]:T[1], Z[0]:Z[1]] = data

        elif len(P) == 2 and len(T) == 2 and len(Z) == 2:
            self.__pzarr['array'][P[0]:P[1], T[0]:T[1], Z[0]:Z[1]] = data

        elif len(P) == 2 and len(T) == 1 and len(Z) == 2:
            self.__pzarr['array'][P[0]:P[1], T[0], Z[0]:Z[1]] = data

        elif len(P) == 2 and len(T) == 2 and len(Z) == 1:
            self.__pzarr['array'][P[0]:P[1], T[0]:T[1], Z[0]] = data

        elif len(P) == 2 and len(T) == 1 and len(Z) == 1:
            self.__pzarr['array'][P[0]:P[1], T[0], Z[0]] = data


    def init_compressor(self, **kwargs):
        self.__compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)

    def init_array(self, store, data_shape, chunk_size, dtype):
        if self.__compressor == None:
            self.init_compressor()

        if len(data_shape) != 5:
            raise ValueError('Data shape must be (P, T, Z, Y, X)')

        self.init_zarr(store)
        self.__pzarr.zeros('array',shape=data_shape,chunks=chunk_size, dtype=dtype,
                           compressor=self.__compressor, overwrite=True)

    def init_zarr(self, store):
        self.__pzarr = store

class StokesZarr(Builder):
    __szarr = None
    __compressor = None

    def write(self, data, P, T, Z):
        """
        Write data to specified index of initialized zarr array

        :param data: (nd-array), data to be saved. Must be the shape that matches indices (P, T, Z, Y, X)
        :param P: (tuple or value), index or index range of the position dimension
        :param T: (tuple or value), index or index range of the time dimension
        :param Z: (tuple or value), index or index range of the Z dimension

        """

        shape = np.shape(data)

        if self.__szarr.__len__() == 0:
            raise ValueError('Array not initialized')

        if len(P) == 1 and len(T) == 1 and len(Z) == 1:

            if len(shape) > 2:
                raise ValueError('Index dimensions do not match data dimensions')
            else:
                self.__szarr['array'][P[0], T[0], Z[0]] = data

        elif len(P) == 1 and len(T) == 2 and len(Z) == 1:
            self.__szarr['array'][P[0], T[0]:T[1], Z[0]] = data

        elif len(P) == 1 and len(T) == 1 and len(Z) == 2:
            self.__szarr['array'][P[0], T[0], Z[0]:Z[1]] = data

        elif len(P) == 1 and len(T) == 2 and len(Z) == 2:
            self.__szarr['array'][P[0], T[0]:T[1], Z[0]:Z[1]] = data

        elif len(P) == 2 and len(T) == 2 and len(Z) == 2:
            self.__szarr['array'][P[0]:P[1], T[0]:T[1], Z[0]:Z[1]] = data

        elif len(P) == 2 and len(T) == 1 and len(Z) == 2:
            self.__szarr['array'][P[0]:P[1], T[0], Z[0]:Z[1]] = data

        elif len(P) == 2 and len(T) == 2 and len(Z) == 1:
            self.__szarr['array'][P[0]:P[1], T[0]:T[1], Z[0]] = data

        elif len(P) == 2 and len(T) == 1 and len(Z) == 1:
            self.__szarr['array'][P[0]:P[1], T[0], Z[0]] = data

    def init_compressor(self, **kwargs):
        self.__compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)

    def init_zarr(self, store):
        self.__szarr = store

    def init_array(self, store, data_shape, chunk_size, dtype):
        if self.__compressor == None:
            self.init_compressor()

        if len(data_shape) != 5:
            raise ValueError('Data shape must be (P, T, Z, Y, X)')

        self.init_zarr(store)
        self.__szarr.zeros('array', shape=data_shape, chunks=chunk_size, dtype=dtype,
                           compressor=self.__compressor, overwrite=True)