import numpy as np
import zarr

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

    def __init__(self, datatype: str, chunksize: tuple):
        """
        datatype is one of "stokes" or "physical"
        :param datatype:
        :param chunksize:
        """
        if datatype == 'stokes':
            self.__builder = StokesZarr()
        elif datatype == 'physical':
            self.__builder = PhysicalZarr()
        else:
            raise NotImplementedError()

    def get_stokes(self):
        # call stokes builder (with params?) and return stokes zarr
        # if path specified, write to path
        pass

    def get_physical(self):
        # call physical builder (with params?) and return physical zarr
        # if path specified, write to path
        pass

    def set_zarr_parameters(self, shape, chunks, **kwargs):
        # use kwargs to assign zarr array parameters
        self.__builder.init_zarr()
        pass

    def set_mm_metadata(self, meta):
        self.__builder.init_meta()
        pass

    def set_stokes(self, stokes_data: np.ndarray):
        # assign class attribute data
        # call self.__builder attributes
        self.__builder.init_arrays()
        pass

    def set_physical(self, physical_data: np.ndarray):
        # assign class attribute data
        # desired keyword args:
        #   data, channel, chunks, compressor
        # can be learned from supplied keywords:
        #   data.shape, chunk compatibility,
        self.__builder.init_arrays()

    # def write_stokes(self):
    #     pass
    #
    # def write_physical(self):
    #     pass

    def write(self):
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

    def init_zarr(self, **kwargs):
        self.__pzarr = zarr.DirectoryStore()

    def init_arrays(self, **kwargs):
        pass


class StokesZarr(Builder):
    __szarr = None

    def init_zarr(self, **kwargs):
        self.__szarr = zarr.DirectoryStore()