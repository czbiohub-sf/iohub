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

class WaveorderWriter:
    """
    given stokes or physical data, construct a standard hierarchy in zarr for output
        should conform to the ome-zarr standard as much as possible

    """
    __builder = None

    def __init__(self, datatype: str):
        """
        datatype is one of "stokes" or "physical"
        :param datatype:
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
        # call self.__builder attributes
        pass

    def write_stokes(self):
        pass

    def write_physical(self):
        pass


class Builder:
    # interface for all builders
    def init_zarr(self): pass
    def init_arrays(self): pass
    def init_meta(self): pass
    def to_disk(self): pass


class StokesZarr(Builder):
    # define stokes zarr structure here
    __szarr = None

    def init_zarr(self, **kwargs):
        self.__szarr = zarr.MemoryStore()


class PhysicalZarr(Builder):
    # define physical zarr structure here
    __pzarr = None

    def init_zarr(self, **kwargs):
        self.__pzarr = zarr.MemoryStore()