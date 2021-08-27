import os
from numcodecs import Blosc
import numpy as np

class Builder:
    """
    ABC for all builders
    """
    def __init__(self):
        self.__zarr = None
        self.__compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)


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
        self.__zarr.zeros('array', shape=data_shape, chunks=chunk_size, dtype=dtype,
                           compressor=self.__compressor, overwrite=overwrite)

    def write(self, data, t, c, z):
        """
        Write data to specified index of initialized zarr array

        :param data: (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        :param t: (list), index or index range of the time dimension
        :param c: (list), index or index range of the channel dimension
        :param z: (list), index or index range of the z dimension

        """

        shape = np.shape(data)

        if self.__zarr.__len__() == 0:
            raise ValueError('Array not initialized')

        if len(c) == 1 and len(t) == 1 and len(z) == 1:

            if len(shape) > 2:
                raise ValueError('Index dimensions do not match data dimensions')
            else:
                self.__zarr['array'][t[0], c[0], z[0]] = data

        elif len(c) == 1 and len(t) == 2 and len(z) == 1:
            self.__zarr['array'][t[0]:t[1], c[0], z[0]] = data

        elif len(c) == 1 and len(t) == 1 and len(z) == 2:
            self.__zarr['array'][t[0], c[0], z[0]:z[1]] = data

        elif len(c) == 1 and len(t) == 2 and len(z) == 2:
            self.__zarr['array'][t[0]:t[1], c[0], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 2 and len(z) == 2:
            self.__zarr['array'][t[0]:t[1], c[0]:c[1], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 1 and len(z) == 2:
            self.__zarr['array'][t[0], c[0]:c[1], z[0]:z[1]] = data

        elif len(c) == 2 and len(t) == 2 and len(z) == 1:
            self.__zarr['array'][t[0]:t[1], c[0]:c[1], z[0]] = data

        elif len(c) == 2 and len(t) == 1 and len(z) == 1:
            self.__zarr['array'][t[0], c[0]:c[1], z[0]] = data

        else:
            raise ValueError('Did not understand data formatting')

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

        if len(chan_names) < len(clims):
            raise ValueError('Contrast Limits specified exceed the number of channels given')

        for i in range(len(chan_names)):
            if not clims or i >= len(clims):
                dict_list.append(self.create_channel_dict(chan_names[i]))
            else:
                dict_list.append(self.create_channel_dict(chan_names[i], clims[i]))

        full_dict = {'multiscales': multiscale_dict,
                     'omero': {
                         'channels': dict_list,
                         'rdefs': rdefs,
                         'version': 0.1}
                     }

        self.__zarr.attrs.put(full_dict)

    # Placeholder function for future compressor customization

    def set_zarr(self, store):
        """
        set this object's zarr store
        Parameters
        ----------
        store

        Returns
        -------

        """
        self.__zarr = store

    def get_zarr(self):
        return self.__zarr

    def _zarr_exists(self, path):
        if os.path.exists(path):
            print(f'Found existing store at {path}')
            return True
        else:
            # print(f'Creating new store at {path}')
            return False

    def create_channel_dict(self, chan_names, clims):
        pass

    def init_compressor(self):
        pass
