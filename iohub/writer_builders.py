from waveorder.io.writer import Builder
import numpy as np
import os
from numcodecs import Blosc
import zarr

class PhysicalZarr(Builder):

    def __init__(self, name=None):
        """

        """
        super().__init__()
        self.__zarr = None
        self.__compressor = None

        if name:
            self.name = name
        else:
            self.name = 'physical_data'

    def create_channel_dict(self, chan_name, clim=None):

        if chan_name == 'Retardance':
            min = 0.0
            max = 1000.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 100.0
        elif chan_name == 'Orientation':
            min = 0.0
            max = np.pi
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else np.pi

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


class StokesZarr(Builder):

    def __init__(self, name=None):
        """

        """
        super().__init__()
        self.__zarr = None
        self.__compressor = None

        if name:
            self.name = name
        else:
            self.name = 'stokes_data'

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


class RawZarr(Builder):

    def __init__(self, name=None):
        """

        """
        super().__init__()
        self.__zarr = None
        self.__compressor = None

        if name:
            self.name = name
        else:
            self.name = 'raw_data'

    def create_channel_dict(self, chan_name, clim=None):

        min = 0
        max = 65535
        start = clim[0] if clim else 0
        end = clim[1] if clim else 65535

        dict_ = {'active': True,
                'coefficient': 1.0,
                'color': '808080',
                'family': 'linear',
                'inverted': False,
                'label': chan_name,
                'window': {'end': end, 'max': max, 'min': min, 'start': start}
                }

        return dict_

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
            name = 'raw_data.zarr'
        if not name.endswith('.zarr'):
            name += '.zarr'
        path = os.path.join(directory, name)

        if not os.path.exists(path):
            store = zarr.open(path)
            self.set_zarr(store)
        else:
            raise FileExistsError(f"zarr already exists at {path}")