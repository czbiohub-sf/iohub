import warnings

import numpy as np
import zarr
from waveorder.io.reader_base import ReaderBase
from pycromanager import Dataset


class PycromanagerReader(ReaderBase):

    stage_positions = None
    z_step_size = None

    def __init__(self, data_path: str):
        super().__init__()

        """
        Reader for data acquired with pycromanager, effectively a wrapper of the pycromanager.Dataset class

        """

        self.dataset = Dataset(data_path)
        self._axes = self.dataset.axes

        self.frames = len(self._axes['time']) if 'time' in self._axes.keys() else 1
        self.channels = len(self._axes['channel']) if 'channel' in self._axes.keys() else 1
        self.slices = len(self._axes['z']) if 'z' in self._axes.keys() else 1
        self.height = self.dataset.image_height
        self.width = self.dataset.image_width
        self.dtype = self.dataset.dtype

        self.mm_meta = self._get_summary_metadata()
        self.channel_names = list(self.dataset.get_channel_names())

    def _get_summary_metadata(self):
        pm_metadata = self.dataset.summary_metadata
        pm_metadata['MicroManagerVersion'] = 'pycromanager'
        pm_metadata['Positions'] = self.get_num_positions()

        img_metadata = self.get_image_metadata(0, 0, 0, 0)
        pm_metadata['z-step_um'] = None
        pm_metadata['StagePositions'] = []

        if 'ZPosition_um_Intended' in img_metadata.keys():
            pm_metadata['z-step_um'] = np.around(abs(self.get_image_metadata(0, 0, 0, 1)['ZPosition_um_Intended'] -
                                                     self.get_image_metadata(0, 0, 0, 0)['ZPosition_um_Intended']),
                                                 decimals=3)

        if 'XPosition_um_Intended' in img_metadata.keys():
            for p in range(self.get_num_positions()):
                img_metadata = self.get_image_metadata(p, 0, 0, 0)
                pm_metadata['StagePositions'].append({img_metadata['Core-XYStage']: (img_metadata['XPosition_um_Intended'],
                                                                                     img_metadata['YPosition_um_Intended'])})

        return {'Summary': pm_metadata}

    def get_num_positions(self) -> int:
        return len(self._axes['position']) if 'position' in self._axes.keys() else 1

    def get_image(self, p, t, c, z) -> np.ndarray:
        image = None
        if 'position' not in self._axes.keys():
            p = None

        if self.dataset.has_image(position=p, time=t, channel=c, z=z):
            image = self.dataset.read_image(position=p, time=t, channel=c, z=z)

        return image

    def get_zarr(self, position: int) -> zarr.array:
        if 'position' in self._axes.keys():
            data = self.dataset.as_array(axes=['position', 'time', 'channel', 'z'], position=position)
        else:
            if position not in (0, None):
                warnings.warn('Position index is not part of this dataset. Returning data at default position.')
            data = self.dataset.as_array(axes=['time', 'channel', 'z'])

        # data is a Dask array
        return data

    def get_array(self, position: int) -> np.ndarray:
        return np.asarray(self.get_zarr(position))

    def get_image_metadata(self, p, t, c, z) -> dict:
        metadata = None
        if 'position' not in self._axes.keys():
            p = None

        if self.dataset.has_image(position=p, time=t, channel=c, z=z):
            metadata = self.dataset.read_metadata(position=p, time=t, channel=c, z=z)

        return metadata
