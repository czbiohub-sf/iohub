import numpy as np
import os
import zarr
from copy import copy
from waveorder.io.reader_interface import ReaderInterface


"""
This reader is written to load data that has been directly streamed from micro-manager
    to a zarr array.  The array dims order is based on the micro-manager order
    
by default supports READ ONLY MODE
"""


class ZarrReader(ReaderInterface):

    def __init__(self,
                 zarrfile: str):

        # zarr files (.zarr) are directories
        if not os.path.isdir(zarrfile):
            raise ValueError("file is not a .zarr file")
        if not '.zarr' in zarrfile:
            raise ValueError("file is not a .zarr file")

        self.zf = zarrfile
        self.store = zarr.open(self.zf, 'r')
        self.plate_meta = self.store.attrs.get('plate')
        self._get_rows()
        self._get_columns()
        self._get_wells()
        self.position_map = dict()
        self._get_positions()
        self.mm_meta = None
        self._set_mm_meta()

        # structure of zarr array
        (self.frames,
         self.channels,
         self.slices,
         self.height,
         self.width) = self.store[self.wells[0]]['array'].shape
        self.positions = len(self.position_map)
        self.channel_names = []
        self.stage_positions = 0
        self.z_step_size = None

    def _get_rows(self):
        rows = []
        for row in self.plate_meta['rows']:
            rows.append(row['name'])
        self.rows = rows

    def _get_columns(self):
        columns = []
        for column in self.plate_meta['columns']:
            columns.append(column['name'])
        self.columns = columns

    def _get_wells(self):
        wells = []
        for well in self.plate_meta['wells']:
            wells.append(well['path'])
        self.wells = wells

    def _get_positions(self):

        idx = 0
        # Assumes that the positions are indexed in the order of Row-->Well-->FOV
        for well in self.wells:
            for pos in self.store[well].attrs.get('well').get('images'):
                name = pos['path']
                self.position_map[idx] = {'name': name, 'well': well}
                idx += 1

    def _set_mm_meta(self):
        self.mm_meta = self.store.attrs.get('Summary')

        mm_version = self.mm_meta['MicroManagerVersion']
        if 'beta' in mm_version:
            if self.mm_meta['Positions'] > 1:
                self.stage_positions = []

                for p in range(len(self.mm_meta['StagePositions'])):
                    pos = self._simplify_stage_position_beta(self.mm_meta['StagePositions'][p])
                    self.stage_positions.append(pos)

        elif mm_version == '1.4.22':
            for ch in self.mm_meta['ChNames']:
                self.channel_names.append(ch)
        else:
            if self.mm_meta['Positions'] > 1:
                self.stage_positions = []

                for p in range(self.mm_meta['Positions']):
                    pos = self._simplify_stage_position(self.mm_meta['StagePositions'][p])
                    self.stage_positions.append(pos)

            for ch in self.mm_meta['ChNames']:
                self.channel_names.append(ch)

        self.z_step_size = self.mm_meta['z-step_um']

    def _simplify_stage_position(self, stage_pos: dict):
        """
        flattens the nested dictionary structure of stage_pos and removes superfluous keys

        Parameters
        ----------
        stage_pos:      (dict) dictionary containing a single position's device info

        Returns
        -------
        out:            (dict) flattened dictionary
        """

        out = copy(stage_pos)
        out.pop('DevicePositions')
        for dev_pos in stage_pos['DevicePositions']:
            out.update({dev_pos['Device']: dev_pos['Position_um']})
        return out

    def _simplify_stage_position_beta(self, stage_pos: dict):
        """
        flattens the nested dictionary structure of stage_pos and removes superfluous keys
        for MM2.0 Beta versions

        Parameters
        ----------
        stage_pos:      (dict) dictionary containing a single position's device info

        Returns
        -------
        new_dict:       (dict) flattened dictionary

        """

        new_dict = {}
        new_dict['Label'] = stage_pos['label']
        new_dict['GridRow'] = stage_pos['gridRow']
        new_dict['GridCol'] = stage_pos['gridCol']

        for sub in stage_pos['subpositions']:
            values = []
            for field in ['x', 'y', 'z']:
                if sub[field] != 0:
                    values.append(sub[field])
            if len(values) == 1:
                new_dict[sub['stageName']] = values[0]
            else:
                new_dict[sub['stageName']] = values

        return new_dict

    def get_image_plane_metadata(self, coord):
        # coord must be (p, t, c, z)
        coord_str = f'({coord[0]}, {coord[1]}, {coord[2]}, {coord[3]})'
        return self.store.attrs.get('ImagePlaneMetadata').get(coord_str)

    def get_zarr(self, pt: tuple) -> zarr.array:
        pos_info = self.position_map[pt[0]]
        well = pos_info['well']
        return self.store[well]

    def get_array(self, pt: tuple) -> np.ndarray:
        well = self.get_zarr(pt)
        return well['array']

    def get_num_positions(self) -> int:
        return self.positions

    @property
    def shape(self):
        return self.frames, self.channels, self.slices, self.height, self.width

