import numpy as np
import os
import zarr
from tifffile import TiffFile
from copy import copy
import logging

from waveorder.io.reader_interface import ReaderInterface


class MicromanagerOmeTiffReader(ReaderInterface):

    def __init__(self,
                 folder: str,
                 extract_data: bool = False):
        """
        reads ome-tiff files into zarr or numpy arrays
        Strategy:
            1. search the file's omexml metadata for the "master file" location
            2. load the master file
            3. read micro-manager metadata into class attributes

        Parameters
        ----------
        folder:         (str) folder or file containing all ome-tiff files
        extract_data:   (bool) True if ome_series should be extracted immediately

        """

        self.log = logging.getLogger(__name__)

        self.positions = {}
        self.mm_meta = None
        self.stage_positions = 0
        self.z_step_size = None
        self.height = 0
        self.width = 0
        self.frames = 0
        self.slices = 0
        self.channels = 0
        self.channel_names = []

        self._missing_dims = None

        self.master_ome_tiff = self._get_master_ome_tiff(folder)
        self._set_mm_meta()

        if extract_data:
            self._create_stores(self.master_ome_tiff)

    def _set_mm_meta(self):
        """
        assign image metadata from summary metadata

        Returns
        -------

        """
        with TiffFile(self.master_ome_tiff) as tif:
            self.mm_meta = tif.micromanager_metadata

            mm_version = self.mm_meta['Summary']['MicroManagerVersion']
            if 'beta' in mm_version:
                if self.mm_meta['Summary']['Positions'] > 1:
                    self.stage_positions = []

                    for p in range(len(self.mm_meta['Summary']['StagePositions'])):
                        pos = self._simplify_stage_position_beta(self.mm_meta['Summary']['StagePositions'][p])
                        self.stage_positions.append(pos)
                # self.channel_names = 'Not Listed'

            elif mm_version == '1.4.22':
                for ch in self.mm_meta['Summary']['ChNames']:
                    self.channel_names.append(ch)

            else:
                if self.mm_meta['Summary']['Positions'] > 1:
                    self.stage_positions = []

                    for p in range(self.mm_meta['Summary']['Positions']):
                        pos = self._simplify_stage_position(self.mm_meta['Summary']['StagePositions'][p])
                        self.stage_positions.append(pos)

                for ch in self.mm_meta['Summary']['ChNames']:
                    self.channel_names.append(ch)

            # dimensions based on mm metadata do not reflect final written dimensions
            # these will change after data is loaded
            self.z_step_size = self.mm_meta['Summary']['z-step_um']
            self.height = self.mm_meta['Summary']['Height']
            self.width = self.mm_meta['Summary']['Width']
            self.frames = self.mm_meta['Summary']['Frames']
            self.slices = self.mm_meta['Summary']['Slices']
            self.channels = self.mm_meta['Summary']['Channels']

            self._check_missing_dims()

    def _check_missing_dims(self):
        """
        establishes which dimensions are not present in the data

        Returns
        -------

        """

        missing_coords = []
        if self.frames == 1:
            missing_coords.append('T')
        if self.slices == 1:
            missing_coords.append('Z')
        if self.channels == 1:
            missing_coords.append('C')
        if bool(missing_coords) is True:
            self._missing_dims = set(missing_coords)

    def _reshape_zarr(self, zar):
        """
        reshape zarr arrays to match (T, C, Z, Y, X)
        if zarr array is lower dimensional, reshape to match the target
        if zarr array is purely 2 or 3 dimensional, no need to reshape

        Parameters
        ----------
        zar:        (zarr.array)

        Returns
        -------
        zar:        (zarr.array)

        """

        if self._missing_dims is None:
            target = np.array(zar).reshape((self.frames, self.channels, self.slices, self.height, self.width))
            return zarr.array(target, chunks=(1, 1, 1, self.height, self.width))

        elif {'T'} == self._missing_dims:
            target = np.array(zar).reshape((self.channels, self.slices, self.height, self.width))
            return zarr.array(target, chunks=(1, 1, self.height, self.width))

        # at least one channel is always present
        # elif {'C'} == self._missing_dims:
        #     target = np.array(zar).reshape((self.frames, self.slices, self.height, self.width))

        elif {'Z'} == self._missing_dims:
            target = np.array(zar).reshape((self.frames, self.channels, self.height, self.width))
            return zarr.array(target, chunks=(1, 1, self.height, self.width))

        else:
            return zar

    def _expand_zarr(self, zar):
        """
        takes a zarr array and, if necessary, expands it to include missing dimensions
        returns zarr array of dims (T, C, Z, Y, X)

        Parameters
        ----------
        zar:        (zarr.array)

        Returns
        -------
        target:     (zarr.array)
        """

        # major assumption -- that supplied zar is always shaped as (T, C, Z, Y, X)
        if self._missing_dims is None:
            return zar

        elif {'T'} == self._missing_dims:
            target = zarr.empty(shape=(1, self.channels, self.slices, self.height, self.width),
                                chunks=(1, 1, 1, self.height, self.width))
            target[0, :, :, :, :] = zar

        elif {'C'} == self._missing_dims:
            target = zarr.empty(shape=(self.frames, 1, self.slices, self.height, self.width),
                                chunks=(1, 1, 1, self.height, self.width))
            target[:, 0, :, :, :] = zar

        elif {'Z'} == self._missing_dims:
            target = zarr.empty(shape=(self.frames, self.channels, 1, self.height, self.width),
                                chunks=(1, 1, 1, self.height, self.width))
            target[:, :, 0, :, :] = zar

        elif {'T', 'C'} == self._missing_dims:
            target = zarr.empty(shape=(1, 1, self.slices, self.height, self.width),
                                chunks=(1, 1, 1, self.height, self.width))
            target[0, 0, :, :, :] = zar

        elif {'T', 'Z'} == self._missing_dims:
            target = zarr.empty(shape=(1, self.channels, 1, self.height, self.width),
                                chunks=(1, 1, 1, self.height, self.width))
            target[0, :, 0, :, :] = zar

        elif {'C', 'Z'} == self._missing_dims:
            target = zarr.empty(shape=(self.frames, 1, 1, self.height, self.width),
                                chunks=(1, 1, 1, self.height, self.width))
            target[:, 0, 0, :, :] = zar

        elif {'T', 'C', 'Z'} == self._missing_dims:
            target = zarr.empty(shape=(1, 1, 1, self.height, self.width),
                                chunks=(1, 1, 1, self.height, self.width))
            target[0, 0, 0, :, :] = zar

        else:
            raise ValueError("missing dims not properly identified")

        return target

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

    def _set_dims_ome_meta(self, tps):
        """
        read ome metadata from tiffpageseries and set class dims

        Parameters
        ----------
        tps: (TiffPageSeries): generated from tifffile.TiffFile.series

        Returns
        -------

        """
        shp = tps.shape
        # get_axes() returns strings like 'TCYX'
        for idx, dim in enumerate(tps.get_axes()):
            if 'T' in dim:
                self.frames = shp[idx]
            if 'C' in dim:
                self.channels = shp[idx]
            if 'Z' in dim:
                self.slices = shp[idx]
            if 'Y' in dim:
                self.height = shp[idx]
            if 'X' in dim:
                self.width = shp[idx]

    def _create_stores(self, master_ome):
        """
        extract all series from ome-tiff and place into dict of (pos: zarr)

        Parameters
        ----------
        master_ome:     (str): full path to master OME-tiff

        Returns
        -------

        """

        self.log.info(f"extracting data from {master_ome}")
        with TiffFile(master_ome) as tif:
            for idx, tiffpageseries in enumerate(tif.series):
                self._set_dims_ome_meta(tiffpageseries)
                z = zarr.open(tiffpageseries.aszarr(), mode='r')
                z = self._reshape_zarr(z)
                z = self._expand_zarr(z)
                self.positions[idx] = z

    def _get_master_ome_tiff(self, folder_):
        """
        given either a single ome.tiff or a folder of ome.tiffs
            load the omexml metadata for a single file and
            search for the element attribute corresponding to the master file

        Parameters
        ----------
        folder_:        (str) full path to folder containing images

        Returns
        -------
        path:           (str) full path to master-ome tiff
        """

        from xml.etree import ElementTree as etree  # delayed import

        ome_master = None

        if os.path.isdir(folder_):
            dirname = folder_
            file = [f for f in os.listdir(folder_) if ".ome.tif" in f][0]
        elif os.path.isfile(folder_) and folder_.endswith('.ome.tif'):
            dirname = os.path.dirname(folder_)
            file = folder_
        else:
            raise ValueError("supplied path contains no ome.tif or is itself not an ome.tif")

        self.log.info(f"checking {file} for ome-master records")

        with TiffFile(os.path.join(dirname, file)) as tiff:
            omexml = tiff.pages[0].description
            # get omexml root from first page
            try:
                root = etree.fromstring(omexml)
            except etree.ParseError as exc:
                try:
                    omexml = omexml.decode(errors='ignore').encode()
                    root = etree.fromstring(omexml)
                except Exception as ex:
                    self.log.error(f"Exception while parsing root from omexml: {ex}")

            # search all elements for tags that identify ome-master tiff
            for element in root:
                # MetadataFile attribute identifies master-ome from a BinaryOnly non-master file
                if element.tag.endswith('BinaryOnly'):
                    self.log.warning(f'OME series: BinaryOnly: not an ome-tiff master file')
                    ome_master = element.attrib['MetadataFile']
                    return os.path.join(dirname, ome_master)
                # Name attribute identifies master-ome from a master-ome file.
                elif element.tag.endswith("Image"):
                    self.log.warning(f'OME series: Master-ome found')
                    ome_master = element.attrib['Name'] + ".ome.tif"
                    return os.path.join(dirname, ome_master)

            if not ome_master:
                raise AttributeError("no ome-master file found")

    def get_zarr(self, position):
        """
        return a zarr array for a given position

        Parameters
        ----------
        position:       (int) position (aka ome-tiff scene)

        Returns
        -------
        position:       (zarr.array)

        """
        if not self.positions:
            self._create_stores(self.master_ome_tiff)
        return self.positions[position]

    def get_array(self, position):
        """
        return a numpy array for a given position

        Parameters
        ----------
        position:   (int) position (aka ome-tiff scene)

        Returns
        -------
        position:   (np.ndarray)

        """

        if not self.positions:
            self._create_stores(self.master_ome_tiff)
        return np.array(self.positions[position])

    def get_num_positions(self):
        """
        get total number of scenes referenced in ome-tiff metadata

        Returns
        -------
        number of positions     (int)

        """
        if self.positions:
            return len(self.positions)
        else:
            self.log.error("ome-tiff scenes not read.")

    @property
    def shape(self):
        """
        return the underlying data shape as a tuple

        Returns
        -------
        (tuple) five elements of (frames, slices, channels, height, width)

        """
        return self.frames, self.channels, self.slices, self.height, self.width
