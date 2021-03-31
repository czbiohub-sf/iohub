import numpy as np
import os
import zarr
from tifffile import TiffFile
import tifffile as tiff
from copy import copy
import logging

# libraries for singlepage tiff sequence reading
import glob
import json
import natsort


# replicate from aicsimageio logging mechanism
###############################################################################

# modify the logging.ERROR level lower for more info
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG
# NOTSET
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
# )
# log = logging.getLogger(__name__)

###############################################################################


class MicromanagerReader:

    def __init__(self,
                 src: str,
                 data_type: str,
                 extract_data: bool = False,
                 log_level: int = logging.ERROR):
        """
        reads ome-tiff files into zarr or numpy arrays
        Strategy:
            1. search the file's omexml metadata for the "master file" location
            2. load the master file
            3. read micro-manager metadata into class attributes

        :param src: str
            folder or file containing all ome-tiff files
        :param data_type: str
            whether data is 'ometiff', 'singlepagetiff', 'zarr'
        :param extract_data: bool
            True if ome_series should be extracted immediately
        :param log_level: int
            One of 0, 10, 20, 30, 40, 50 for NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL respectively

        """

        logging.basicConfig(
            level=log_level,
            format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
        )
        self.log = logging.getLogger(__name__)

        # identify data structure type
        if data_type == 'ometiff':
            self.reader = MicromanagerOmeTiffReader(src, extract_data)
        elif data_type == 'singlepagetiff':
            self.reader = MicromanagerSequenceReader(src, extract_data)
        else:
            raise NotImplementedError(f"reader of type {data_type} is not implemented")

        self.mm_meta = self.reader.mm_meta
        self.stage_positions = self.reader.stage_positions
        self.height = self.reader.height
        self.width = self.reader.width
        self.frames = self.reader.frames
        self.slices = self.reader.slices
        self.channels = self.reader.channels
        self.channel_names = self.reader.channel_names

    def get_zarr(self, position):
        """
        return a zarr array for a given position
        :param position: int
            position (aka ome-tiff scene)
        :return: zarr.array
        """
        return self.reader.get_zarr(position)

    def get_array(self, position):
        """
        return a numpy array for a given position
        :param position: int
            position (aka ome-tiff scene)
        :return: np.ndarray
        """
        return self.reader.get_array(position)

    def get_num_positions(self):
        """
        get total number of scenes referenced in ome-tiff metadata
        :return: int
        """
        return self.reader.get_num_positions()

    @property
    def shape(self):
        """
        return the underlying data shape as a tuple
        :return: tuple
        """
        return self.frames, self.slices, self.channels, self.height, self.width


class MicromanagerOmeTiffReader:

    def __init__(self,
                 folder: str,
                 extract_data: bool = False):
        """
        reads ome-tiff files into zarr or numpy arrays
        Strategy:
            1. search the file's omexml metadata for the "master file" location
            2. load the master file
            3. read micro-manager metadata into class attributes

        :param folder: str
            folder or file containing all ome-tiff files
        :param extract_data: bool
            True if ome_series should be extracted immediately
        :param log_level: int
            One of 0, 10, 20, 30, 40, 50 for NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL respectively

        """

        self.log = logging.getLogger(__name__)

        self.positions = {}
        self.mm_meta = None
        self.stage_positions = 0
        self.height = 0
        self.width = 0
        self.frames = 0
        self.slices = 0
        self.channels = 0
        self.channel_names = []

        self.master_ome_tiff = self._get_master_ome_tiff(folder)
        self._set_mm_meta()

        if extract_data:
            self._create_stores(self.master_ome_tiff)

    def _set_mm_meta(self):
        """
        assign image metadata from summary metadata
        :return:
        """
        with TiffFile(self.master_ome_tiff) as tif:
            self.mm_meta = tif.micromanager_metadata

            if 'beta' in self.mm_meta['Summary']['MicroManagerVersion']:

                if self.mm_meta['Summary']['Positions'] > 1:
                    self.stage_positions = []

                    for p in range(len(self.mm_meta['Summary']['StagePositions'])):
                        pos = self._simplify_stage_position_beta(self.mm_meta['Summary']['StagePositions'][p])
                        self.stage_positions.append(pos)

                self.channel_names = 'Not Listed'

            else:
                if self.mm_meta['Summary']['Positions'] > 1:
                    self.stage_positions = []

                    for p in range(self.mm_meta['Summary']['Positions']):
                        pos = self._simplify_stage_position(self.mm_meta['Summary']['StagePositions'][p])
                        self.stage_positions.append(pos)

                for ch in self.mm_meta['Summary']['ChNames']:
                    self.channel_names.append(ch)

            self.height = self.mm_meta['Summary']['Height']
            self.width = self.mm_meta['Summary']['Width']
            self.frames = self.mm_meta['Summary']['Frames']
            self.slices = self.mm_meta['Summary']['Slices']
            self.channels = self.mm_meta['Summary']['Channels']

    def _simplify_stage_position(self, stage_pos: dict):
        """
        flattens the nested dictionary structure of stage_pos and removes superfluous keys
        :param stage_pos: dictionary containing a single position's device info
        :return:
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
        :param stage_pos: dictionary containing a single position's device info
        :return:
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

    def _create_stores(self, master_ome):
        """
        extract all series from ome-tiff and place into dict of (pos: zarr)
        :param master_ome: full path to master OME-tiff
        :return:
        """
        self.log.info(f"extracting data from {master_ome}")
        with TiffFile(master_ome) as tif:
            for idx, tiffpageseries in enumerate(tif.series):
                self.positions[idx] = zarr.open(tiffpageseries.aszarr(), mode='r')

    def _get_master_ome_tiff(self, folder_):
        """
        given either a single ome.tiff or a folder of ome.tiffs
            load the omexml metadata for a single file and
            search for the element attribute corresponding to the master file

        :param folder_: full path to folder containing images
        :return: full path to master-ome tiff
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
        :param position: int
            position (aka ome-tiff scene)
        :return: zarr.array
        """
        if not self.positions:
            self._create_stores(self.master_ome_tiff)
        return self.positions[position]

    def get_array(self, position):
        """
        return a numpy array for a given position
        :param position: int
            position (aka ome-tiff scene)
        :return: np.ndarray
        """
        if not self.positions:
            self._create_stores(self.master_ome_tiff)
        return np.array(self.positions[position])

    def get_num_positions(self):
        """
        get total number of scenes referenced in ome-tiff metadata
        :return: int
        """
        if self.positions:
            return len(self.positions)
        else:
            self.log.error("ome-tiff scenes not read.")

    @property
    def shape(self):
        """
        return the underlying data shape as a tuple
        :return: tuple
        """
        return self.frames, self.slices, self.channels, self.height, self.width


class MicromanagerSequenceReader:

    def __init__(self,
                 folder,
                 extract_data):
        """
        reads single-page tiff files generated by micro-manager into zarr or numpy arrays
        Strategy:
            1. Gather summary metadata from any metadata.txt
            2. Build a map between image coordinates and image file names
            3. Upon call to extract data, assign each scene to self.positions (similar to OmeTiffReader)

        :param folder: str
            folder containing position subdirectories, which contain singlepage tiff sequences
        :param extract_data: bool
            True if zarr arrays should be extracted immediately
        """

        if not os.path.isdir(folder):
            raise NotImplementedError("supplied path for singlepage tiff sequence reader is not a folder")

        self.log = logging.getLogger(__name__)
        self.positions = {}
        self.mm_meta = None
        self.stage_positions = 0
        self.height = 0
        self.width = 0
        self.frames = 0
        self.slices = 0
        self.channels = 0
        self.channel_names = []

        self.coord_to_filename = {}

        # identify type of subdirectory
        sub_dirs = self._get_sub_dirs(folder)
        if sub_dirs:
            pos_path = os.path.join(folder, sub_dirs[0])
        else:
            raise AttributeError("supplied folder does not contain position or default subdirectories")

        # pull one metadata sample and extract experiment dimensions
        metadata_path = os.path.join(pos_path, 'metadata.txt')
        with open(metadata_path, 'r') as f:
            self.mm_meta = json.load(f)

        self.mm_version = self.mm_meta['Summary']['MicroManagerVersion']
        if self.mm_version == '1.4.22':
            self._mm1_meta_parser()
        elif 'beta' in self.mm_version:
            self._mm2beta_meta_parser()
        elif 'gamma' in self.mm_version:
            self._mm2gamma_meta_parser()
        else:
            raise NotImplementedError(
                f'Current MicroManager reader only supports version 1.4.22 and 2.0 but {self.mm_version} was detected')

        # create coordinate to filename maps
        self.coord_to_filename = self.read_tiff_series(folder)

        if extract_data:
            self._create_stores()

    def get_zarr(self, position_):
        if not self.positions:
            self._create_stores()
        return self.positions[position_]

    def get_array(self, position_):
        if not self.positions:
            self._create_stores()
        return np.array(self.positions[position_])

    def get_num_positions(self):
        if self.positions:
            return len(self.positions)
        else:
            self.log.error("singlepage tiffs not loaded")

    def _create_stores(self):
        """
        extract all singlepage tiffs at each coordinate and place them in a zarr array
        coordinates are of shape = (pos, time, channel, z)
        :return:
        """
        self.log.info("")
        z = zarr.zeros(shape=(self.frames,
                              self.channels,
                              self.slices,
                              self.height,
                              self.width),
                       chunks=(1,
                               1,
                               1,
                               self.height,
                               self.width))
        for c, fn in self.coord_to_filename.items():
            self.log.info(f"reading coord = {c} from filename = {fn}")
            z[c[1], c[2], c[3]] = zarr.open(tiff.imread(fn, aszarr=True))
            self.positions[c[0]] = z

    def read_tiff_series(self, folder: str):
        """
        given a folder containing position subfolders, each of which contains
            single-page-tiff series acquired in mm2.0 gamma, parse the metadata
            to map image coordinates to filepaths/names
        :param folder: str
        :return: dict
            keys are coordinates and values are filenames.  Coordinates follow (p, t, c, z) indexing.
        """
        positions = [p for p in os.listdir(folder) if os.path.isdir(os.path.join(folder, p))]
        if not positions:
            raise FileNotFoundError("no position subfolder found in supplied folder")

        metadatas = [os.path.join(folder, position, 'metadata.txt') for position in positions]
        if not metadatas:
            raise FileNotFoundError("no metadata.txt file found in position directories")

        coord_filename_map = {}
        for idx, metadata in enumerate(metadatas):
            with open(metadata, 'r+') as m:
                j = json.load(m)
                coord_filename_map.update(self._extract_coord_to_filename(j,
                                                                          folder,
                                                                          positions[idx]))

        return coord_filename_map

    def _extract_coord_to_filename(self,
                                   json_,
                                   parent_folder,
                                   position=None):
        """
        given a micro-manager generated metadata json, extract image coordinates and their corresponding image filepaths
        build a mapping between the two.
        :param json_: dict
            dict generated from json.load
        :param parent_folder: str
            full path to file
        :param position: str
            mm1.4.22 metadata does not associate positions with images in the metadata.  This has to be provided.
        :return:
        """
        coords = set()
        meta = dict()

        # separate coords from meta
        for element in json_.keys():
            # present for mm2-gamma metadata
            if "Coords" in element:
                coords.add(element)
            if "Metadata" in element:
                meta[element.split('-')[2]] = element

            # present in mm1.4.22 metadata
            if "FrameKey" in element:
                coords.add(element)

        if not coords:
            raise ValueError("no image coordinates present in metadata")

        # build a dict of coord to filename maps
        coord_to_filename = dict()
        for c in coords:
            # indices common to both mm2 and mm1
            ch_idx = json_[c]['ChannelIndex']
            pos_idx = json_[c]['PositionIndex']
            time_idx = json_[c]['FrameIndex']
            z_idx = json_[c]['SliceIndex']

            # extract filepath for this coordinate
            try:
                # for mm2-gamma. filename contains position folder
                if c.split('-')[2] in meta:
                    filepath = json_[meta[c.split('-')[2]]]['FileName']
                # for mm1, file name does not contain position folder
                else:
                    filepath = json_[c]['FileName']
                    filepath = os.path.join(position, filepath)  # position name is not present in metadata
            except KeyError as ke:
                self.log.error(f"metadata for supplied image coordinate {c} not found")
                raise ke

            coordinate = (pos_idx, time_idx, ch_idx, z_idx)
            coord_to_filename[coordinate] = os.path.join(parent_folder, filepath)

        return coord_to_filename

    def _get_sub_dirs(self, f):
        """
        subdir walk
        from https://github.com/mehta-lab/reconstruct-order

        :param f: str
        :return: list
        """
        sub_dir_path = glob.glob(os.path.join(f, '*/'))
        sub_dir_name = [os.path.split(subdir[:-1])[1] for subdir in sub_dir_path]
        #    assert subDirName, 'No sub directories found'
        return natsort.natsorted(sub_dir_name)

    def _mm1_meta_parser(self):
        """
        set image metadata.
        from https://github.com/mehta-lab/reconstruct-order

        :return:
        """
        self.width = self.mm_meta['Summary']['Width']
        self.height = self.mm_meta['Summary']['Height']
        self.frames = self.mm_meta['Summary']['Frames']
        self.slices = self.mm_meta['Summary']['Slices']
        self.channels = self.mm_meta['Summary']['Channels']

    def _mm2beta_meta_parser(self):
        """
        set image metadata
        from https://github.com/mehta-lab/reconstruct-order
        :return:
        """
        self.width = int(self.mm_meta['Summary']['UserData']['Width']['PropVal'])
        self.height = int(self.mm_meta['Summary']['UserData']['Height']['PropVal'])
        self.time_stamp = self.mm_meta['Summary']['StartTime']

    def _mm2gamma_meta_parser(self):
        """
        set image metadata
        from https://github.com/mehta-lab/reconstruct-order
        :return:
        """
        keys_list = list(self.mm_meta.keys())
        if 'FrameKey-0-0-0' in keys_list[1]:
            roi_string = self.mm_meta[keys_list[1]]['ROI']
            self.width = int(roi_string.split('-')[2])
            self.height = int(roi_string.split('-')[3])
        elif 'Metadata-' in keys_list[2]:
            self.width = self.mm_meta[keys_list[2]]['Width']
            self.height = self.mm_meta[keys_list[2]]['Height']
        else:
            raise ValueError('Metadata file incompatible with metadata reader')
        self.frames = self.mm_meta['Summary']['Frames']
        self.slices = self.mm_meta['Summary']['Slices']
        self.channels = self.mm_meta['Summary']['Channels']


# def main():
#     no_positions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_files_tpzc_200tp_1p_5z_3c_2k_1'
#     multipositions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_stack_tpzc_50tp_4p_5z_3c_2k_1'
#
#     master_new_folder = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/test_1/'
#     non_master_new_folder = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/test_1/'
#     non_master_new_large_folder = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_stack_tpzc_50tp_4p_5z_3c_2k_1/'
#     non_master_old_large_folder = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_50tp_4p_5z_3c_2k_1/'
#
#     master_old_folder = '/Volumes/comp_micro/rawdata/hummingbird/Janie/2021_02_03_40x_04NA_A549/48hr_RSV_IFN/Coverslip_1/C1_MultiChan_Stack_1/'
#     non_master_old_folder = '/Volumes/comp_micro/rawdata/hummingbird/Janie/2021_02_03_40x_04NA_A549/48hr_RSV_IFN/Coverslip_1/C1_MultiChan_Stack_1/'
#
#     ivan_dataset = '/Volumes/comp_micro/rawdata/falcon/Ivan/20210128 HEK CAAX SiRActin/FOV1_1'
#     ivan_file = 'FOV1_1_MMStack_Default_23.ome.tif'
#
#     mm1_single = '/Users/bryant.chhun/Desktop/mm2_sampledata/packaged for gdd/mm1422_kazansky_one_position/mm1422_kazansky_one_position'
#     mm1_multi_snake = '/Users/bryant.chhun/Desktop/mm2_sampledata/packaged for gdd/mm1422_kazansky_HCS_snake/mm1422_kazansky_HCS_snake'
#     mm1_multi_grid = '/Users/bryant.chhun/Desktop/mm2_sampledata/packaged for gdd/mm1422_kazansky_grid/mm1422_kazansky_grid'
#     mm1_multi_large = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm1422/autosave_mm1422_50tp_4p_3c_2k_1'
#
#     r = MicromanagerReader(non_master_old_large_folder,
#                            data_type='ometiff',
#                            extract_data=True)
#     print(r.get_zarr(0))
#     # print(r.get_master_ome())
#     # print(r.get_num_positions())
#
#
# if __name__ == "__main__":
#     main()
