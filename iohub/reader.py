import numpy as np
import os
import zarr
from tifffile import TiffFile
from copy import copy
import logging


# replicate from aicsimageio logging mechanism
###############################################################################

# modify the logging.ERROR level lower for more info
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG
# NOTSET
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


class MicromanagerReader:

    def __init__(self, folder: str, extract_data: bool = False, log_level: int = logging.ERROR):
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

        logging.basicConfig(
            level=log_level,
            format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
        )
        self.log = logging.getLogger(__name__)

        self._positions = {}
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
        with TiffFile(self.master_ome_tiff) as tif:
            self.mm_meta = tif.micromanager_metadata

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

    def _create_stores(self, master_ome):
        """
        extract all series from ome-tiff and place into dict of (pos: zarr)
        :param master_ome: full path to master OME-tiff
        :return:
        """
        log.info(f"extracting data from {master_ome}")
        with TiffFile(master_ome) as tif:
            for idx, tiffpageseries in enumerate(tif.series):
                self._positions[idx] = zarr.open(tiffpageseries.aszarr(), mode='r')

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

        log.info(f"checking {file} for ome-master records")

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
                    log.error(f"Exception while parsing root from omexml: {ex}")

            # search all elements for tags that identify ome-master tiff
            for element in root:
                # MetadataFile attribute identifies master-ome from a BinaryOnly non-master file
                if element.tag.endswith('BinaryOnly'):
                    log.warning(f'OME series: BinaryOnly: not an ome-tiff master file')
                    ome_master = element.attrib['MetadataFile']
                    return os.path.join(dirname, ome_master)
                # Name attribute identifies master-ome from a master-ome file.
                elif element.tag.endswith("Image"):
                    log.warning(f'OME series: Master-ome found')
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
        if not self._positions:
            self._create_stores(self.master_ome_tiff)
        return self._positions[position]

    def get_array(self, position):
        """
        return a numpy array for a given position
        :param position: int
            position (aka ome-tiff scene)
        :return: np.ndarray
        """
        if not self._positions:
            self._create_stores(self.master_ome_tiff)
        return np.array(self._positions[position])

    def get_num_positions(self):
        """
        get total number of scenes referenced in ome-tiff metadata
        :return: int
        """
        if self._positions:
            return len(self._positions)
        else:
            log.error("ome-tiff scenes not read.")

    @property
    def shape(self):
        """
        return the underlying data shape as a tuple
        :return: tuple
        """
        return self.frames, self.slices, self.channels, self.height, self.width


# def main():
#     no_positions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_files_tpzc_200tp_1p_5z_3c_2k_1'
#     # multipositions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_stack_tpzc_50tp_4p_5z_3c_2k_1'
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
#     r = MicromanagerReader(ivan_dataset)
#     print(r.get_zarr(3))
#     # print(r.get_master_ome())
#     # print(r.get_num_positions())
#
#
# if __name__ == "__main__":
#     main()
