import numpy as np
import os
import zarr
from tifffile import TiffFile


class MicromanagerReader:

    def __init__(self, folder):
        """
        loads ome-tiff files in folder into zarr or numpy arrays
        Strategy:
            1. read all files in folder
            2. search for the file whose omexml does not contain the element tag "BinaryOnly"-> this is the master ome-tiff
            3. each series represents a micro-manager stage position.  upon call for data, assign this data to a dict

        :param folder: folder containing all ome-tiff files
        :param reader: tifffile or aicsimageio
        """

        self._positions = {}
        self.mm_meta = None
        self.stage_positions = 0
        self.height = 0
        self.width = 0
        self.frames = 0
        self.slices = 0
        self.channels = 0
        self.chnames = []

        self.master_ome_tiff = self._get_master_ome_tiff(folder)
        print(self.master_ome_tiff)
        self._set_mm_meta()

    def _set_mm_meta(self):
        with TiffFile(self.master_ome_tiff) as tif:
            self.mm_meta = tif.micromanager_metadata

            if self.mm_meta['Summary']['Positions'] > 1:
                self.stage_positions = []
                for p in range(self.mm_meta['Summary']['Positions']):
                    print(f"appending {self.mm_meta['Summary']['StagePositions'][p]}")
                    self.stage_positions.append(self.mm_meta['Summary']['StagePositions'][p])

            for ch in self.mm_meta['Summary']['ChNames']:
                print(f"appending {ch}")
                self.chnames.append(ch)

            self.height = self.mm_meta['Summary']['Height']
            self.width = self.mm_meta['Summary']['Width']
            self.frames = self.mm_meta['Summary']['Frames']
            self.slices = self.mm_meta['Summary']['Slices']
            self.channels = self.mm_meta['Summary']['Channels']

    def _extract_data(self, master_ome):
        """
        extract all series from ome-tiff and place into dict of (pos: zarr)
        :param master_ome: full path to master OME-tiff
        :return:
        """
        with TiffFile(master_ome) as tif:
            for idx, tiffpageseries in enumerate(tif.series):
                self._positions[idx] = zarr.open(tiffpageseries.aszarr(), mode='r')

    def _get_master_ome_tiff(self, folder_):
        """
        search all ome.tif in directory
        check the omexml for element tag corresponding to companion files (not-master-ome)

        :param folder_: full path to folder containing images
        :return: full path to master-ome tiff
        """

        from xml.etree import ElementTree as etree  # delayed import

        def tag_search(root_, tag_name='BinaryOnly'):
            """
            returns True if tag_name is present
            """
            for element in root_:
                if element.tag.endswith(tag_name):
                    # todo: question: can we load a full OME-XML from companion file rather than search all files?
                    print(f'OME series: not an ome-tiff master file')
                    return True
            return False

        for file in os.listdir(folder_):
            print(f"checking {file} for ome-master records")
            if not file.endswith('.ome.tif'):
                continue
            with TiffFile(os.path.join(folder_, file)) as tiff:
                omexml = tiff.pages[0].description
                # get omexml root from first page
                try:
                    root = etree.fromstring(omexml)
                except etree.ParseError as exc:
                    try:
                        omexml = omexml.decode(errors='ignore').encode()
                        root = etree.fromstring(omexml)
                    except Exception as ex:
                        print(f"Exception while parsing root from omexml: {ex}")

                # search for tag corresponding to non-ome-tiff files
                if not tag_search(root, "BinaryOnly"):
                    ome_master = file
                else:
                    continue

        return os.path.join(folder_, ome_master)

    def get_zarr(self, position=0):
        if len(self._positions) == 0:
            self._extract_data(self.master_ome_tiff)
        # if no position specified, return a full zarr array containing all positions
        # if position == -1:
        #     print("no position specified, returning full dataset")
        #     full_shape = (len(self._positions), ) + self._positions[0].shape
        #     working_z = zarr.empty(full_shape, chunks=(1, )*(len(full_shape)-2)+full_shape[-2:])
        #     for pos in range(len(self._positions)):
        #         working_z[pos] = self._positions[pos]
        #         return working_z
        # else:
        #     return self._positions[position]
        return self._positions[position]

    def get_array(self, position=0):
        if len(self._positions) == 0:
            self._extract_data(self.master_ome_tiff)
        return np.array(self._positions[position])

    def get_master_ome(self):
        return self.master_ome_tiff

    def get_num_positions(self):
        return len(self._positions)


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
#     r = MicromanagerReader(non_master_old_large_folder)
#     print(r.get_zarr(3))
#     # print(r.get_master_ome())
#     # print(r.get_num_positions())
#
#
# if __name__ == "__main__":
#     main()
