import numpy as np
import os
import zarr
from tifffile import TiffFile


class MicromanagerReader:

    def __init__(self, folder, reader='tifffile'):
        """
        loads ome-tiff files in folder into zarr or numpy arrays
        Strategy:
            1. read all files in folder
            2. search for file which references the most series in the record -> this is the master ome-tiff
            3. each series represents a micro-manager stage position.  upon call for data, assign this data to a dict

        # todo: check that dict assignment of zarr is memory-ok.  If not, see how aicsimageio does it
        # todo: towards integration with aicsimageio: look up
            differences between ome.tiff meta and micromanager meta

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
        self._set_mm_meta()

    def _set_mm_meta(self):
        with TiffFile(self.master_ome_tiff) as tif:
            self.mm_meta = tif.micromanager_metadata

            if self.mm_meta['Summary']['Positions'] > 1:
                self.stage_positions = []
                for p in range(self.mm_meta['Summary']['Positions']):
                    print(f"appending {self.mm_meta['Summary']['Positions'][p]}")
                    self.stage_positions.append(self.mm_meta['Summary']['Positions'][p])

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
        search for tifffile that contains the most series references, this is the master-ome
            if the folder contains multiple files of one series, any of those files can serve as master-ome
        :param folder_: full path to folder containing images
        :return: full path to master-ome tiff
        """
        series_count = 0
        ome_master = None
        for file in os.listdir(folder_):
            print(file)
            if not file.endswith('.ome.tif'):
                continue
            with TiffFile(os.path.join(folder_, file)) as tif:
                print(tif.series)
                if len(tif.series) > series_count:
                    series_count = len(tif.series)
                    ome_master = file
        return os.path.join(folder_, ome_master)

    def get_zarr(self, position=0):
        if len(self._positions) == 0:
            self._extract_data(self.master_ome_tiff)
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
#     multipositions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_stack_tpzc_50tp_4p_5z_3c_2k_1'
#
#     r = MicromanagerReader(multipositions)
#     r.get_zarr()
#     print(r.get_master_ome())
#     print(r.get_num_positions())
#
#
# if __name__ == "__main__":
#     main()