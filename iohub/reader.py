import numpy as np
import os
import zarr


class OmeTiffReader:

    def __init__(self, folder, reader='tifffile'):
        """
        loads ome-tiff files in folder into zarr or numpy arrays
        Strategy:
            1. read all files in folder
            2. search for file which references the most series in the record -> this is the master ome-tiff
            3. each series represents a micro-manager stage position.  upon call for data, assign this data to a dict

        :param folder: folder containing all ome-tiff files
        :param reader: tifffile or aicsimageio
        """

        self.positions = {}
        self.mm_meta = None
        self.dim_order = {}
        self.dtypes = {}

        if reader == 'tifffile':
            import tifffile as tf
            self.reader = tf
        elif reader == 'aicsimageio':
            raise NotImplementedError("aicsimageio is not implemented yet")
        else:
            raise NotImplementedError(f"reader {reader} is not implemented")

        self.master_ome_tiff = self._get_master_ome_tiff(folder)

    def _extract_data(self, master_ome):
        """
        extract all series from ome-tiff and place into dict of (pos: zarr)
        :param master_ome: full path to master OME-tiff
        :return:
        """
        with self.reader.TiffFile(master_ome) as tif:
            for idx, tiffpageseries in enumerate(tif.series):
                self.positions[idx] = zarr.open(tiffpageseries.aszarr(), mode='r')
                self.dim_order[idx] = tiffpageseries.axes
                self.dtypes[idx] = tiffpageseries.dtype

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
            with self.reader.TiffFile(os.path.join(folder_, file)) as tif:
                print(tif.series)
                if len(tif.series) > series_count:
                    series_count = len(tif.series)
                    ome_master = file
        return os.path.join(folder_, ome_master)

    def get_zarr(self, position=0):
        if len(self.positions) == 0:
            self._extract_data(self.master_ome_tiff)
        return self.positions[position]

    def get_array(self, position=0):
        if len(self.positions) == 0:
            self._extract_data(self.master_ome_tiff)
        return np.array(self.positions[position])

    def get_order(self, position=0):
        if len(self.positions) == 0:
            self._extract_data(self.master_ome_tiff)
        return self.dim_order[position]

    def get_dtype(self, position=0):
        if len(self.positions) == 0:
            self._extract_data(self.master_ome_tiff)
        return self.dtypes[position]

    def get_master_ome(self):
        return self.master_ome_tiff

    def get_num_positions(self):
        return len(self.positions)


def main():
    no_positions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_files_tpzc_200tp_1p_5z_3c_2k_1'
    multipositions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_stack_tpzc_50tp_4p_5z_3c_2k_1'

    r = OmeTiffReader(multipositions)
    r.get_zarr()
    print(r.get_master_ome())
    print(r.get_num_positions())


if __name__ == "__main__":
    main()

# """
# Strategy:
# - class has methods that enable other readers (aicsimageio, tifffile-zarr == maybe not needed, but enables fast swap later)
#
# - standardize the array dimension order
# - class enables slicing against a dimension
# - class enables position list loading into zarr
#
# """

# class Read:
#
#     def __init__(self, f, reader='tifffile'):
#         """
#
#         :param f:
#         :param reader: tifffile or aicsimageio
#         """
#
#         # strategy
#         """
#         ** build a reference for position_index-filename-zarr **
#
#         ( do string parsing to extract only the "master" position file )
#         1. read file, extract mm-meta
#         2. from mm-meta find number of positions
#         3. read same file, extract ome-meta
#         4. from ome-meta, find file name
#         5. given above 1-4, assign
#         """
#
#         if reader == 'tifffile':
#             import tifffile as tf
#             self.reader = tf
#         elif reader == 'aicsimageio':
#             raise NotImplementedError("aicsimageio is not implemented yet")
#         else:
#             raise NotImplementedError(f"reader {reader} is not implemented")
#
#         self.folder = f
#
#         self.positions_filenames = set()
#         self.positions = {}  # index: zarr
#
#         # extract master ome-tiff for each position using filename string matching --> this is not ideal
#         # ideally we can intercept the warnings from tifffile.imread, which states when we have master ome-tiff
#         for subfile_ in os.listdir(self.folder):
#             if subfile_.endswith('.ome.tif') and not (fnmatch.fnmatch(subfile_.split('_')[-1], '?.ome.tif') or
#                                                       (fnmatch.fnmatch(subfile_.split('_')[-1], '??.ome.tif'))):
#             # if subfile_.endswith('.ome.tif') and not fnmatch.fnmatch(subfile_, glob.glob('*[0-9]_.ome.tif')):
#             # if fnmatch.fnmatch(subfile_, '*Pos[0-9]*_[0-9]*.ome.tif') or fnmatch.fnmatch(subfile_, '*Default.ome.tif'):
#                 self.positions_filenames.add(subfile_)
#         print(f"folder contains {len(self.positions_filenames)} ome-tiffs, one for each scene")
#
#         # iterate per position_filenames, read mm-meta, ome-meta, extract zarr, assign to dict
#         #   confirm ome-meta filename matches position filename
#         for pos in self.positions_filenames:
#
#             mm_meta, tf_zarr, tiff_file = self._parse_meta(os.path.join(self.folder, pos))
#
#             # if mm_meta['Summary']['Positions'] != len(self.positions_filenames):
#             #     raise FileNotFoundError(f"micro-manager states {mm_meta['Summary']['Positions']} "
#             #                             f"positions but found {len(self.positions_filenames)} master ome-tiffs")
#
#             # search stage positions for proper position index
#             if mm_meta['Summary']['Positions'] > 1:
#                 # when multiple position and prefix is "Pos###_###"
#                 for idx, stage_position in enumerate(mm_meta['Summary']['StagePositions']):
#                     if stage_position["Label"] in pos:
#                         self.positions[idx] = tf_zarr
#             else:
#                 # when single position and the prefix is "Default"
#                 self.positions[0] = tf_zarr
#
#     def _parse_meta(self, subfile):
#
#         with self.reader.TiffFile(subfile) as tif:
#             print(subfile)
#
#             _mm_meta = tif.micromanager_metadata
#             # _ome_meta = minidom.parseString(tif.ome_metadata)
#             tfz = tif.aszarr()
#             z = zarr.open(tfz, mode='r')
#
#             return _mm_meta, z, tif
#
#     def get_zarr(self, pos_index):
#         return self.positions[pos_index]
#
#     def get_array(self, pos_index):
#         return np.array(self.positions[pos_index])