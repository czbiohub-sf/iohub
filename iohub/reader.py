
from waveorder.io.singlepagetiff import MicromanagerSequenceReader
from waveorder.io.multipagetiff import MicromanagerOmeTiffReader
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
        reads data output from micro-manager and returns a zarr array or numpy array
        supports singlepage tiff sequences or ome-tiffs

        Parameters
        ----------
        src:            (str) folder or file containing all ome-tiff files
        data_type:      (str) whether data is 'ometiff', 'singlepagetiff', 'zarr'
        extract_data:   (bool) True if ome_series should be extracted immediately
        log_level:      (int) One of 0, 10, 20, 30, 40, 50 for NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL respectively
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

        Parameters
        ----------
        position:   (int) position (aka ome-tiff scene)

        Returns
        -------
        zarr.array for the provided position
        """
        return self.reader.get_zarr(position)

    def get_array(self, position):
        """
        return a numpy array for a given position

        Parameters
        ----------
        position:   (int) position (aka ome-tiff scene)

        Returns
        -------
        np.array for the provided position
        """
        return self.reader.get_array(position)

    def get_num_positions(self):
        """
        get total number of scenes referenced in ome-tiff metadata

        Returns:
        -------
        int of number of positions
        """
        return self.reader.get_num_positions()

    @property
    def shape(self):
        """
        return the underlying data shape as a tuple

        Returns:
        -------
        tuple of (frames, slices, channels, height, width)

        """
        return self.frames, self.slices, self.channels, self.height, self.width


def main():
    no_positions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_files_tpzc_200tp_1p_5z_3c_2k_1'
    multipositions = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_stack_tpzc_50tp_4p_5z_3c_2k_1'

    master_new_folder = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/test_1/'
    non_master_new_folder = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/test_1/'
    non_master_new_large_folder = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/image_stack_tpzc_50tp_4p_5z_3c_2k_1/'
    # non_master_old_large_folder = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_50tp_4p_5z_3c_2k_1/'

    master_old_folder = '/Volumes/comp_micro/rawdata/hummingbird/Janie/2021_02_03_40x_04NA_A549/48hr_RSV_IFN/Coverslip_1/C1_MultiChan_Stack_1/'
    non_master_old_folder = '/Volumes/comp_micro/rawdata/hummingbird/Janie/2021_02_03_40x_04NA_A549/48hr_RSV_IFN/Coverslip_1/C1_MultiChan_Stack_1/'

    ivan_dataset = '/Volumes/comp_micro/rawdata/falcon/Ivan/20210128 HEK CAAX SiRActin/FOV1_1'
    ivan_file = 'FOV1_1_MMStack_Default_23.ome.tif'

    # singlepage tiffs (1422)
    mm1_single = '/Users/bryant.chhun/Desktop/mm2_sampledata/packaged for gdd/mm1422_kazansky_one_position/mm1422_kazansky_one_position'
    mm1_multi_snake = '/Users/bryant.chhun/Desktop/mm2_sampledata/packaged for gdd/mm1422_kazansky_HCS_snake/mm1422_kazansky_HCS_snake'
    mm1_multi_grid = '/Users/bryant.chhun/Desktop/mm2_sampledata/packaged for gdd/mm1422_kazansky_grid/mm1422_kazansky_grid'

    # ome tiffs (1422)
    mm1_multi_pos = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm1422/ome-tiffs/mm1422_4p_2t_5z_3c_512_1'

    # ome tiffs (2.0)
    # with position
    mm2_p_t_z_c = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_4p_2t_5z_1c_2k_1'
    mm2_p_t_z = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_4p_2t_5z_2k_1'
    mm2_p_t_c = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_4p_2t_3c_2k_1'
    mm2_p_z_c = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_4p_5z_3c_2k_1'
    mm2_p_t = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_4p_2t_2k_1'
    mm2_p_c = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_4p_3c_2k_1'
    mm2_p_z = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_4p_5z_2k_1'

    # without position
    mm2_t_z_c = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_1t_5z_3c_2k_1'
    mm2_t_z = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_1t_5z_2k_1'
    mm2_t_c = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_50tp_2c_2k_1'
    mm2_z_c = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_5z_3c_2k_1'

    # without three
    mm2_p = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_4p_2k_1'
    mm2_t = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_1t_2k_1'
    mm2_c = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_1c_2k_1'
    mm2_z = '/Users/bryant.chhun/Desktop/Data/reconstruct-order-2/mm2.0_20201113_5z_2k_1'

    r = MicromanagerReader(multipositions,
                           # data_type='singlepagetiff',
                           data_type='ometiff',
                           extract_data=True)
    print(r.get_zarr(0))
    # print(r.get_zarr(1))
    # print(r.get_master_ome())
    print(r.get_num_positions())


if __name__ == "__main__":
    main()
