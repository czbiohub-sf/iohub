import zarr
import os
import glob
import natsort
import tifffile as tiff
from waveorder.io.singlepagetiff import MicromanagerSequenceReader
from waveorder.io.multipagetiff import MicromanagerOmeTiffReader
from waveorder.io.upti import UPTIReader
from waveorder.io.zarrfile import ZarrReader
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

#todo: add dim_order to all reader objects
class WaveorderReader:

    def __init__(self,
                 src: str,
                 data_type: str = None,
                 extract_data: bool = False,
                 log_level: int = logging.ERROR):
        """
        reads data output from micro-manager and returns a zarr array or numpy array
        supports singlepage tiff sequences or ome-tiffs

        Parameters
        ----------
        src:            (str) folder or file containing all ome-tiff files or zarr root
        data_type:      (str) whether data is 'ometiff', 'singlepagetiff', or 'zarr'
        extract_data:   (bool) True if ome_series should be extracted immediately
        log_level:      (int) One of 0, 10, 20, 30, 40, 50 for NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL respectively
        """

        logging.basicConfig(
            level=log_level,
            format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
        )
        self.log = logging.getLogger(__name__)

        # catch if the user accidentally specifies one of the tiff files and not the parent folder
        if data_type == 'ometiff':
            if src.endswith('.tif'):
                src = os.path.dirname(src)

            if len(glob.glob(os.path.join(src, '*.tif'))) == 0:
                raise FileNotFoundError('Specified folder does not contain any ome.tif files')

        # Try to infer datatype
        if not data_type:
            try:
                zarr.open(src, 'r')
                data_type = 'zarr'
            except:
                if os.path.exists(src):
                    if src.endswith('.tif'):
                        src = os.path.dirname(src)

                    files = glob.glob(os.path.join(src, '*.tif'))
                    if len(files) == 0:

                        sub_dirs = self._get_sub_dirs(src)
                        if sub_dirs:
                            path = os.path.join(src, sub_dirs[0])
                            files = glob.glob(os.path.join(path, '*.tif'))
                            if len(files) == 0:
                                raise FileNotFoundError(f'No compatible data found under {src}')
                            else:
                                with tiff.TiffFile(os.path.join(path, files[0])) as tf:
                                    if len(tf.pages) == 1 and tf.pages[0].is_multipage.name == 'UNDEFINED':
                                        data_type = 'singlepagetiff'

                        else:
                            raise FileNotFoundError(f'No compatible data found under {src}, please specify the top '
                                                    'level micromanager directory')
                    else:
                        with tiff.TiffFile(files[0]) as tf:
                            if len(tf.pages) > 1:
                                data_type = 'ometiff'
                            else:
                                if tf.pages[0].is_multipage == 0 and tf.is_ome == True:
                                    data_type = 'ometiff'
                                else:
                                    raise FileNotFoundError(f'No compatible data found under {src}')
                else:
                    raise FileExistsError(f'{src} does not exist')

        # identify data structure type
        if data_type == 'ometiff':
            self.reader = MicromanagerOmeTiffReader(src, extract_data)
        elif data_type == 'singlepagetiff':
            self.reader = MicromanagerSequenceReader(src, extract_data)
        elif data_type == 'zarr':
            self.reader = ZarrReader(src)
        elif data_type == 'upti':
            self.reader = UPTIReader(src, extract_data)
        else:
            raise NotImplementedError(f"reader of type {data_type} is not implemented")

    def _get_sub_dirs(self, f):
        """
        subdir walk
        from https://github.com/mehta-lab/reconstruct-order

        Parameters
        ----------
        f:              (str)

        Returns
        -------
        sub_dir_name    (list) natsorted list of subdirectories
        """

        sub_dir_path = glob.glob(os.path.join(f, '*/'))
        sub_dir_name = [os.path.split(subdir[:-1])[1] for subdir in sub_dir_path]
        #    assert subDirName, 'No sub directories found'
        return natsort.natsorted(sub_dir_name)

    def get_zarr(self, position=0):
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

    def get_array(self, position=0):
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

    def get_image(self, p, t, c, z):
        return self.reader.get_image(p, t, c, z)

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
        return self.frames, self.channels, self.slices, self.height, self.width

    @property
    def mm_meta(self):
        return self.reader.mm_meta

    @mm_meta.setter
    def mm_meta(self, value):
        assert(type(value) is dict)
        self.reader.mm_meta = value

    @property
    def stage_positions(self):
        return self.reader.stage_positions

    @stage_positions.setter
    def stage_positions(self, value):
        assert(type(value) is list)
        self.reader.stage_positions = value

    @property
    def z_step_size(self):
        return self.reader.z_step_size

    @z_step_size.setter
    def z_step_size(self, value):
        self.reader.z_step_size = value

    @property
    def height(self):
        return self.reader.height

    @height.setter
    def height(self, value):
        self.reader.height = value

    @property
    def width(self):
        return self.reader.width

    @width.setter
    def width(self, value):
        self.reader.width = value

    @property
    def frames(self):
        return self.reader.frames

    @frames.setter
    def frames(self, value):
        self.reader.frames = value

    @property
    def slices(self):
        return self.reader.slices

    @slices.setter
    def slices(self, value):
        self.reader.slices = value

    @property
    def channels(self):
        return self.reader.channels

    @channels.setter
    def channels(self, value):
        self.reader.channels = value

    @property
    def channel_names(self):
        return self.reader.channel_names

    @channel_names.setter
    def channel_names(self, value):
        self.reader.channel_names = value
