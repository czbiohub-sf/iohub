import glob
import logging
import os
import psutil
import textwrap
import tifffile as tiff
import numpy as np
from waveorder.waveorder_reconstructor import fluorescence_microscopy, waveorder_microscopy


def extract_reconstruction_parameters(reconstructor, magnification=None):
    """
    Function that extracts the reconstruction parameters from a waveorder reconstructor.  Works
    for both fluorescence_microscopy class and waveorder_microscopy class.

    Parameters
    ----------
    reconstructor:      (waveorder reconstructor object) initalized reconstructor
    magnification:      (float or None) magnification of the microscope setup (value not saved in reconstructor)

    Returns
    -------
    attr_dict           (dict) dictionary of reconstruction parameters in their native units

    """

    ps = reconstructor.ps
    if ps:
        ps = ps * magnification if magnification else ps

    if isinstance(reconstructor, waveorder_microscopy):
        attr_dict = {'phase_dimension': reconstructor.phase_deconv,
                     'wavelength (nm)': np.round(reconstructor.lambda_illu * 1000 * reconstructor.n_media,1),
                     'pad_z': reconstructor.pad_z,
                     'n_objective_media': reconstructor.n_media,
                     'bg_correction_option': reconstructor.bg_option,
                     'objective_NA': reconstructor.NA_obj * reconstructor.n_media,
                     'condenser_NA': reconstructor.NA_illu * reconstructor.n_media,
                     'magnification': magnification,
                     'swing': reconstructor.chi if reconstructor.N_channel == 4 else reconstructor.chi / 2 / np.pi,
                     'pixel_size': ps}

    elif isinstance(reconstructor, fluorescence_microscopy):
        attr_dict = {'fluor_wavelength (nm)': list(reconstructor.lambda_emiss * reconstructor.n_media * 1000),
                     'pad_z': reconstructor.pad_z,
                     'n_objective_media': reconstructor.n_media,
                     'objective_NA': reconstructor.NA_obj * reconstructor.n_media,
                     'magnification': magnification,
                     'pixel_size': ps}

    else:
        attr_dict = dict()

    return attr_dict


def load_bg(bg_path, height, width, ROI=None):
    """
    Parameters
    ----------
    bg_path         : (str) path to the folder containing background images
    height          : (int) height of image in pixels
    width           : (int) width of image in pixels
    ROI             : (tuple)  ROI of the background images to use, if None, full FOV will be used

    Returns
    -------
    bg_data   : (ndarray) Array of background data w/ dimensions (N_channel, Y, X)
    """

    bg_paths = glob.glob(os.path.join(bg_path, '*.tif'))
    bg_paths.sort()
    bg_data = np.zeros([len(bg_paths), height, width])

    for i in range(len(bg_paths)):
        img = tiff.imread(bg_paths[i])

        if ROI is not None and ROI != (0, 0, width, height): # TODO: Remove for 1.0.0
            warning_msg = """
            Warning: earlier versions of recOrder would have averaged over the background ROI. This behavior is 
            now considered a bug, and future versions of recOrder will not average over the background. 
            """
            logging.warning(warning_msg)
        else:
            bg_data[i, :, :] = img

    return bg_data


def create_grid_from_coordinates(xy_coords, rows, columns):
    """
    Function to create a grid from XY-position coordinates.  Useful for generating HCS Zarr metadata.

    Parameters
    ----------
    xy_coords:          (list) XY Stage position list in the order in which it was acquired: (X, Y) tuple.
    rows:               (int) number of rows in the grid-like acquisition
    columns:            (int) number of columns in the grid-like acquisition

    Returns
    -------
    pos_index_grid      (array) A grid-like array mimicking the shape of the acquisition where the value in the array
                                corresponds to the position index at that location.
    """

    coords = dict()
    coords_list = []
    for idx, pos in enumerate(xy_coords):
        coords[idx] = pos
        coords_list.append(pos)

    # sort by X and then by Y
    coords_list.sort(key=lambda x: x[0])
    coords_list.sort(key=lambda x: x[1])

    # reshape XY coordinates into their proper 2D shape
    grid = np.reshape(coords_list, (rows, columns, 2))
    pos_index_grid = np.zeros((rows,columns), 'uint16')
    keys = list(coords.keys())
    vals = list(coords.values())

    for row in range(rows):
        for col in range(columns):

            # append position index (key) into a final grid by indexed into the coordinate map (values)
            pos_index_grid[row, col] = keys[vals.index(list(grid[row, col]))]

    return pos_index_grid

class MockEmitter:
    def emit(self, value):
        pass

def get_unimodal_threshold(input_image):
    """Determines optimal unimodal threshold
    https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
    https://www.mathworks.com/matlabcentral/fileexchange/45443-rosin-thresholding
    :param np.array input_image: generate mask for this image
    :return float best_threshold: optimal lower threshold for the foreground
     hist
    """

    hist_counts, bin_edges = np.histogram(
        input_image,
        bins=256,
        range=(input_image.min(), np.percentile(input_image, 99.5))
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # assuming that background has the max count
    max_idx = np.argmax(hist_counts)
    int_with_max_count = bin_centers[max_idx]
    p1 = [int_with_max_count, hist_counts[max_idx]]

    # find last non-empty bin
    pos_counts_idx = np.where(hist_counts > 0)[0]
    last_binedge = pos_counts_idx[-1]
    p2 = [bin_centers[last_binedge], hist_counts[last_binedge]]

    best_threshold = -np.inf
    max_dist = -np.inf
    for idx in range(max_idx, last_binedge, 1):
        x0 = bin_centers[idx]
        y0 = hist_counts[idx]
        a = [p1[0] - p2[0], p1[1] - p2[1]]
        b = [x0 - p2[0], y0 - p2[1]]
        cross_ab = a[0] * b[1] - b[0] * a[1]
        per_dist = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if per_dist > max_dist:
            best_threshold = x0
            max_dist = per_dist
    assert best_threshold > -np.inf, 'Error in unimodal thresholding'
    return best_threshold

def ram_message():
    BYTES_PER_GB = 2**30
    gb_available = psutil.virtual_memory().total / BYTES_PER_GB
    if gb_available < 32:
        return ' \n'.join(textwrap.wrap(
               f'\033[91mWARNING:\033[0m recOrder reconstructions often require more than the {gb_available:.1f} ' \
               f'GB of RAM that this computer is equipped with. We recommend starting with reconstructions of small ' \
               f'volumes ~1000 x 1000 x 10 and working up to larger volumes while monitoring your RAM usage with '
               f'Task Manager or htop.',
        ))
    else:
        return f'{gb_available:.1f} GB of RAM is available.'

def rec_bkg_to_wo_bkg(recorder_option) -> str:
    """
    Converts recOrder's background options to waveorder's background options.

    Parameters
    ----------
    recorder_option

    Returns
    -------
    waveorder_option

    """
    if recorder_option == 'local_fit+':
        return 'local_fit'
    else:
        return recorder_option