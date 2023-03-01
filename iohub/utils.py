import glob
import logging
import os
import psutil
import textwrap
import tifffile as tiff
import numpy as np
from colorspacious import cspace_convert
from matplotlib.colors import hsv_to_rgb
from waveorder.waveorder_reconstructor import waveorder_microscopy


def extract_reconstruction_parameters(reconstructor, magnification=None):
    """
    Function that extracts the reconstruction parameters from a waveorder reconstructor.  Works for waveorder_microscopy class.

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
        attr_dict = {
            "phase_dimension": reconstructor.phase_deconv,
            "wavelength (nm)": np.round(
                reconstructor.lambda_illu * 1000 * reconstructor.n_media, 1
            ),
            "pad_z": reconstructor.pad_z,
            "n_objective_media": reconstructor.n_media,
            "bg_correction_option": reconstructor.bg_option,
            "objective_NA": reconstructor.NA_obj * reconstructor.n_media,
            "condenser_NA": reconstructor.NA_illu * reconstructor.n_media,
            "magnification": magnification,
            "swing": reconstructor.chi
            if reconstructor.N_channel == 4
            else reconstructor.chi / 2 / np.pi,
            "pixel_size": ps,
        }

    else:
        attr_dict = dict()

    return attr_dict


def load_bg(bg_path, height, width, ROI=None):
    """
    Parameters
    ----------
    bg_path         : (str) path to the folder containing background images
    height          : (int) height of image in pixels # Remove for 1.0.0
    width           : (int) width of image in pixels # Remove for 1.0.0
    ROI             : (tuple)  ROI of the background images to use, if None, full FOV will be used # Remove for 1.0.0

    Returns
    -------
    bg_data   : (ndarray) Array of background data w/ dimensions (N_channel, Y, X)
    """

    bg_paths = glob.glob(os.path.join(bg_path, "*.tif"))
    bg_paths.sort()

    # Backwards compatibility warning
    if ROI is not None and ROI != (
        0,
        0,
        width,
        height,
    ):  # TODO: Remove for 1.0.0
        warning_msg = """
        Earlier versions of recOrder (0.1.2 and earlier) would have averaged over the background ROI. 
        This behavior is now considered a bug, and future versions of recOrder (0.2.0 and later) 
        will not average over the background. 
        """
        logging.warning(warning_msg)

    # Load background images
    bg_img_list = []
    for bg_path in bg_paths:
        bg_img_list.append(tiff.imread(bg_path))
    bg_img_arr = np.array(bg_img_list)  # CYX

    # Error if shapes do not match
    # TODO: 1.0.0 move these validation check to waveorder's Polscope_bg_correction
    if bg_img_arr.shape[1:] != (height, width):
        error_msg = "The background image has a different X/Y size than the acquired image."
        raise ValueError(error_msg)

    return bg_img_arr  # CYX


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
    pos_index_grid = np.zeros((rows, columns), "uint16")
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
        range=(input_image.min(), np.percentile(input_image, 99.5)),
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
    assert best_threshold > -np.inf, "Error in unimodal thresholding"
    return best_threshold


def ram_message():
    """
    Determine if the system's RAM capacity is sufficient for running reconstruction.
    The message should be treated as a warning if the RAM detected is less than 32 GB.

    Returns
    -------
    ram_report    (is_warning, message)
    """
    BYTES_PER_GB = 2**30
    gb_available = psutil.virtual_memory().total / BYTES_PER_GB
    is_warning = gb_available < 32

    if is_warning:
        message = " \n".join(
            textwrap.wrap(
                f"recOrder reconstructions often require more than the {gb_available:.1f} "
                f"GB of RAM that this computer is equipped with. We recommend starting with reconstructions of small "
                f"volumes ~1000 x 1000 x 10 and working up to larger volumes while monitoring your RAM usage with "
                f"Task Manager or htop.",
            )
        )
    else:
        message = f"{gb_available:.1f} GB of RAM is available."

    return (is_warning, message)


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
    if recorder_option == "local_fit+":
        return "local_fit"
    else:
        return recorder_option


def generic_hsv_overlay(
    H, S, V, H_scale=None, S_scale=None, V_scale=None, mode="2D"
):
    """
    Generates a generic HSV overlay in either 2D or 3D

    Parameters
    ----------
    H:          (nd-array) data to use in the Hue channel
    S:          (nd-array) data to use in the Saturation channel
    V:          (nd-array) data to use in the Value channel
    H_scale:    (tuple) values at which to clip the hue data for display
    S_scale:    (tuple) values at which to clip the saturation data for display
    V_scale:    (tuple) values at which to clip the value data for display
    mode:       (str) '3D' or '2D'

    Returns
    -------
    overlay:    (nd-array) RGB overlay array of shape (Z, Y, X, 3) or (Y, X, 3)

    """

    if H.shape != S.shape or H.shape != S.shape or S.shape != V.shape:
        raise ValueError(
            f"Channel shapes do not match: {H.shape} vs. {S.shape} vs. {V.shape}"
        )

    if mode == "3D":
        overlay_final = np.zeros((H.shape[0], H.shape[1], H.shape[2], 3))
        slices = H.shape[0]
    else:
        overlay_final = np.zeros((1, H.shape[-2], H.shape[-1], 3))
        H = np.expand_dims(H, axis=0)
        S = np.expand_dims(S, axis=0)
        V = np.expand_dims(V, axis=0)
        slices = 1

    for i in range(slices):
        H_ = np.interp(H[i], H_scale, (0, 1))
        S_ = np.interp(S[i], S_scale, (0, 1))
        V_ = np.interp(V[i], V_scale, (0, 1))

        hsv = np.transpose(np.stack([H_, S_, V_]), (1, 2, 0))
        overlay_final[i] = hsv_to_rgb(hsv)

    return overlay_final[0] if mode == "2D" else overlay_final


def ret_ori_overlay(
    retardance, orientation, ret_max=10, mode="2D", cmap="JCh"
):
    """
    This function will create an overlay of retardance and orientation with two different colormap options.
    HSV is the standard Hue, Saturation, Value colormap while JCh is a similar colormap but is perceptually uniform.

    Parameters
    ----------
    retardance:             (nd-array) retardance array of shape (N, Y, X) or (Y, X) in nanometers
    orientation:            (nd-array) orientation array of shape (N, Y, X) or (Y, X) in radian [0, pi]
    ret_max:                (float) maximum displayed retardance.  Typically use adjusted contrast limits.
    mode:                   (str) '2D' or '3D'
    cmap:                   (str) 'JCh' or 'HSV'

    Returns
    -------
    overlay                 (nd-array) overlaid image of shape (N, Y, X, 3) or (Y, X, 3) RGB image

    """

    orientation = np.copy(orientation) * 180 / np.pi
    noise_level = 1

    if retardance.shape != orientation.shape:
        raise ValueError(
            f"Retardance and Orientation shapes do not match: {retardance.shape} vs. {orientation.shape}"
        )

    if mode == "3D":
        overlay_final = np.zeros(
            (retardance.shape[0], retardance.shape[1], retardance.shape[2], 3)
        )
        slices = retardance.shape[0]
    else:
        overlay_final = np.zeros(
            (1, retardance.shape[-2], retardance.shape[-1], 3)
        )
        orientation = np.expand_dims(orientation, axis=0)
        retardance = np.expand_dims(retardance, axis=0)
        slices = 1

    for i in range(slices):
        ret_ = np.copy(
            retardance[i]
        )  # copy to avoid modifying original array.
        ret_[ret_ > ret_max] = ret_max  # clip retardance to specified max.

        # FIX ME: this binning code leads to artifacts.
        # levels = 32
        # ori_binned = (
        #     np.round(orientation[i] / 180 * levels + 0.5) / levels - 1 / levels
        # ) # bin orientation into 32 levels.
        # ori_ = np.interp(ori_binned, (0, 1), (0, 360))

        ori_ = orientation[i] * (
            360.0 / 180.0
        )  # convert 180 degree range into 360 to match periodicity of hue.
        if cmap == "JCh":
            J = ret_
            C = np.ones_like(J) * 60
            C[retardance[i] < noise_level] = 0
            h = ori_

            JCh = np.stack((J, C, h), axis=-1)
            JCh_rgb = cspace_convert(JCh, "JCh", "sRGB1")

            JCh_rgb[JCh_rgb < 0] = 0
            JCh_rgb[JCh_rgb > 1] = 1

            overlay_final[i] = JCh_rgb
        elif cmap == "HSV":
            I_hsv = np.transpose(
                np.stack(
                    [
                        ori_ / 360,
                        np.ones_like(ori_),
                        np.minimum(1, ret_ / np.max(ret_)),
                    ]
                ),
                (1, 2, 0),
            )
            overlay_final[i] = hsv_to_rgb(I_hsv)

        else:
            raise ValueError(f"Colormap {cmap} not understood")

    return overlay_final[0] if mode == "2D" else overlay_final
