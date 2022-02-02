import glob
import os
import tifffile as tiff
import numpy as np

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

    bg_paths = glob.glob(os.path.join(bg_path,'*.tif'))
    bg_paths.sort()
    bg_data = np.zeros([len(bg_paths), height, width])

    for i in range(len(bg_paths)):
        img = tiff.imread(bg_paths[i])

        if ROI is not None and ROI != (0, 0, height, width):
            bg_data[i, :, :] = np.mean(img[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]])

        else:
            bg_data[i, :, :] = img

    return bg_data

def create_grid_from_coordinates(xy_coords, rows, columns):

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
