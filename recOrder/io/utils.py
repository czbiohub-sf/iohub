import glob
import os
import tifffile as tiff
import numpy as np

def gather_sub_dir(self, dir_):
    files = glob.glob(dir_ + '*')

    dirs = []
    for direc in files:
        if os.path.isdir(direc):
            dirs.append(direc)

    return dirs

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
