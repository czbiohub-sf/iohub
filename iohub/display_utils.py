""" Utility functions for displaying data """

import numpy as np
from PIL.ImageColor import colormap

from iohub.ngff_meta import ChannelMeta, WindowDict

""" Dictionary with key works and most popular fluorescent probes """
CHANNEL_COLORS = {
    # emission around 510-525 nm
    "lime": [
        "Green",
        "GFP",
        "FITC",
        "mNeon",
    ],
    # emission around 580 - 610 nm
    "magenta": [
        "Magenta",
        "TXR",
        "RFP",
        "mScarlet",
        "mCherry",
    ],
    # emission around 440 - 460 nmm
    "blue": ["Blue", "DAPI", "BFP", "Hoechst"],
    "red": ["Red"],
    "yellow": ["Yellow", "Cy3"],  # Emission around 540-570 nm
    "orange": ["Orange", "Cy5"],  # emission around 650-680 nm
}


def color_to_hex(color: str) -> str:
    """
    Convert the color string to HEX (i.e 'red' -> 'FF0000')
    (https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageColor.html#getrgb)
    (https://www.w3.org/TR/css-color-3/#svg-color)

    Parameters
    ----------
    color : str The name of the color from the CSS3 Specifications

    Returns
    -------
    str the HEX value in uppercase and without the '#'
    """
    return colormap[color][1:].upper()


def channel_display_settings(
    chan_name: str,
    clim: tuple[float, float, float, float] = None,
    first_chan: bool = False,
):
    """This will create a dictionary used for OME-zarr metadata.
    Allows custom contrast limits and channel.
    names for display. Defaults everything to grayscale.

    Parameters
    ----------
    chan_name : str
        Desired name of the channel for display
    clim : tuple[float, float, float, float], optional
        Contrast limits (start, end, min, max)
    first_chan : bool, optional
        Whether or not this is the first channel of the dataset
        (display will be set to active),
        by default False

    Returns
    -------
    dict
        Display settings adherent to ome-zarr standards
    """
    U16_FMAX = float(np.iinfo(np.uint16).max)
    channel_settings = {
        "Retardance": (0.0, 100.0, 0.0, 1000.0),
        "Orientation": (0.0, np.pi, 0.0, np.pi),
        "Phase2D": (-0.2, 0.2, -10, 10),
        "Phase3D": (-0.2, 0.2, -10, 10),
        "BF": (0.0, 5.0, 0.0, U16_FMAX),
        "S0": (0.0, 1.0, 0.0, U16_FMAX),
        "S1": (-0.5, 0.5, -10.0, 10.0),
        "S2": (-0.5, 0.5, -10.0, 10.0),
        "S3": (-1.0, 1.0, -10.0, -10.0),
        "Other": (0, U16_FMAX, 0.0, U16_FMAX),
    }
    if not clim:
        if chan_name in channel_settings.keys():
            clim = channel_settings[chan_name]
        else:
            clim = channel_settings["Other"]
    # Mapping channel name to color
    for key in CHANNEL_COLORS:
        # Does chan_name have any of the values in the CHANNEL_COLORS[key]
        # list as a substring?
        if any([value in chan_name for value in CHANNEL_COLORS[key]]):
            display_color = color_to_hex(key)
            break
        else:
            display_color = color_to_hex("white")

    window = WindowDict(start=clim[0], end=clim[1], min=clim[2], max=clim[3])
    return ChannelMeta(
        active=first_chan,
        coefficient=1.0,
        color=display_color,
        family="linear",
        inverted=False,
        label=chan_name,
        window=window,
    )
