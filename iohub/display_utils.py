# Utility functions for displaying data

from typing import Tuple

import numpy as np

from iohub.ngff_meta import ChannelMeta, WindowDict

def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert rgb values to hex
    Example: Red (255,0,0) -> 'FF0000'
    Parameters
    ----------
    r : int
        red
    g : int
        green
    b : int
        blue
    Returns
    -------
    str: Hex string corresponding to the RGB value
    """
    hex_color = "{:02X}{:02X}{:02X}".format(r, g, b)
    return hex_color

# Dictionary with key works and most popular fluorescent probes
channel_colors = {
    "green": ["GFP", "Green", "Alexa488", "GCaMP","FITC", "mNeon"],
    "magenta": ["TXR", "RFP", "mScarlet", "mCherry", "dTomato", "Cy5","Alexa561"],
    "blue": ["DAPI", "Blue", "BFP"],
    "red": ["Red"],
    "orange": ["Orange", "Cy3"],
    "yellow":["Alexa561"],
    "white": [
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
        "BF",
        "Phase2D",
        "Phase3D",
        "Retardance",
        "Orientation",
    ],
}

popular_colors = {
    "red": rgb_to_hex(255, 0, 0),
    "green": rgb_to_hex(0, 255, 0),
    "blue": rgb_to_hex(0, 0, 255),
    "yellow": rgb_to_hex(255, 255, 0),
    "orange": rgb_to_hex(255, 165, 0),
    "purple": rgb_to_hex(128, 0, 128),
    "pink": rgb_to_hex(255, 192, 203),
    "gray": rgb_to_hex(128, 128, 128),
    "brown": rgb_to_hex(165, 42, 42),
    "white": rgb_to_hex(255, 255, 255),
    "black": rgb_to_hex(0, 0, 0),
    "cyan": rgb_to_hex(0, 255, 255),
    "magenta": rgb_to_hex(255, 0, 255),
    "crimson": rgb_to_hex(220, 20, 60),
    "darkorange": rgb_to_hex(255, 140, 0),
    "tomato": rgb_to_hex(255, 99, 71),
    "turquoise": rgb_to_hex(64, 224, 208),
    "lime":rgb_to_hex(50,205,50),
}

def channel_display_settings(
    chan_name: str,
    clim: Tuple[float, float, float, float] = None,
    first_chan: bool = False,
):
    """This will create a dictionary used for OME-zarr metadata.
    Allows custom contrast limits and channel.
    names for display. Defaults everything to grayscale.

    Parameters
    ----------
    chan_name : str
        Desired name of the channel for display
    clim : Tuple[float, float, float, float], optional
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
            display_color = "FFFFFF"
            
    #Mapping channel name to color
    for key in channel_colors:
        if chan_name in channel_colors[key]:
            display_color = popular_colors[key]
            break
        else:
            display_color = display_color = "FFFFFF"

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






