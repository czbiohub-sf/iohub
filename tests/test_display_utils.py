#%%
import numpy as np
from iohub.display_utils import rgb_to_hex, channel_display_settings


def test_rbg_to_hex():
    """Test conversion from RGB to HEX"""
    red = (255, 0, 0)
    lime = (50, 205, 50)
    blue = (0, 0, 255)
    red_hex = rgb_to_hex(*red)
    lime_hex = rgb_to_hex(*lime)
    blue_hex = rgb_to_hex(*blue)
    assert red_hex == "FF0000"
    assert lime_hex == "32CD32"
    assert blue_hex == "0000FF"


def test_channel_display_settings():
    """Test if channel name exists in dictonary, then assigns a color"""
    channel_name = "GFP"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "32CD32"
    channel_name = "S0"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "FFFFFF"
    channel_name = "TXR"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "FF00FF"
    channel_name = "RANDOM_CHANNEL"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "FFFFFF"
