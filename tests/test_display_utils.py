from iohub.ngff.display import channel_display_settings


def test_channel_display_settings():
    """Test if channel name exists in dictonary, then assigns a color"""
    channel_name = "GFP"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "00FF00"
    channel_name = "GFP EX488 EM525-45"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "00FF00"
    channel_name = "S0"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "FFFFFF"
    channel_name = "TXR"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "FF00FF"
    channel_name = "RANDOM_CHANNEL"
    channel_meta = channel_display_settings(channel_name)
    assert channel_meta.color == "FFFFFF"
