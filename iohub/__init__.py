"""
iohub
=====

N-dimensional bioimaging data I/O with OME metadata in Python
"""

import logging
import os

from iohub.ngff import open_ome_zarr
from iohub.reader import read_images

__all__ = ["open_ome_zarr", "read_images"]


_level = os.environ.get("IOHUB_LOG_LEVEL", logging.INFO)
if str(_level).isdigit():
    _level = int(_level)

logging.basicConfig()
logging.getLogger(__name__).setLevel(_level)
