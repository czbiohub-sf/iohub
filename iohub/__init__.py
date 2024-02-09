#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""iohub
N-dimensional bioimaging data I/O with OME metadata in Python
"""


import logging

from iohub.ngff import open_ome_zarr
from iohub.reader import read_images

__all__ = ["open_ome_zarr", "read_images"]


logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.INFO)
