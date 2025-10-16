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


def _configure_logging():
    """Configure logging for iohub."""
    level = os.environ.get("IOHUB_LOG_LEVEL", logging.INFO)
    if str(level).isdigit():
        level = int(level)
    iohub_logger = logging.getLogger(__name__)
    iohub_logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(fmt="%(levelname)s:%(name)s:%(message)s")
    )
    iohub_logger.addHandler(stream_handler)


_configure_logging()
