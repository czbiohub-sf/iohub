"""
iohub

=====

N-dimensional bioimaging data I/O with OME metadata in Python
"""

import logging
import os
from importlib.metadata import version

import lazy_loader as lazy

# Load exports from __init__.pyi on first access so CLI startup does not import
# xarray, pandas, and dask.
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

__version__ = version(__name__)


def _configure_logging():
    """Configure logging for iohub."""
    level = os.environ.get("IOHUB_LOG_LEVEL", logging.INFO)
    if str(level).isdigit():
        level = int(level)
    iohub_logger = logging.getLogger(__name__)
    iohub_logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt="%(levelname)s:%(name)s:%(message)s"))
    iohub_logger.addHandler(stream_handler)


_configure_logging()
