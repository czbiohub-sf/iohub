"""Core zarr implementation abstraction for iohub."""

import lazy_loader as lazy

# Load exports from __init__.pyi on first access. Importing a core submodule
# should not also import arrays and xarray.
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
