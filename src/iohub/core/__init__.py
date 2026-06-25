"""Core zarr implementation abstraction for iohub."""

import lazy_loader as lazy

# Lazy loading (SPEC 1): importing ``iohub.core`` (or ``iohub.core.types`` for a
# leaf like ``NGFFVersion``) no longer drags in ``arrays`` -> xarray. Exports are
# declared in ``__init__.pyi``.
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
