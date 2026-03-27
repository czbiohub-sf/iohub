"""Core zarr implementation abstraction for iohub."""

from iohub.core.arrays import NGFFNDArray
from iohub.core.compat import ngff_version_for_format, zarr_format_for_version
from iohub.core.config import (
    CompressorConfig,
    ImplementationConfig,
    TensorStoreConfig,
    ZarrConfig,
)
from iohub.core.errors import (
    ArraySpecError,
    ImplementationNotFoundError,
    IohubError,
    PathNormalizationError,
    StoreOpenError,
)
from iohub.core.protocol import (
    ArrayBackend,
    ArrayIO,
    GroupBackend,
    ZarrImplementation,
)
from iohub.core.registry import (
    available_implementations,
    get_implementation,
    register_implementation,
    set_default_implementation,
)
from iohub.core.specs import ArraySpec
from iohub.core.types import (
    AccessMode,
    NGFFVersion,
    StorePath,
    ZarrFormat,
)
from iohub.core.utils import normalize_path, pad_shape

__all__ = [
    # Arrays
    "NGFFNDArray",
    # Protocol facets
    "GroupBackend",
    "ArrayBackend",
    "ArrayIO",
    "ZarrImplementation",
    # Config
    "CompressorConfig",
    "ImplementationConfig",
    "TensorStoreConfig",
    "ZarrConfig",
    # Types
    "AccessMode",
    "NGFFVersion",
    "StorePath",
    "ZarrFormat",
    # Errors
    "ArraySpecError",
    "ImplementationNotFoundError",
    "IohubError",
    "PathNormalizationError",
    "StoreOpenError",
    # Registry
    "available_implementations",
    "get_implementation",
    "register_implementation",
    "set_default_implementation",
    # Specs
    "ArraySpec",
    # Compat
    "zarr_format_for_version",
    "ngff_version_for_format",
    # Utils
    "normalize_path",
    "pad_shape",
]
