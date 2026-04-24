"""Core zarr implementation abstraction for iohub."""

from iohub.core.arrays import NGFFArray
from iohub.core.compat import get_ome_attrs, ngff_version_for_format, zarr_format_for_version
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
from iohub.core.ozx import (
    OZX_EXTENSION,
    OzxStore,
    OzxSummary,
    is_ozx_path,
    pack_ozx,
    read_ozx_comment,
    read_ozx_json_first,
    read_ozx_version,
    summarize_ozx,
    write_ozx_comment,
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
    "OZX_EXTENSION",
    # Types
    "AccessMode",
    "ArrayBackend",
    "ArrayIO",
    # Specs
    "ArraySpec",
    # Errors
    "ArraySpecError",
    # Config
    "CompressorConfig",
    # Protocol facets
    "GroupBackend",
    "ImplementationConfig",
    "ImplementationNotFoundError",
    "IohubError",
    # Arrays
    "NGFFArray",
    "NGFFVersion",
    # OZX (RFC-9 zipped OME-Zarr)
    "OzxStore",
    "OzxSummary",
    "PathNormalizationError",
    "StoreOpenError",
    "StorePath",
    "TensorStoreConfig",
    "ZarrConfig",
    "ZarrFormat",
    "ZarrImplementation",
    # Registry
    "available_implementations",
    "get_implementation",
    "is_ozx_path",
    "ngff_version_for_format",
    # Utils
    "normalize_path",
    "pack_ozx",
    "pad_shape",
    "read_ozx_comment",
    "read_ozx_json_first",
    "read_ozx_version",
    "register_implementation",
    "set_default_implementation",
    "summarize_ozx",
    "write_ozx_comment",
    # Compat
    "zarr_format_for_version",
]
