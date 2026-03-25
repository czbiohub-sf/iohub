"""NGFF version <-> zarr format mapping and compatibility utilities."""

from __future__ import annotations

from iohub.core.types import NGFFVersion, ZarrFormat

NGFF_TO_ZARR_FORMAT: dict[str, int] = {"0.4": 2, "0.5": 3}
ZARR_FORMAT_TO_NGFF: dict[int, str] = {2: "0.4", 3: "0.5"}


def zarr_format_for_version(version: NGFFVersion) -> ZarrFormat:
    """Map NGFF version string to zarr format integer."""
    try:
        return NGFF_TO_ZARR_FORMAT[version]  # type: ignore[return-value]
    except KeyError:
        raise ValueError(f"Unknown NGFF version: {version!r}. Supported: {list(NGFF_TO_ZARR_FORMAT)}")


def ngff_version_for_format(zarr_format: ZarrFormat) -> NGFFVersion:
    """Map zarr format integer to NGFF version string."""
    try:
        return ZARR_FORMAT_TO_NGFF[zarr_format]  # type: ignore[return-value]
    except KeyError:
        raise ValueError(f"Unknown zarr format: {zarr_format!r}. Supported: {list(ZARR_FORMAT_TO_NGFF)}")
