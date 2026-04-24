"""Pydantic config models for zarr implementations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CompressorConfig(BaseModel):
    """Default compressor settings for array creation."""

    cname: str = "zstd"
    clevel: int = 1
    shuffle: str = "bitshuffle"


def _default_codec_pipeline() -> str:
    """Use zarrs (Rust) codec pipeline if available, else default Python."""
    try:
        import zarrs  # noqa: F401

        return "zarrs.ZarrsCodecPipeline"
    except ImportError:
        return "zarr.core.codec_pipeline.BatchedCodecPipeline"


class ZarrConfig(BaseModel):
    """Config for the zarr-python implementation."""

    codec_pipeline: str = Field(default_factory=_default_codec_pipeline)
    validate_checksums: bool = True
    compressor: CompressorConfig = Field(default_factory=CompressorConfig)


class TensorStoreConfig(BaseModel):
    """Config for the TensorStore implementation.

    Parameters
    ----------
    file_io_concurrency : int or None
        Concurrency limit for TensorStore's ``file_io_concurrency``
        resource. Raise above the default (32) on high-latency networked
        filesystems (e.g. NFS) where the default under-saturates the link.
    cache_pool_bytes : int or None
        Aggregate byte budget for TensorStore's chunk cache pool. ``None``
        disables caching.
    recheck_cached_data : bool, "open" or None
        Controls whether cached chunk data is re-validated on each read.
        ``None`` (default) uses the TensorStore driver default, which
        revalidates cached metadata on every access — one stat/GETATTR per
        chunk. ``"open"`` checks freshness only when the array is opened
        and trusts the cache thereafter — recommended for long-running
        read-heavy workloads on NFS/VAST where the underlying zarr files
        do not change. ``False`` disables freshness checks entirely.
    """

    compressor: CompressorConfig = Field(default_factory=CompressorConfig)
    data_copy_concurrency: int = Field(default=4, ge=1)
    context: dict | None = None
    file_io_concurrency: int | None = None
    file_io_sync: bool = True
    file_io_locking: Literal["auto", "disabled"] = "auto"
    cache_pool_bytes: int | None = None
    recheck_cached_data: bool | Literal["open"] | None = None
    extra_context: dict | None = None


ImplementationConfig = ZarrConfig | TensorStoreConfig
