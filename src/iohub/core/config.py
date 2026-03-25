"""Pydantic config models for zarr implementations."""

from __future__ import annotations

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
    """Config for the TensorStore implementation."""

    data_copy_concurrency: int = Field(default=4, ge=1)
    context: dict | None = None


ImplementationConfig = ZarrConfig | TensorStoreConfig
