"""Pydantic config models for zarr implementations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CompressorConfig(BaseModel):
    """Default compressor settings for array creation."""

    cname: str = "zstd"
    clevel: int = 1
    shuffle: str = "bitshuffle"


# Dotted paths to the codec pipelines selectable via the implementation name.
PYTHON_CODEC_PIPELINE = "zarr.core.codec_pipeline.BatchedCodecPipeline"
ZARRS_CODEC_PIPELINE = "zarrs.ZarrsCodecPipeline"


class ZarrConfig(BaseModel):
    """Config for the zarr-python implementation.

    The codec pipeline defaults to the pure-Python pipeline. The selected
    implementation name (``zarr-python`` vs ``zarrs-python``) determines which
    pipeline is used when no explicit ``codec_pipeline`` is given; an explicit
    value here always takes precedence.
    """

    codec_pipeline: str = PYTHON_CODEC_PIPELINE
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


CZARR_CODEC_PIPELINE = "czarr.pipeline.CzarrPipeline"


class CzarrConfig(BaseModel):
    """Config for the czarr (GPU-accelerated) implementation.

    Benchmark-only backend (see SRI-3); not intended for iohub's main branch.

    By default this behaves like ``zarrs-python``: a drop-in codec-pipeline
    swap (``czarr.pipeline.CzarrPipeline``) plus per-codec registry overrides,
    with zarr's plain ``LocalStore`` and CPU buffer prototypes untouched.
    Reads/writes stay ``numpy.ndarray`` in/out and only the decode step runs
    on the GPU (nvCOMP) -- bytes are still read from disk on the CPU first.

    Parameters
    ----------
    batch_size : int or None
        ``codec_pipeline.batch_size``. ``None`` leaves zarr's own default
        (micro-batched); czarr's own ``configure_gpu()`` defaults this to
        "one whole-batch nvCOMP call" for perf -- set explicitly to match.
    async_concurrency : int
        ``async.concurrency`` -- concurrent decode-batch limit.
    rmm_pool_gb : float or None
        If set, installs a process-wide RMM memory pool on first use via
        ``czarr.use_rmm_pool()`` -- czarr does not revert this on teardown,
        regardless of ``gpu_buffers``.
    gpu_buffers : bool
        Opt-in to the real disk-to-GPU path (GPUDirect Storage / cuFile),
        for the SRI-9 disk-to-GPU benchmark scenario specifically. Swaps
        zarr's global buffer/ndbuffer prototypes to GPU and opens the store
        as ``czarr.GPULocalStore`` (its ``get()`` only takes the cuFile fast
        path when a GPU buffer is requested) -- verified on an H100 node:
        cuFile fires per shard and results are bit-identical to the CPU
        path. Reads then return ``cupy.ndarray`` internally, converted back
        to ``numpy.ndarray`` by this implementation to satisfy the Protocol.

        This mutates process-global zarr state more invasively than the
        default (any other backend's ``np.asarray(handle[selection])`` in
        the *same process* will raise ``TypeError`` on the resulting cupy
        arrays) -- do not interleave with other backends in one process;
        give the disk-to-GPU benchmark run its own process.
    """

    compressor: CompressorConfig = Field(default_factory=CompressorConfig)
    batch_size: int | None = None
    async_concurrency: int = 32
    rmm_pool_gb: float | None = None
    gpu_buffers: bool = False


ImplementationConfig = ZarrConfig | TensorStoreConfig | CzarrConfig
