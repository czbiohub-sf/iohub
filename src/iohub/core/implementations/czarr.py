"""Czarr implementation -- zarr-python groups + a czarr GPU codec pipeline.

czarr (https://github.com/srivarra/czarr) decodes/encodes Zarr v3 chunks on
NVIDIA GPUs via nvCOMP, with cuFile/GPUDirect Storage for disk I/O where
supported. It requires Linux, a supported NVIDIA GPU, and cupy -- importing
the package itself pulls in cupy unconditionally (a candidate upstream fix,
see SRI-3 discussion), so the whole module is guarded and degrades to
"unavailable" rather than breaking iohub's own import on non-GPU hosts.

Design: czarr ships ``czarr.pipeline.CzarrPipeline``, a
``BatchedCodecPipeline`` subclass that is a drop-in swap for zarr's default
codec pipeline -- structurally identical to how ``zarrs.ZarrsCodecPipeline``
plugs into :class:`ZarrsPythonImplementation`. This implementation uses
exactly that (``codec_pipeline.path`` + per-codec registry overrides) by
default, so every read/write is a plain ``zarr.Array`` operation with plain
``numpy.ndarray`` in/out -- no array-wrapping class, no tier-1/tier-2
fallback duality to reason about, and no buffer-prototype mutation. That
last point matters: ``czarr.configure_gpu()`` (the library's own one-call
setup) also swaps zarr's global buffer/ndbuffer prototypes to GPU, which
makes *any other backend's* ``np.asarray(handle[selection])`` raise
``TypeError`` on a stray ``cupy.ndarray`` for the rest of the process's
lifetime (see SRI-3 discussion -- hit this directly in manual testing).
Skipping that swap by default avoids it entirely for ordinary reads.

Caveat verified on an H100 node: with the default config, only *decode*
runs on the GPU -- bytes are still read from disk into host memory by
zarr's plain ``LocalStore`` first (0 cuFile calls observed). That's enough
to benchmark nvCOMP decode throughput, but it is not the "skip the host
bounce" GPUDirect Storage story that's the actual point of prototyping
czarr for SRI-9. Real cuFile disk reads require ``CzarrConfig(gpu_buffers=
True)`` (see that field's docstring) -- confirmed this fires 26 cuFile
calls for a whole-timepoint read of the SRI-3 pick dataset and returns
bit-identical data, ~6x faster warm than the decode-only path in that one
measurement (not a real benchmark -- that's SRI-4).

Even the default (non-``gpu_buffers``) mode is not fully isolation-safe for
multi-backend benchmarking: the per-codec registry overrides
(``zarr.config["codecs"]["zstd"]``, etc.) are still process-global, so a
plain ``zarr-python``/``zarrs-python`` read *after* a ``CzarrImplementation``
has been constructed will silently pick up GPU decode for those codec IDs
too -- no crash, just a quietly-contaminated CPU baseline. The robust fix
either way is the same one flagged in the SRI-3 discussion: give each
backend its own process in the SRI-4 harness rather than relying on
construction order within one process.

Two more constraints verified on real writes, both load-bearing for how the
benchmark suite (SRI-2/SRI-7) must generate data:

* **czarr's ``Blosc`` codec is decode-only** -- ``encode()`` raises
  ``NotImplementedError``. iohub's own default compressor (``ArraySpec.
  create()``'s default, and what the real Mantis v2 data uses) is exactly
  this codec (``blosc``/zstd/bitshuffle). Reading existing blosc-compressed
  data works fine (verified against real data); *writing* new data through
  ``CzarrImplementation`` with the default codec raises. Any scenario that
  needs czarr to write (a roundtrip scenario, or generating synthetic test
  fixtures via czarr itself) must use a codec czarr can encode -- verified
  working: a raw ``zstd`` codec (not blosc-wrapped). This is also what
  czarr's own error message and README recommend for new writes.
* **``gpu_buffers=True`` is read-only-safe, not write-safe.** Opening for
  write (``mode="w"``) or read-write (``mode="r+"``) with ``gpu_buffers=
  True`` raises a native ``cuFileError: HANDLE_NOT_REGISTERED`` deep inside
  czarr's ``GPULocalStore`` -- reproduced for both a fresh group's metadata
  write and a metadata read under ``mode="r+"``. Only ``mode="r"`` against
  an already-written store is confirmed reliable. ``open_group`` now raises
  a clear ``ValueError`` for any other mode when ``gpu_buffers=True``,
  rather than letting that cuFile traceback surface. Practical effect: the
  SRI-9 disk-to-GPU benchmark (read-only by nature) is unaffected; nothing
  else should ever construct ``CzarrConfig(gpu_buffers=True)``.

This backend is benchmark-only (SRI-3): it prototypes czarr as a 4th iohub
zarr backend purely to compare I/O performance against zarr-python,
zarrs-python, and TensorStore. It stays on a branch and is not intended to
be merged to iohub's main branch, so the code bar here is "good enough to
benchmark" rather than production-hardened.
"""

from __future__ import annotations

import platform
from typing import Any

import numpy as np
import zarr

from iohub.core.config import CzarrConfig
from iohub.core.downsample import downsample_block, target_region_to_source
from iohub.core.implementations.zarr_python import ZarrPythonImplementation
from iohub.core.types import StorePath

try:
    if platform.system() != "Linux":
        raise RuntimeError("czarr only supports Linux")
    import cupy as cp
    import czarr as _czarr
    from czarr.codecs.compressors import LZ4, Blosc, Gzip, Zlib
    from czarr.codecs.compressors import Zstd as _CzarrZstd
    from czarr.codecs.filters import BitRound, Delta, FixedScaleOffset, Shuffle
    from czarr.codecs.sharding import CzarrShardingCodec

    _CZARR_AVAILABLE = True
except Exception:  # noqa: BLE001 -- importing czarr touches cupy/CUDA/platform, many failure modes
    cp = None  # type: ignore[assignment]
    _czarr = None  # type: ignore[assignment]
    _CZARR_AVAILABLE = False

#: Codec classes that shadow a stdlib zarr codec_id -- each needs an explicit
#: ``zarr.config["codecs"][<id>]`` override to win the registry lookup.
#: czarr registers them under the shared id at import time, but only
#: ``czarr.configure_gpu()`` sets this override map; we set it ourselves so
#: we can skip ``configure_gpu()``'s other, more invasive global state.
_COMPAT_CODECS = (
    (_CzarrZstd, LZ4, Gzip, Zlib, Blosc, Shuffle, Delta, FixedScaleOffset, BitRound, CzarrShardingCodec)
    if _CZARR_AVAILABLE
    else ()
)


def _to_numpy(data: Any) -> np.ndarray:
    """Bring a czarr read result to host memory (a no-op unless ``gpu_buffers=True``)."""
    if _CZARR_AVAILABLE and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)


class CzarrImplementation(ZarrPythonImplementation):
    """GPU-accelerated array I/O via czarr's codec pipeline.

    Group operations (metadata, hierarchy) are inherited unchanged from
    :class:`ZarrPythonImplementation`. Array creation/opening is also
    inherited unchanged -- with ``config.gpu_buffers=False`` (the default)
    every handle is a plain ``zarr.Array``; the GPU acceleration comes
    entirely from the process-global codec-pipeline swap done in
    ``__init__``, exactly as :class:`ZarrsPythonImplementation` does for the
    ``zarrs`` (Rust) pipeline. See the module docstring for the
    ``gpu_buffers`` opt-in and its trade-offs.

    Raises ``ImportError`` at construction time if czarr/cupy/CUDA is
    unavailable, so ``get_implementation("czarr")`` fails once, clearly,
    rather than registering a backend that silently no-ops or crashes on
    first read.
    """

    def __init__(self, config: CzarrConfig | None = None):
        if not _CZARR_AVAILABLE:
            raise ImportError(
                "czarr is not available: it requires Linux, an NVIDIA GPU, and cupy. "
                "Install czarr with a CUDA-matched extra (e.g. czarr[cu12]) on a "
                "CUDA-capable Linux host."
            )
        self.config = config or CzarrConfig()
        settings: dict[str, Any] = {
            "codec_pipeline.path": "czarr.pipeline.CzarrPipeline",
            "async.concurrency": self.config.async_concurrency,
        }
        if self.config.batch_size is not None:
            settings["codec_pipeline.batch_size"] = self.config.batch_size
        for cls in _COMPAT_CODECS:
            settings[f"codecs.{cls.codec_name}"] = f"{cls.__module__}.{cls.__qualname__}"
        if self.config.gpu_buffers:
            settings["buffer"] = "zarr.core.buffer.gpu.Buffer"
            settings["ndbuffer"] = "zarr.core.buffer.gpu.NDBuffer"
        if self.config.rmm_pool_gb is not None:
            _czarr.use_rmm_pool(initial_size=int(self.config.rmm_pool_gb * (1 << 30)))
        _czarr.register_nvcomp_allocator()
        zarr.config.set(settings)

    # -- Group operations ----------------------------------------------------

    def open_group(self, path: StorePath, mode: str, zarr_format: int | None = None) -> zarr.Group:
        if self.config.gpu_buffers:
            if mode != "r":
                # Verified on an H100 node: GPULocalStore.set() (mode="w") and
                # even a metadata .get() under mode="r+" both raise a native
                # `cuFileError: HANDLE_NOT_REGISTERED` from czarr's cuFile
                # handle registration -- read-only (mode="r") against an
                # already-written store is the only combination confirmed to
                # work. Fail clearly here instead of surfacing that cuFile
                # traceback. Candidate upstream bug report, see SRI-3 discussion.
                raise ValueError(
                    f"CzarrConfig(gpu_buffers=True) only supports mode='r' against an "
                    f"already-written store (got mode={mode!r}). Writing or opening "
                    f"read-write with gpu_buffers=True hits a cuFile HANDLE_NOT_REGISTERED "
                    f"error in czarr's GPULocalStore. Create/write with "
                    f"CzarrConfig(gpu_buffers=False) (or another implementation) first, "
                    f"then reopen read-only with gpu_buffers=True to benchmark GPUDirect reads."
                )
            # GPULocalStore.get() only takes the cuFile fast path when a GPU
            # buffer is requested (which the prototype swap above arranges) --
            # plain LocalStore never issues cuFile calls regardless of buffer
            # prototype. Both pieces are required together for real GDS reads.
            store = _czarr.GPULocalStore(path)
            return zarr.open_group(store=store, mode=mode, zarr_format=zarr_format)
        return super().open_group(path, mode, zarr_format=zarr_format)

    # -- Array I/O (bring GPU reads back to host memory to satisfy the Protocol) -

    def read(self, handle: zarr.Array, selection: Any) -> np.ndarray:
        return _to_numpy(handle[selection])

    def write(self, handle: zarr.Array, selection: Any, data: np.ndarray) -> None:
        handle[selection] = data

    def read_oindex(self, handle: zarr.Array, selection: Any) -> np.ndarray:
        return _to_numpy(handle.oindex[selection])

    def write_oindex(self, handle: zarr.Array, selection: Any, data: np.ndarray) -> None:
        handle.oindex[selection] = data

    # -- High-performance operations ------------------------------------------

    def downsample_region(
        self,
        source: zarr.Array,
        target: zarr.Array,
        target_region: tuple[slice, ...],
        factors: list[int],
        method: str = "mean",
    ) -> None:
        """Read (GPU-accelerated when possible), downsample on host, write.

        Overridden from :class:`ZarrPythonImplementation` only to swap
        ``np.asarray`` for :func:`_to_numpy` -- with ``gpu_buffers=True``,
        czarr hands back a ``cupy.ndarray``, which ``np.asarray`` cannot
        convert implicitly. The downsample math itself is still pure-numpy/
        CPU-bound: czarr has no GPU-accelerated downsample kernel yet
        (candidate upstream PR, see SRI-3 discussion).
        """
        source_region = target_region_to_source(target_region, factors, source.shape)
        source_data = _to_numpy(source[source_region])
        downsampled = downsample_block(source_data, factors, method)
        target[target_region] = downsampled

    # create_array/create_array_v2/open_array/resize/get_shape/get_dtype/
    # get_chunks/get_shards/iter_work_regions/downsample/to_dask/
    # write_from_dask are inherited from ZarrPythonImplementation unchanged:
    # with gpu_buffers=False every handle is a plain zarr.Array, and even
    # with gpu_buffers=True it is still a plain zarr.Array (just backed by a
    # GPULocalStore with GPU buffer prototypes) -- there is no separate
    # array-wrapper class in this design.
