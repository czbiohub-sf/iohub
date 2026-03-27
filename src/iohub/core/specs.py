"""ArraySpec -- zarr v3 array creation specification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import DTypeLike

from iohub.core.config import CompressorConfig


@dataclass(frozen=True)
class ArraySpec:
    """Zarr v3 array creation specification.

    Passed to ``ZarrImplementation.create_array()`` to create new arrays.
    Backend-agnostic: zarr-python and TensorStore both accept this.
    """

    shape: tuple[int, ...]
    data_type: str
    chunk_grid: dict
    chunk_key_encoding: dict
    codecs: list[dict]
    fill_value: int | float = 0
    dimension_names: list[str] | None = None
    attributes: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        chunks: tuple[int, ...],
        shards: tuple[int, ...] | None = None,
        fill_value: int | float = 0,
        dimension_names: list[str] | None = None,
        compressor: CompressorConfig | None = None,
    ) -> ArraySpec:
        """Build a zarr v3 ``ArraySpec`` from iohub parameters.

        Only used for zarr v3 (OME-NGFF 0.5) stores.
        Zarr v2 stores continue to use ``group.create_array()`` directly.
        """
        comp = compressor or CompressorConfig()
        codecs: list[dict] = [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {
                "name": "blosc",
                "configuration": {
                    "cname": comp.cname,
                    "clevel": comp.clevel,
                    "typesize": np.dtype(dtype).itemsize,
                    "shuffle": comp.shuffle,
                },
            },
        ]
        if shards:
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": list(shards)},
            }
            codecs = [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": list(chunks),
                        "codecs": codecs,
                        "index_codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "crc32c"},
                        ],
                    },
                }
            ]
        else:
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": list(chunks)},
            }

        return cls(
            shape=shape,
            data_type=str(np.dtype(dtype)),
            chunk_grid=chunk_grid,
            chunk_key_encoding={
                "name": "default",
                "configuration": {"separator": "/"},
            },
            codecs=codecs,
            fill_value=fill_value,
            dimension_names=dimension_names,
        )


def make_array_spec(
    shape: tuple[int, ...],
    dtype: DTypeLike,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None = None,
    fill_value: int | float = 0,
    dimension_names: list[str] | None = None,
    compressor: CompressorConfig | None = None,
) -> ArraySpec:
    """Alias for ``ArraySpec.create()`` — kept for backwards compatibility."""
    return ArraySpec.create(
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        shards=shards,
        fill_value=fill_value,
        dimension_names=dimension_names,
        compressor=compressor,
    )
