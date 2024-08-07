from pathlib import Path

import numpy as np
import pytest
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader

from iohub.ngff.nodes import (
    Position,
    TransformationMeta,
    _pad_shape,
    open_ome_zarr,
)


def _mock_fov(
    tmp_path: Path,
    shape: tuple[int, ...],
    scale: tuple[float, float, float],
) -> Position:
    ds_path = tmp_path / "ds.zarr"
    channels = [str(i) for i in range(shape[1])]

    fov = open_ome_zarr(
        ds_path,
        layout="fov",
        mode="a",
        channel_names=channels,
        axes=Position._DEFAULT_AXES[-len(shape) :],
    )

    transform = [
        TransformationMeta(
            type="scale",
            scale=_pad_shape(scale, len(shape)),
        )
    ]

    if "0" not in fov:
        fov.create_zeros(
            "0",
            shape=shape,
            dtype=np.uint16,
            chunks=_pad_shape(shape[-2:], len(shape)),
            transform=transform,
        )

    return fov


@pytest.mark.parametrize("ndim", [2, 5])
def test_pyramid(tmp_path: Path, ndim: int) -> None:
    # not all shapes not divisible by 2
    shape = (2, 2, 67, 115, 128)[-ndim:]
    scale = (2, 0.5, 0.5)[-min(3, ndim) :]
    levels = 4

    fov = _mock_fov(tmp_path, shape, scale)

    fov.initialize_pyramid(levels=levels)

    assert len(fov.array_keys()) == levels

    for level in range(levels):
        assert str(level) in fov.array_keys()

        level_shape = np.asarray(fov[str(level)].shape)
        ratio = np.ceil(shape / level_shape).astype(int)

        assert np.all(ratio[:-3] == 1)  # time and channel aren't scaled
        assert np.all(ratio[-3:] == 2**level)

        level_scale = np.asarray(
            fov.metadata.multiscales[0]
            .datasets[level]
            .coordinate_transformations[0]
            .scale
        )
        assert np.all(level_scale[:-3] == 1)
        assert np.allclose(level_scale[-3:] / scale, 2**level)

        assert fov.metadata.multiscales[0].datasets[level].path == str(level)

    reader = Reader(parse_url(tmp_path / "ds.zarr"))
    for node in reader():
        assert any(isinstance(spec, Multiscales) for spec in node.specs)
