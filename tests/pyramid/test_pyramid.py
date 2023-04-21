from typing import Tuple
from pathlib import Path

import numpy as np

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Multiscales

from iohub.ngff import open_ome_zarr, Position
from iohub.ngff_meta import TransformationMeta


def _mock_fov(
    tmp_path: Path,
    shape: Tuple[int, ...],
    scale: Tuple[float, float, float],
) -> Position:

    ds_path = tmp_path / "ds.zarr"
    channels = [str(i) for i in range(shape[1])]

    fov = open_ome_zarr(ds_path, layout="fov", mode="a", channel_names=channels)
    transform = [
        TransformationMeta(
            type="scale",
            scale=(1, 1) + scale,
        )
    ]

    if "0" not in fov.array_keys():
        fov.create_zeros(
            "0",
            shape=shape,
            dtype=np.uint16,
            chunks=(1, 1, 128) + shape[-2:],
            transform=transform,
        )
    
    return fov


def test_pyramid(tmp_path: Path) -> None:
    shape = (2, 2, 67, 115, 128)  # not all shapes not divisible by 2
    scale = (2, 0.5, 0.5)
    levels = 4

    fov = _mock_fov(tmp_path, shape, scale)

    fov.initialize_pyramid(levels=levels)

    assert len(fov.array_keys()) == levels

    for level in range(levels):
        assert str(level) in fov.array_keys()

        level_shape = np.asarray(fov[str(level)].shape)
        ratio = np.ceil(shape / level_shape).astype(int) 

        assert np.all(ratio[:2] == 1)  # time and channel aren't scaled
        assert np.all(ratio[2:] == 2 ** level)

        level_scale = np.asarray(
            fov.metadata.multiscales[0].datasets[level].coordinate_transformations[0].scale
        )
        assert np.all(level_scale[:2] == 1)
        assert np.allclose(scale / level_scale[2:], 2 ** level)

        assert fov.metadata.multiscales[0].datasets[level].path == str(level)

    reader = Reader(parse_url(tmp_path / "ds.zarr"))
    for node in reader():
        assert any(isinstance(spec, Multiscales) for spec in node.specs)
