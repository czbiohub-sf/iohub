import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from ndtiff import Dataset
from tifffile import TiffFile

from iohub.convert import TIFFConverter
from iohub.ngff import Position, open_ome_zarr
from iohub.reader import MMStack, NDTiffDataset
from tests.conftest import (
    mm2gamma_ome_tiffs,
    ndtiff_v2_datasets,
    ndtiff_v3_labeled_positions,
)


def pytest_generate_tests(metafunc):
    if "mm2gamma_ome_tiff" in metafunc.fixturenames:
        metafunc.parametrize("mm2gamma_ome_tiff", mm2gamma_ome_tiffs)
    if "ndtiff_datasets" in metafunc.fixturenames:
        metafunc.parametrize(
            "ndtiff_datasets",
            ndtiff_v2_datasets + [ndtiff_v3_labeled_positions],
        )
    if "grid_layout" in metafunc.fixturenames:
        metafunc.parametrize("grid_layout", [True, False])
    if "chunks" in metafunc.fixturenames:
        metafunc.parametrize(
            "chunks", ["XY", "XYZ", (1, 1, 3, 256, 256), None]
        )


def _check_scale_transform(position: Position) -> None:
    """Check scale transformation of the highest resolution level."""
    tf = (
        position.metadata.multiscales[0]
        .datasets[0]
        .coordinate_transformations[0]
    )
    assert tf.type == "scale"
    assert tf.scale[:2] == [1.0, 1.0]


def _check_chunks(
    position: Position, chunks: Literal["XY", "XYZ"] | tuple[int] | None
) -> None:
    """Check chunk size of the highest resolution level."""
    img = position["0"]
    match chunks:
        case "XY":
            assert img.chunks == (1,) * 3 + img.shape[-2:]
        case "XYZ" | None:
            assert img.chunks == (1,) * 2 + img.shape[-3:]
        case tuple():
            assert img.chunks == chunks
        case _:
            assert False


def test_converter_inputs(mm2gamma_ome_tiff, tmpdir):
    # Test that output directory needs to end with ".zarr"
    output = tmpdir / "converted"
    with pytest.raises(ValueError):
        _ = TIFFConverter(mm2gamma_ome_tiff, output)


def test_converter_ometiff(mm2gamma_ome_tiff, grid_layout, chunks, tmpdir):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    output = tmpdir / "converted.zarr"
    converter = TIFFConverter(
        mm2gamma_ome_tiff, output, grid_layout=grid_layout, chunks=chunks
    )
    assert isinstance(converter.reader, MMStack)
    with TiffFile(next(mm2gamma_ome_tiff.glob("*.tif*"))) as tf:
        raw_array = tf.asarray()
        assert (
            converter.summary_metadata == tf.micromanager_metadata["Summary"]
        )
    assert np.prod([d for d in converter.dim if d > 0]) == np.prod(
        raw_array.shape
    )
    assert list(converter.metadata.keys()) == [
        "iohub_version",
        "Summary",
    ]
    converter()
    with open_ome_zarr(output, mode="r") as result:
        intensity = 0
        if grid_layout and converter.p > 1:
            assert len(result) < converter.p
        for pos_name, pos in result.positions():
            _check_scale_transform(pos)
            _check_chunks(pos, chunks)
            intensity += pos["0"][:].sum()
            assert os.path.isfile(
                os.path.join(
                    output, pos_name, "0", "image_plane_metadata.json"
                )
            )
        assert intensity == raw_array.sum()


@pytest.fixture(scope="module")
def example_ome_tiff() -> Path:
    for d in mm2gamma_ome_tiffs:
        if d.name == "mm2.0-20201209_4p_2t_5z_1c_512k_1":
            return d


@pytest.fixture(scope="function")
def mock_hcs_ome_tiff_reader(
    example_ome_tiff, monkeypatch: pytest.MonkeyPatch
):
    # dataset with 4 positions without HCS site names
    mock_stage_positions = [
        {"Label": "A1-Site_0"},
        {"Label": "A1-Site_1"},
        {"Label": "B4-Site_0"},
        {"Label": "H12-Site_0"},
    ]
    expected_ngff_name = {"A/1/0", "A/1/1", "B/4/0", "H/12/0"}
    monkeypatch.setattr(
        "iohub.convert.MMStack.stage_positions",
        mock_stage_positions,
    )
    return example_ome_tiff, expected_ngff_name


@pytest.fixture(scope="function")
def mock_non_hcs_ome_tiff_reader(
    example_ome_tiff, monkeypatch: pytest.MonkeyPatch
):
    # dataset with 4 positions without HCS site names
    mock_stage_positions = [
        {"Label": "0"},
        {"Label": "1"},
        {"Label": "2"},
        {"Label": "3"},
    ]
    monkeypatch.setattr(
        "iohub.convert.MMStack.stage_positions",
        mock_stage_positions,
    )
    return example_ome_tiff


def test_converter_ometiff_mock_hcs(mock_hcs_ome_tiff_reader, tmpdir):
    data, expected_ngff_name = mock_hcs_ome_tiff_reader
    output = tmpdir / "converted.zarr"
    converter = TIFFConverter(data, output, hcs_plate=True)
    converter()
    with open_ome_zarr(output, mode="r") as plate:
        assert expected_ngff_name == {name for name, _ in plate.positions()}


def test_converter_ometiff_mock_non_hcs(mock_non_hcs_ome_tiff_reader, tmpdir):
    data = mock_non_hcs_ome_tiff_reader
    output = tmpdir / "converted.zarr"
    with pytest.raises(ValueError, match="HCS position labels"):
        TIFFConverter(data, output, hcs_plate=True)


def test_converter_ometiff_hcs_numerical(example_ome_tiff, tmpdir):
    output = tmpdir / "converted.zarr"
    converter = TIFFConverter(example_ome_tiff, output, hcs_plate=True)
    converter()
    with open_ome_zarr(output, mode="r") as plate:
        for name, _ in plate.positions():
            for segment in name.split("/"):
                assert segment.isdigit()


def test_converter_ndtiff(ndtiff_datasets: Path, grid_layout, chunks, tmpdir):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    output = tmpdir / "converted.zarr"
    converter = TIFFConverter(
        ndtiff_datasets, output, grid_layout=grid_layout, chunks=chunks
    )
    assert isinstance(converter.reader, NDTiffDataset)
    raw_array = np.asarray(Dataset(str(ndtiff_datasets)).as_array())
    assert np.prod([d for d in converter.dim if d > 0]) == np.prod(
        raw_array.shape
    )
    assert list(converter.metadata.keys()) == [
        "iohub_version",
        "Summary",
    ]
    converter()
    with open_ome_zarr(output, mode="r") as result:
        intensity = 0
        for pos_name, pos in result.positions():
            _check_scale_transform(pos)
            _check_chunks(pos, chunks)
            intensity += pos["0"][:].sum()
            assert os.path.isfile(
                os.path.join(
                    output, pos_name, "0", "image_plane_metadata.json"
                )
            )
    assert intensity == raw_array.sum()
    with open(
        os.path.join(output, pos_name, "0", "image_plane_metadata.json")
    ) as f:
        metadata = json.load(f)
        key = "0/0/0"
        assert key in metadata
        assert "ElapsedTime-ms" in metadata[key]


def test_converter_ndtiff_v3_position_labels(tmpdir):
    output = tmpdir / "converted.zarr"
    converter = TIFFConverter(ndtiff_v3_labeled_positions, output)
    converter()
    with open_ome_zarr(output, mode="r") as result:
        assert result.channel_names == ["Channel0"]
        assert [name.split("/")[1] for name, _ in result.positions()] == [
            "Pos0",
            "Pos1",
            "Pos2",
        ]
