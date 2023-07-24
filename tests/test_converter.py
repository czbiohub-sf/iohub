import logging
import os
from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ndtiff import Dataset
from tifffile import TiffFile, TiffSequence

from iohub.convert import TIFFConverter
from iohub.ngff import Position, open_ome_zarr
from iohub.reader import (
    MicromanagerOmeTiffReader,
    MicromanagerSequenceReader,
    NDTiffReader,
)


CONVERTER_TEST_SETTINGS = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=20000,
)

CONVERTER_TEST_GIVEN = dict(
    grid_layout=st.booleans(),
    label_positions=st.booleans(),
    scale_voxels=st.booleans(),
)


def _check_scale_transform(position: Position, scale_voxels: bool):
    tf = position.metadata.multiscales[0].coordinate_transformations[0]
    if scale_voxels:
        assert tf.type == "scale"
        assert tf.scale[:2] == [1.0, 1.0]
    else:
        assert tf.type == "identity"


@given(**CONVERTER_TEST_GIVEN)
@settings(CONVERTER_TEST_SETTINGS)
def test_converter_ometiff(
    setup_test_data,
    setup_mm2gamma_ome_tiffs,
    grid_layout,
    label_positions,
    scale_voxels,
):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    _, _, data = setup_mm2gamma_ome_tiffs
    with TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "converted.zarr")
        converter = TIFFConverter(
            data,
            output,
            grid_layout=grid_layout,
            label_positions=label_positions,
            scale_voxels=scale_voxels,
        )
        assert isinstance(converter.reader, MicromanagerOmeTiffReader)
        with TiffSequence(glob(os.path.join(data, "*.tif*"))) as ts:
            raw_array = ts.asarray()
            with TiffFile(ts[0]) as tf:
                assert (
                    converter.summary_metadata
                    == tf.micromanager_metadata["Summary"]
                )
        assert np.prod([d for d in converter.dim if d > 0]) == np.prod(
            raw_array.shape
        )
        assert list(converter.metadata.keys()) == [
            "iohub_version",
            "Summary",
        ]
        converter.run(check_image=True)
        with open_ome_zarr(output, mode="r") as result:
            intensity = 0
            for _, pos in result.positions():
                _check_scale_transform(pos, scale_voxels)
                intensity += pos["0"][:].sum()
        assert intensity == raw_array.sum()


def test_converter_ometiff_hcs_not_available(
    setup_test_data, setup_mm2gamma_ome_tiffs
):
    _, _, data = setup_mm2gamma_ome_tiffs
    with TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "converted.zarr")
        with pytest.raises(ValueError, match="position"):
            _ = TIFFConverter(data, output, hcs_plate=True)


@pytest.fixture(scope="function")
def mock_hcs_ome_tiff_reader(
    setup_mm2gamma_ome_tiffs, monkeypatch: pytest.MonkeyPatch
):
    all_ometiffs, _, _ = setup_mm2gamma_ome_tiffs
    # dataset with 4 positions without HCS site names
    data = os.path.join(all_ometiffs, "mm2.0-20201209_4p_2t_5z_1c_512k_1")
    mock_stage_positions = [
        {"Label": "A1-Site_0"},
        {"Label": "A1-Site_1"},
        {"Label": "B4-Site_0"},
        {"Label": "H12-Site_0"},
    ]
    expected_ngff_name = {"A/1/0", "A/1/1", "B/4/0", "H/12/0"}
    monkeypatch.setattr(
        "iohub.convert.MicromanagerOmeTiffReader.stage_positions",
        mock_stage_positions,
    )
    return data, expected_ngff_name


def test_converter_ometiff_mock_hcs(setup_test_data, mock_hcs_ome_tiff_reader):
    data, expected_ngff_name = mock_hcs_ome_tiff_reader
    with TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "converted.zarr")
        converter = TIFFConverter(data, output, hcs_plate=True)
        converter.run()
        with open_ome_zarr(output, mode="r") as plate:
            assert expected_ngff_name == {
                name for name, _ in plate.positions()
            }


@given(**CONVERTER_TEST_GIVEN)
@settings(CONVERTER_TEST_SETTINGS)
def test_converter_ndtiff(
    setup_test_data,
    setup_pycromanager_test_data,
    grid_layout,
    label_positions,
    scale_voxels,
):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    _, _, data = setup_pycromanager_test_data
    with TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "converted.zarr")
        converter = TIFFConverter(
            data,
            output,
            grid_layout=grid_layout,
            label_positions=label_positions,
            scale_voxels=scale_voxels,
        )
        assert isinstance(converter.reader, NDTiffReader)
        raw_array = np.asarray(Dataset(data).as_array())
        assert np.prod([d for d in converter.dim if d > 0]) == np.prod(
            raw_array.shape
        )
        assert list(converter.metadata.keys()) == [
            "iohub_version",
            "Summary",
        ]
        converter.run(check_image=True)
        with open_ome_zarr(output, mode="r") as result:
            intensity = 0
            for _, pos in result.positions():
                _check_scale_transform(pos, scale_voxels)
                intensity += pos["0"][:].sum()
        assert intensity == raw_array.sum()


@given(**CONVERTER_TEST_GIVEN)
@settings(CONVERTER_TEST_SETTINGS)
def test_converter_singlepagetiff(
    setup_test_data,
    setup_mm2gamma_singlepage_tiffs,
    grid_layout,
    label_positions,
    scale_voxels,
    caplog,
):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    _, _, data = setup_mm2gamma_singlepage_tiffs
    with TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "converted.zarr")
        converter = TIFFConverter(
            data,
            output,
            grid_layout=grid_layout,
            label_positions=label_positions,
            scale_voxels=scale_voxels,
        )
        assert isinstance(converter.reader, MicromanagerSequenceReader)
        if scale_voxels:
            assert "Pixel size detection is not supported" in caplog.text
        with TiffSequence(glob(os.path.join(data, "**/*.tif*"))) as ts:
            raw_array = ts.asarray()
        assert np.prod([d for d in converter.dim if d > 0]) == np.prod(
            raw_array.shape
        )
        assert list(converter.metadata.keys()) == [
            "iohub_version",
            "Summary",
        ]
        converter.run(check_image=True)
        with open_ome_zarr(output, mode="r") as result:
            intensity = 0
            for _, pos in result.positions():
                intensity += pos["0"][:].sum()
        assert intensity == raw_array.sum()
