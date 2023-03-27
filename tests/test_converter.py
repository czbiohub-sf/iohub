import logging
import os
from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ndtiff import Dataset
from tifffile import TiffFile, TiffSequence

from iohub.convert import TIFFConverter
from iohub.ngff import open_ome_zarr
from iohub.reader import (
    MicromanagerOmeTiffReader,
    MicromanagerSequenceReader,
    NDTiffReader,
)


@given(grid_layout=st.booleans(), label_positions=st.booleans())
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=20000
)
def test_converter_ometiff(
    setup_test_data, setup_mm2gamma_ome_tiffs, grid_layout, label_positions
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
                intensity += pos["0"][:].sum()
        assert intensity == raw_array.sum()


@given(grid_layout=st.booleans(), label_positions=st.booleans())
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=20000
)
def test_converter_ndtiff(
    setup_test_data, setup_pycromanager_test_data, grid_layout, label_positions
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
                intensity += pos["0"][:].sum()
        assert intensity == raw_array.sum()


@given(grid_layout=st.booleans(), label_positions=st.booleans())
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=20000
)
def test_converter_singlepagetiff(
    setup_test_data,
    setup_mm2gamma_singlepage_tiffs,
    grid_layout,
    label_positions,
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
        )
        assert isinstance(converter.reader, MicromanagerSequenceReader)
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
