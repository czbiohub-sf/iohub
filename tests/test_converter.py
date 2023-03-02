import logging
import os
import shutil
from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
import zarr
from tifffile import TiffFile, TiffSequence

from iohub.convert import TIFFConverter
from iohub.ngff import open_ome_zarr
from iohub.reader import MicromanagerOmeTiffReader, NDTiffReader


def test_converter_ometiff(setup_test_data, setup_mm2gamma_ome_tiffs):
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    test_data, _, _ = setup_mm2gamma_ome_tiffs
    for data in os.scandir(test_data):
        if data.name.startswith("."):
            # skip system files such as .DS_store
            continue
        with TemporaryDirectory() as tmp_dir:
            output = os.path.join(tmp_dir, "converted.zarr")
            converter = TIFFConverter(data.path, output)
            assert isinstance(converter.reader, MicromanagerOmeTiffReader)
            with TiffSequence(glob(os.path.join(data.path, "*.tif*"))) as ts:
                raw_array = ts.asarray()
                with TiffFile(ts[0]) as tf:
                    assert (
                        converter.summary_metadata
                        == tf.micromanager_metadata["Summary"]
                    )
            assert converter.dtype == raw_array.dtype
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


def test_pycromanager_converter_initialize(
    setup_data_save_folder, get_pycromanager_data_dir
):
    folder, pm_data = get_pycromanager_data_dir
    save_folder = setup_data_save_folder

    input = pm_data
    output = os.path.join(
        save_folder, "mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1.zarr"
    )

    if os.path.exists(output):
        shutil.rmtree(output)

    converter = ZarrConverter(input, output)

    assert isinstance(converter.reader, WaveorderReader)
    assert isinstance(converter.writer, WaveorderWriter)
    assert converter.summary_metadata is not None

    assert converter.dim_order == ["position", "time", "channel", "z"]
    assert converter.p_dim == 0
    assert converter.t_dim == 1
    assert converter.c_dim == 2
    assert converter.z_dim == 3
    assert converter.p == 2
    assert converter.t == 3
    assert converter.c == 2
    assert converter.z == 7


def test_pycromanager_converter_run(
    setup_data_save_folder, get_pycromanager_data_dir
):
    folder, pm_data = get_pycromanager_data_dir
    save_folder = setup_data_save_folder

    input = pm_data
    output = os.path.join(
        save_folder, "mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1.zarr"
    )

    if os.path.exists(output):
        shutil.rmtree(output)

    converter = ZarrConverter(input, output)
    wo_dataset = WaveorderReader(input)

    converter.run_conversion()
    zs = zarr.open(output, "r")

    assert os.path.exists(
        os.path.join(
            save_folder,
            "mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1_ImagePlaneMetadata.txt",
        )
    )

    cnt = 0
    for t in range(3):
        for p in range(2):
            for c in range(2):
                for z in range(7):
                    image = zs["Row_0"][f"Col_{p}"][f"Pos_00{p}"]["arr_0"][
                        t, c, z
                    ]
                    wo_image = wo_dataset.get_image(p, t, c, z)
                    assert np.array_equal(image, wo_image)
                    cnt += 1
