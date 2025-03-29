import random
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from iohub import open_ome_zarr
from iohub._version import __version__
from iohub.cli.cli import cli
from tests.conftest import (
    hcs_ref,
    mm2gamma_ome_tiffs,
    ndtiff_v2_datasets,
    ndtiff_v3_labeled_positions,
)

from ..ngff.test_ngff import _temp_copy


def pytest_generate_tests(metafunc):
    if "mm2gamma_ome_tiff" in metafunc.fixturenames:
        metafunc.parametrize("mm2gamma_ome_tiff", mm2gamma_ome_tiffs)
    if "verbose" in metafunc.fixturenames:
        metafunc.parametrize("verbose", ["-v", False])
    if "ndtiff_dataset" in metafunc.fixturenames:
        metafunc.parametrize(
            "ndtiff_dataset",
            ndtiff_v2_datasets + [ndtiff_v3_labeled_positions],
        )


def test_cli_entry():
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_cli_help():
    runner = CliRunner()
    for option in ("-h", "--help"):
        result = runner.invoke(cli, option)
        assert result.exit_code == 0
        assert "Usage" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, "--version")
    assert result.exit_code == 0
    assert "version" in result.output
    assert str(__version__) in result.output


def test_cli_info_mock(mm2gamma_ome_tiff, verbose):
    runner = CliRunner()
    # resolve path with pathlib to be consistent with `click.Path`
    # this will not normalize partition symbol to lower case on Windows
    path = mm2gamma_ome_tiff.resolve()
    with patch("iohub.cli.cli.print_info") as mock:
        cmd = ["info", str(path)]
        if verbose:
            cmd.append(verbose)
        result = runner.invoke(cli, cmd)
        mock.assert_called_with(path, verbose=bool(verbose))
        assert result.exit_code == 0
        assert "Reading" in result.output


def test_cli_info_unknown(tmp_path):
    runner = CliRunner()
    empty_file = tmp_path / "unknown.txt"
    empty_file.touch()
    result = runner.invoke(cli, ["info", str(tmp_path)])
    assert result.exit_code == 0
    assert "No compatible" in result.output


def test_cli_info_ndtiff(ndtiff_dataset, verbose):
    runner = CliRunner()
    cmd = ["info", str(ndtiff_dataset)]
    if verbose:
        cmd.append(verbose)
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
    assert "Channel names" in result.output
    assert re.search(r"FOVs:\s+\d", result.output)
    assert "scale (um)" in result.output


def test_cli_info_ome_zarr(verbose):
    runner = CliRunner()
    cmd = ["info", str(hcs_ref)]
    if verbose:
        cmd.append(verbose)
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
    assert re.search(r"Wells:\s+1", result.output)
    assert ("Chunk size" in result.output) == bool(verbose)
    assert ("No. bytes decompressed" in result.output) == bool(verbose)
    # Test on single position
    result_pos = runner.invoke(cli, ["info", str(hcs_ref / "B" / "03" / "0")])
    assert "Channel names" in result_pos.output
    assert "scale (um)" in result_pos.output
    assert "Chunk size" in result_pos.output
    assert "84.4 MiB" in result_pos.output


@pytest.mark.parametrize("grid_layout", ["-g", None])
def test_cli_convert_ome_tiff(grid_layout, tmpdir):
    dataset = mm2gamma_ome_tiffs[0]
    runner = CliRunner()
    output_dir = tmpdir / "converted.zarr"
    cmd = ["convert", "-i", str(dataset), "-o", output_dir]
    if grid_layout:
        cmd.append(grid_layout)
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0, result.output
    assert "Converting" in result.output


def test_cli_set_scale(caplog):
    with _temp_copy(hcs_ref) as store_path:
        store_path = Path(store_path)
        position_path = Path(store_path) / "B" / "03" / "0"

        with open_ome_zarr(
            position_path, layout="fov", mode="r+"
        ) as input_dataset:
            old_scale = input_dataset.scale

        random_z = random.uniform(0, 1)

        runner = CliRunner()
        result_pos = runner.invoke(
            cli,
            [
                "set-scale",
                "-i",
                str(position_path),
                "-z",
                random_z,
                "-y",
                0.5,
                "-x",
                0.5,
            ],
        )
        assert result_pos.exit_code == 0
        assert any("Updating" in record.message for record in caplog.records)
        with open_ome_zarr(position_path, layout="fov") as output_dataset:
            assert tuple(output_dataset.scale[-3:]) == (random_z, 0.5, 0.5)
            assert output_dataset.scale != old_scale
            for i, record in enumerate(
                output_dataset.zattrs["iohub"]["previous_transforms"]
            ):
                for transform in record["transforms"]:
                    if transform["type"] == "scale":
                        assert transform["scale"][-3:][i] == old_scale[-3:][i]

        # Test plate-expands-into-positions behavior
        runner = CliRunner()
        result_pos = runner.invoke(
            cli,
            [
                "set-scale",
                "-i",
                str(store_path),
                "-x",
                0.1,
            ],
        )
        with open_ome_zarr(position_path, layout="fov") as output_dataset:
            assert output_dataset.scale[-1] == 0.1
            for transform in output_dataset.zattrs["iohub"][
                "previous_transforms"
            ][-1]["transforms"]:
                if transform["type"] == "scale":
                    assert transform["scale"][-1] == 0.5


def test_cli_rename_wells_help():
    runner = CliRunner()
    cmd = ["rename-wells"]
    for option in ("-h", "--help"):
        cmd.append(option)
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 0
        assert ">> iohub rename-wells" in result.output


def test_cli_rename_wells(csv_data_file_1):
    with _temp_copy(hcs_ref) as store_path:
        runner = CliRunner()
        cmd = [
            "rename-wells",
            "-i",
            str(store_path),
            "-c",
            str(csv_data_file_1),
        ]
        result = runner.invoke(cli, cmd)

        assert result.exit_code == 0
        assert "Renaming" in result.output
