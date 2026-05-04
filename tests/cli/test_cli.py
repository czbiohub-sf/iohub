import random
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from iohub import __version__, open_ome_zarr
from iohub.cli.cli import cli
from tests.conftest import (
    hcs_ref,
    mm2gamma_ome_tiffs,
    ndtiff_v2_datasets,
    ndtiff_v3_labeled_positions,
)
from tests.ngff.test_ngff import _temp_copy


def pytest_generate_tests(metafunc):
    if "mm2gamma_ome_tiff" in metafunc.fixturenames:
        metafunc.parametrize("mm2gamma_ome_tiff", mm2gamma_ome_tiffs)
    if "verbose" in metafunc.fixturenames:
        metafunc.parametrize("verbose", ["-v", False])
    if "ndtiff_dataset" in metafunc.fixturenames:
        metafunc.parametrize(
            "ndtiff_dataset",
            [*ndtiff_v2_datasets, ndtiff_v3_labeled_positions],
        )


def test_cli_entry():
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 2
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


def test_cli_convert_version(tmpdir):
    dataset = mm2gamma_ome_tiffs[0]
    runner = CliRunner()
    output_dir = tmpdir / "converted.zarr"
    cmd = ["convert", "-i", str(dataset), "-o", output_dir, "-v", "0.5"]
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0, result.output
    with open_ome_zarr(output_dir, mode="r") as store:
        assert store.version == "0.5"


def test_cli_convert_invalid_version(tmpdir):
    dataset = mm2gamma_ome_tiffs[0]
    runner = CliRunner()
    output_dir = tmpdir / "converted.zarr"
    cmd = ["convert", "-i", str(dataset), "-o", output_dir, "-v", "0.3"]
    result = runner.invoke(cli, cmd)
    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_cli_info_verbose_ozx_shows_rfc9(tmpdir):
    """``iohub info -v <ozx>`` shows the FOV summary plus the RFC-9 block."""
    import numpy as np

    from iohub.core.ozx import pack_ozx
    from tests.conftest import make_fov_zarr

    src = Path(tmpdir) / "src.zarr"
    make_fov_zarr(src, np.zeros((1, 1, 1, 4, 4), dtype=np.uint8))
    ozx = Path(tmpdir) / "src.ozx"
    pack_ozx(src, ozx)

    runner = CliRunner()
    res = runner.invoke(cli, ["info", "-v", str(ozx)])
    assert res.exit_code == 0, res.output
    # FOV summary still present.
    assert "Format:" in res.output
    assert "Channel names:" in res.output
    # RFC-9 block appended in verbose mode.
    assert "=== RFC-9 archive ===" in res.output
    assert "OME version:" in res.output
    assert "jsonFirst:" in res.output


def test_cli_info_non_verbose_ozx_skips_rfc9(tmpdir):
    """Non-verbose ``info`` on a .ozx omits the RFC-9 block."""
    import numpy as np

    from iohub.core.ozx import pack_ozx
    from tests.conftest import make_fov_zarr

    src = Path(tmpdir) / "src.zarr"
    make_fov_zarr(src, np.zeros((1, 1, 1, 2, 2), dtype=np.uint8))
    ozx = Path(tmpdir) / "src.ozx"
    pack_ozx(src, ozx)

    runner = CliRunner()
    res = runner.invoke(cli, ["info", str(ozx)])
    assert res.exit_code == 0, res.output
    assert "RFC-9 archive" not in res.output


def test_cli_convert_zarr_to_ozx_and_back(tmpdir):
    """`iohub convert` packs .zarr → .ozx and unpacks .ozx → .zarr based on suffix."""
    import numpy as np

    from tests.conftest import make_fov_zarr

    src = Path(tmpdir) / "src.zarr"
    data = np.arange(16, dtype=np.uint8).reshape(1, 1, 1, 4, 4)
    make_fov_zarr(src, data)

    runner = CliRunner()

    ozx = Path(tmpdir) / "out.ozx"
    res = runner.invoke(cli, ["convert", "-i", str(src), "-o", str(ozx)])
    assert res.exit_code == 0, res.output
    assert "packed" in res.output
    assert ozx.is_file()

    # Unpack: ozx → zarr
    restored = Path(tmpdir) / "restored.zarr"
    res = runner.invoke(cli, ["convert", "-i", str(ozx), "-o", str(restored)])
    assert res.exit_code == 0, res.output
    assert "unpacked" in res.output
    assert restored.is_dir()

    with open_ome_zarr(restored, mode="r") as pos:
        np.testing.assert_array_equal(pos["0"][:], np.arange(16, dtype=np.uint8).reshape(1, 1, 1, 4, 4))


@pytest.mark.parametrize(
    ("flag", "value", "route"),
    [
        ("--chunks", "XY", "pack"),
        ("-g", None, "pack"),
        ("--chunks", "XY", "unpack"),
        ("-g", None, "unpack"),
        ("--ome-zarr-version", "0.5", "unpack"),
    ],
)
def test_cli_convert_rejects_irrelevant_flags(tmpdir, flag, value, route):
    """TIFF-only flags on pack/unpack and version on unpack must error loudly."""
    import numpy as np

    from tests.conftest import make_fov_zarr

    src_zarr = Path(tmpdir) / "src.zarr"
    make_fov_zarr(src_zarr, np.zeros((1, 1, 1, 2, 2), dtype=np.uint8))
    if route == "pack":
        src, dst = src_zarr, Path(tmpdir) / "out.ozx"
    else:
        ozx = Path(tmpdir) / "src.ozx"
        from iohub.core.ozx import pack_ozx as _pack

        _pack(src_zarr, ozx)
        src, dst = ozx, Path(tmpdir) / "out.zarr"

    cmd = ["convert", "-i", str(src), "-o", str(dst), flag]
    if value is not None:
        cmd.append(value)

    runner = CliRunner()
    res = runner.invoke(cli, cmd)
    assert res.exit_code != 0
    assert "do not apply" in res.output or "apply only" in res.output


def test_cli_set_scale(caplog):
    with _temp_copy(hcs_ref) as store_path:
        store_path = Path(store_path)
        position_path = Path(store_path) / "B" / "03" / "0"

        with open_ome_zarr(position_path, layout="fov", mode="r+") as input_dataset:
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
                str(random_z),
                "-y",
                "0.5",
                "-x",
                "0.5",
            ],
        )
        assert result_pos.exit_code == 0
        assert any("Updating" in record.message for record in caplog.records)
        with open_ome_zarr(position_path, layout="fov") as output_dataset:
            assert tuple(output_dataset.scale[-3:]) == (random_z, 0.5, 0.5)
            assert output_dataset.scale != old_scale
            for i, record in enumerate(output_dataset.zattrs["iohub"]["previous_transforms"]):
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
            for transform in output_dataset.zattrs["iohub"]["previous_transforms"][-1]["transforms"]:
                if transform["type"] == "scale":
                    assert transform["scale"][-1] == 0.5


def test_cli_rename_wells_help():
    runner = CliRunner()
    cmd = ["rename-wells"]
    for option in ("-h", "--help"):
        cmd.append(option)
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 0


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
