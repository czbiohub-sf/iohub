import os
import pathlib
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from iohub._version import __version__
from iohub.cli.cli import cli


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


@given(verbose=st.booleans())
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=2000
)
def test_cli_info_mock(setup_test_data, setup_mm2gamma_ome_tiffs, verbose):
    _, f1, f2 = setup_mm2gamma_ome_tiffs
    runner = CliRunner()
    with patch("iohub.cli.cli.print_info") as mock:
        cmd = ["info", f1, f2]
        if verbose:
            cmd += ["-v"]
        result = runner.invoke(cli, cmd)
        # resolve path with pathlib to be consistent with `click.Path`
        # this will not normalize partition symbol to lower case on Windows
        mock.assert_called_with(
            str(pathlib.Path(f2).resolve()), verbose=verbose
        )
        assert result.exit_code == 0
        assert "Reading" in result.output


@given(verbose=st.booleans())
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=2000
)
def test_cli_info_ndtiff(
    setup_test_data, setup_pycromanager_test_data, verbose
):
    _, _, data = setup_pycromanager_test_data
    runner = CliRunner()
    cmd = ["info", data]
    if verbose:
        cmd += ["-v"]
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
    assert re.search(r"Positions:\s+2", result.output)


@given(verbose=st.booleans())
def test_cli_info_ome_zarr(setup_test_data, setup_hcs_ref, verbose):
    runner = CliRunner()
    cmd = ["info", setup_hcs_ref]
    if verbose:
        cmd += ["-v"]
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
    assert re.search(r"Wells:\s+1", result.output)


@given(f=st.booleans(), g=st.booleans(), p=st.booleans(), s=st.booleans())
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=20000
)
def test_cli_convert_ome_tiff(
    setup_test_data, setup_mm2gamma_ome_tiffs, f, g, p, s
):
    _, _, input_dir = setup_mm2gamma_ome_tiffs
    runner = CliRunner()
    f = "-f ometiff" if f else ""
    g = "-g" if g else ""
    p = "-p" if p else ""
    with TemporaryDirectory() as tmp_dir:
        output_dir = os.path.join(tmp_dir, "converted.zarr")
        cmd = ["convert", "-i", input_dir, "-o", output_dir, "-s", s]
        if f:
            cmd += ["-f", "ometiff"]
        if g:
            cmd += ["-g"]
        if p:
            cmd += ["-p"]
        result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
    assert "Status" in result.output
