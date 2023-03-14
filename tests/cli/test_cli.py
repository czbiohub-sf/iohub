import os
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


def test_cli_info_mock(setup_test_data, setup_mm2gamma_ome_tiffs):
    _, f1, f2 = setup_mm2gamma_ome_tiffs
    runner = CliRunner()
    with patch("iohub.cli.cli.print_reader_info") as mock:
        result = runner.invoke(cli, f"info {f1} {f2}")
        mock.assert_called()
        assert result.exit_code == 0
        assert "Reading" in result.output


def test_cli_info(setup_test_data, setup_pycromanager_test_data):
    _, _, data = setup_pycromanager_test_data
    runner = CliRunner()
    result = runner.invoke(cli, f"info {data}")
    assert result.exit_code == 0
    assert "Positions:\t 2" in result.output


@given(f=st.booleans(), g=st.booleans(), p=st.booleans())
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=20000
)
def test_cli_convert_ome_tiff(
    setup_test_data, setup_mm2gamma_ome_tiffs, f, g, p
):
    _, _, input_dir = setup_mm2gamma_ome_tiffs
    runner = CliRunner()
    f = "-f ometiff" if f else ""
    g = "-g" if g else ""
    p = "-p" if p else ""
    with TemporaryDirectory() as tmp_dir:
        output_dir = os.path.join(tmp_dir, "converted.zarr")
        result = runner.invoke(
            cli, f"convert -i {input_dir} -o {output_dir} {f} {g} {p}"
        )
    assert result.exit_code == 0
    assert "Status" in result.output
