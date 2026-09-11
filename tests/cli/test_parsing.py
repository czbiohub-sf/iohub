import numpy as np
import pytest
import typer
from click.testing import CliRunner
from typer.main import get_command

from iohub import open_ome_zarr
from iohub.cli.parsing import (
    InputPositionDirpaths,
    expand_position_dirpaths,
    install_eat_all_positions,
)


def _make_plate(tmpdir):
    plate_path = tmpdir / "dataset.zarr"
    position_list = [("A", "1", "0"), ("B", "2", "0"), ("X", "4", "1")]
    with open_ome_zarr(plate_path, mode="w", layout="hcs", channel_names=["1", "2"]) as dataset:
        for position in position_list:
            pos = dataset.create_position(*position)
            pos.create_zeros("0", shape=(1, 1, 1, 1, 1), dtype=np.uint8)
    return plate_path, position_list


def test_expand_plate_root(tmpdir):
    """A plate root expands into all of its positions."""
    plate_path, position_list = _make_plate(tmpdir)
    processed = expand_position_dirpaths([str(plate_path)])
    assert len(processed) == len(position_list)
    for i, position in enumerate(position_list):
        assert processed[i].parts[-3:] == position


def test_expand_single_position(tmpdir):
    plate_path, _ = _make_plate(tmpdir)
    processed = expand_position_dirpaths([str(plate_path / "A" / "1" / "0")])
    assert len(processed) == 1


def test_expand_multiple_patterns(tmpdir):
    """Repeated ``-i`` values are concatenated."""
    plate_path, _ = _make_plate(tmpdir)
    processed = expand_position_dirpaths([str(plate_path / "A" / "1" / "0"), str(plate_path / "B" / "2" / "0")])
    assert len(processed) == 2


def test_expand_glob(tmpdir):
    """A glob pattern matches positions and ignores non-directories."""
    plate_path, position_list = _make_plate(tmpdir)
    # Drop a stray file alongside the positions to confirm it is filtered out.
    (plate_path / "A" / "1" / "zarr.json").write_text("{}", encoding="utf-8")
    processed = expand_position_dirpaths([str(plate_path / "*" / "*" / "*")])
    assert len(processed) == len(position_list)
    assert all(p.is_dir() for p in processed)


def test_expand_no_match_raises(tmpdir):
    with pytest.raises(typer.BadParameter):
        expand_position_dirpaths([str(tmpdir / "does_not_exist" / "*")])


def _eat_all_app():
    """A tiny Typer app whose -i uses the shared InputPositionDirpaths type."""
    app = typer.Typer()

    @app.command()
    def go(
        input_position_dirpaths: InputPositionDirpaths,
        x: bool = typer.Option(False, "-x"),
    ):
        typer.echo(f"i={list(input_position_dirpaths)} x={x}")

    @app.command()
    def other():  # second command forces a group
        ...

    cli = get_command(app)
    install_eat_all_positions(cli)
    return cli


def test_option_eat_all_consumes_multiple_tokens():
    """Guard against Typer-internals drift: one -i must eat several tokens.

    If a Typer upgrade changes the vendored parser, OptionEatAll breaks and
    this test fails loudly (rather than silently dropping arguments).
    """
    runner = CliRunner()
    result = runner.invoke(_eat_all_app(), ["go", "-i", "a", "b", "c"])
    assert result.exit_code == 0, result.output
    assert "i=['a', 'b', 'c']" in result.output
    assert "Error" not in result.output


def test_option_eat_all_stops_at_next_flag():
    """Eating stops at the next option, so trailing flags still parse."""
    runner = CliRunner()
    result = runner.invoke(_eat_all_app(), ["go", "-i", "a", "b", "-x"])
    assert result.exit_code == 0, result.output
    assert "i=['a', 'b'] x=True" in result.output


def test_option_eat_all_single_token():
    runner = CliRunner()
    result = runner.invoke(_eat_all_app(), ["go", "-i", "a"])
    assert result.exit_code == 0, result.output
    assert "i=['a']" in result.output
