from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from natsort import natsorted
from typer.core import TyperGroup, TyperOption

if TYPE_CHECKING:
    from typer._click.core import Context
    from typer._click.parser import _OptionParser

__all__ = [
    "InputPositionDirpaths",
    "OptionEatAll",
    "expand_position_dirpaths",
    "install_eat_all_positions",
]


def expand_position_dirpaths(patterns: list[str]) -> list[Path]:
    """Expand patterns into OME-Zarr FOV position dirpaths.

    Each pattern may be a position path, a plate root (expanded into all of its
    positions), or a glob. Non-directory matches are ignored. Raises
    ``typer.BadParameter`` if nothing matches.
    """
    from iohub.ngff import Plate, open_ome_zarr

    positions: list[Path] = []
    for pattern in patterns:
        # glob.glob (not Path.glob) handles absolute patterns.
        for match in natsorted(glob.glob(pattern)):  # noqa: PTH207
            path = Path(match)
            if not path.is_dir():
                continue
            with open_ome_zarr(path, mode="r") as node:
                if isinstance(node, Plate):
                    positions.extend(path / name for name, _ in node.positions())
                else:
                    positions.append(path)
    if not positions:
        raise typer.BadParameter(f"No positions matched: {list(patterns)}")
    return positions


class OptionEatAll(TyperOption):
    """An option that collects all following tokens, up to the next flag.

    ``-i a b c`` yields ``["a", "b", "c"]``, so unquoted shell globs like
    ``-i input.zarr/*/*/*`` work. Use only on a ``multiple`` (list) option.
    """

    def add_to_parser(self, parser: _OptionParser, ctx: Context) -> None:
        super().add_to_parser(parser, ctx)
        for opt_name in self.opts:
            registered = parser._long_opt.get(opt_name) or parser._short_opt.get(opt_name)
            if registered is None:
                continue
            append_one = registered.process

            def eat_all(value, state, _append=append_one, _prefixes=registered.prefixes):
                _append(value, state)
                # Keep eating until the next option flag (or the end of args).
                while state.rargs and not any(state.rargs[0].startswith(p) for p in _prefixes):
                    _append(state.rargs.pop(0), state)

            registered.process = eat_all
            break


def install_eat_all_positions(group: TyperGroup) -> None:
    """Re-class every ``input_position_dirpaths`` option to ``OptionEatAll``.

    Typer exposes no ``cls=`` hook for options, so the swap happens on the
    already-built command params after ``typer.main.get_command``.
    """
    for command in group.commands.values():
        for param in command.params:
            if isinstance(param, TyperOption) and param.name == "input_position_dirpaths":
                param.__class__ = OptionEatAll


InputPositionDirpaths = Annotated[
    list[str],
    typer.Option(
        "--input-position-dirpaths",
        "-i",
        help=(
            "Input positions. Accepts position paths, a plate root (expanded "
            "into all positions), or a shell glob. One -i may "
            "take several space-separated paths."
        ),
    ),
]
