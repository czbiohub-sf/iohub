from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from natsort import natsorted
from typer.core import TyperGroup, TyperOption

from iohub.ngff import Plate, open_ome_zarr

if TYPE_CHECKING:
    # Typer's vendored Click parser types. Imported only for type checking so
    # the module never couples to these private names at import time (a Typer
    # upgrade that renames them fails the guard test, not every iohub import).
    from typer._click.core import Context
    from typer._click.parser import _OptionParser


def expand_position_dirpaths(patterns: list[str]) -> list[Path]:
    """Expand input patterns into OME-Zarr FOV position dirpaths.

    Each pattern may be one of:

    - a single position path (used as-is),
    - a plate root, which is expanded into all of its positions, or
    - a glob pattern such as ``input.zarr/*/*/*``.

    Globs are expanded whether the shell expanded them first (``OptionEatAll``
    collects the resulting tokens) or they reach Python intact (e.g. quoted, or
    on a shell that does not glob, such as on Windows). Non-directory matches
    (such as ``zarr.json``) are ignored.

    Raises ``typer.BadParameter`` if no pattern matches any directory.
    """
    positions: list[Path] = []
    for pattern in patterns:
        # glob.glob handles arbitrary absolute/relative user patterns (Path.glob
        # can't take an absolute pattern). Non-directory matches (e.g. a
        # well-level zarr.json swept up by ``*/*/*``) are skipped.
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
    """A Typer option that greedily consumes the tokens that follow it.

    ``-i a b c`` is collected as ``["a", "b", "c"]`` (consumption stops at the
    next option flag), so an *unquoted* shell glob like ``-i input.zarr/*/*/*``
    works without quoting. Typer/Click options otherwise take only a fixed
    number of values, so we hook the parser ourselves — the Typer-era version
    of the classic Click ``OptionEatAll`` recipe.

    Use only on a ``multiple`` (list) option: each eaten token is forwarded to
    the option's append action, i.e. ``-i a b c`` behaves like ``-i a -i b -i c``.

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
    already-built command params after ``typer.main.get_command``. Idempotent.
    """
    for command in group.commands.values():
        for param in command.params:
            if isinstance(param, TyperOption) and param.name == "input_position_dirpaths":
                param.__class__ = OptionEatAll


# Shared ``-i`` option: a position path, plate root, or glob. ``OptionEatAll``
# (installed in cli.py) lets one ``-i`` consume several space-separated paths.
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
