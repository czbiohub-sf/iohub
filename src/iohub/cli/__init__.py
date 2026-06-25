"""iohub command-line interface.

The ``OptionEatAll`` building blocks are re-exported here for downstream Typer
CLIs that want the same greedy ``-i`` behavior.
"""

from iohub.cli.parsing import (
    InputPositionDirpaths,
    OptionEatAll,
    expand_position_dirpaths,
    install_eat_all_positions,
)

__all__ = [
    "InputPositionDirpaths",
    "OptionEatAll",
    "expand_position_dirpaths",
    "install_eat_all_positions",
]
