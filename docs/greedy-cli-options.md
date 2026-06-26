# Greedy CLI options (`OptionEatAll`)

A Typer option normally takes one value, so `-i a b c` fails. iohub makes `-i`
greedy: one flag eats every path that follows, so `-i input.zarr/*/*/*` works
unquoted. These building blocks are exported for your own Typer CLI to reuse.

```python
from iohub.cli import (
    InputPositionDirpaths,      # the shared -i option (a list[str])
    expand_position_dirpaths,   # expand paths/plate roots/globs -> position dirpaths
    install_eat_all_positions,  # make the -i option greedy
    OptionEatAll,               # the greedy-option class itself
)
```
## Reuse iohub's position input

The common case: accept the same `-i` as iohub, with plate/glob expansion and
validation included.

```python
import typer
from typer.core import TyperGroup
from typer.main import get_command
from iohub.cli import InputPositionDirpaths, expand_position_dirpaths, install_eat_all_positions

app = typer.Typer()


@app.command()
def process(input_position_dirpaths: InputPositionDirpaths):
    for position in expand_position_dirpaths(input_position_dirpaths):
        ...  # `position` is a Path to an OME-Zarr FOV


cli = get_command(app)
assert isinstance(cli, TyperGroup)
install_eat_all_positions(cli)  # make every -i greedy

if __name__ == "__main__":
    cli()
```

```bash
mytool process -i input.zarr/*/*/*   # unquoted glob, every position
mytool process -i input.zarr         # plate root -> all positions
```

`install_eat_all_positions` finds the option by the parameter name
`input_position_dirpaths`, so use the `InputPositionDirpaths` type and it works.

## Make your own greedy option

`OptionEatAll` is the reusable primitive. Typer offers no `cls=` hook, so swap the
class onto the built option after `get_command`:

```python
from typing import Annotated
import typer
from typer.core import TyperOption
from typer.main import get_command
from iohub.cli import OptionEatAll

app = typer.Typer()


@app.command()
def cat(files: Annotated[list[str], typer.Option("-f", "--files")]):
    typer.echo(files)


cli = get_command(app)
for command in cli.commands.values():
    for param in command.params:
        if isinstance(param, TyperOption) and "-f" in param.opts:
            param.__class__ = OptionEatAll

cli()
```

```bash
mytool cat -f a b c   # -> ['a', 'b', 'c']
```
