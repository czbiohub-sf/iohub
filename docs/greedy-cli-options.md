# Options with multiple values

`OptionEatAll` lets a Typer list option accept several values after one flag.
It stops at the next option. For example, `-i a b c -v` passes `a`, `b`, and `c`
to `-i` and leaves `-v` as a separate option.

## Reuse iohub's position option

Save this as `positions.py`. The callback keeps the app a command group even
when it has only one command.

```python
import typer
from typer.main import get_command

from iohub.cli import InputPositionDirpaths, expand_position_dirpaths, install_eat_all_positions

app = typer.Typer()


@app.callback()
def main():
    """List OME-Zarr positions."""


@app.command()
def process(input_position_dirpaths: InputPositionDirpaths):
    for position in expand_position_dirpaths(input_position_dirpaths):
        typer.echo(position)


cli = get_command(app)
install_eat_all_positions(cli)

if __name__ == "__main__":
    cli()
```

```bash
python positions.py process -i input.zarr
python positions.py process -i 'input.zarr/A/*/*'
```

`expand_position_dirpaths` expands plate roots and globs, skips file matches,
and raises `typer.BadParameter` if no directories match.
`install_eat_all_positions` updates the group's immediate commands. It looks
for the parameter name `input_position_dirpaths`, so keep that name in your
command.

## Define another option

Typer does not expose an option-class argument. Apply `OptionEatAll` after
`get_command` builds the command. Use it only with list options.

Save this single-command example as `files.py`:

```python
from typing import Annotated

import typer
from typer.core import TyperOption
from typer.main import get_command

from iohub.cli import OptionEatAll

app = typer.Typer()


@app.command()
def show(files: Annotated[list[str], typer.Option("-f", "--files")]):
    typer.echo(files)


cli = get_command(app)
for param in cli.params:
    if isinstance(param, TyperOption) and param.name == "files":
        param.__class__ = OptionEatAll

if __name__ == "__main__":
    cli()
```

```bash
python files.py -f a b c
# ['a', 'b', 'c']
```
