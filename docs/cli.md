# CLI reference

## Selecting input positions

`set-scale` and `compute-pyramid` accept position paths, plate roots, and glob
patterns with `-i` or `--input-position-dirpaths`. A plate root selects every
position in the plate. One `-i` accepts multiple paths, up to the next option.

```bash
# All positions in a plate
iohub compute-pyramid -i input.zarr --levels 4

# Positions matching a glob
iohub compute-pyramid -i 'input.zarr/A/*/*' --levels 4

# Two specific positions
iohub set-scale -i input.zarr/A/1/0 input.zarr/B/2/0 -z 2
```

iohub expands quoted globs. Your shell expands unquoted ones.

To use these options in another Typer CLI, see
[Options with multiple values](greedy-cli-options.md).

## Commands

::: mkdocs-typer2
    :module: iohub.cli.cli
    :name: iohub
    :pretty: true
