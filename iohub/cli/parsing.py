from pathlib import Path
from typing import Callable, List

import click
from natsort import natsorted

from iohub.ngff import Plate, open_ome_zarr


def _validate_and_process_paths(
    ctx: click.Context, opt: click.Option, value: List[str]
) -> list[Path]:
    # Sort and validate the input paths,
    # expanding plates into lists of positions
    input_paths = [Path(path) for path in natsorted(value)]
    for path in input_paths:
        with open_ome_zarr(path, mode="r") as dataset:
            if isinstance(dataset, Plate):
                plate_path = input_paths.pop()
                for position in dataset.positions():
                    input_paths.append(plate_path / position[0])

    return input_paths


def input_position_dirpaths() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--input-position-dirpaths",
            "-i",
            cls=OptionEatAll,
            type=tuple,
            required=True,
            callback=_validate_and_process_paths,
            help=(
                "List of paths to input positions, "
                "each with the same TCZYX shape. "
                "Supports wildcards e.g. 'input.zarr/*/*/*'."
            ),
        )(f)

    return decorator


# Copied directly from https://stackoverflow.com/a/48394004
# Enables `-i ./input.zarr/*/*/*`
class OptionEatAll(click.Option):
    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop("save_other_options", True)
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(
                name
            )
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval
