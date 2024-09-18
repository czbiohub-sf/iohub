import pathlib

import click

from iohub import open_ome_zarr
from iohub._version import __version__
from iohub.cli.parsing import input_position_dirpaths
from iohub.convert import TIFFConverter
from iohub.reader import print_info

VERSION = __version__

_DATASET_PATH = click.Path(
    exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path
)


@click.group()
@click.help_option("-h", "--help")
@click.version_option(version=VERSION)
def cli():
    """\u001b[34;1m iohub: N-dimensional bioimaging I/O \u001b[0m"""


@cli.command()
@click.help_option("-h", "--help")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=_DATASET_PATH,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show usage guide to open dataset in Python "
    "and full tree for HCS Plates in OME-Zarr",
)
def info(files, verbose):
    """View basic metadata of a list of FILES.

    Supported formats are Micro-Manager-acquired TIFF datasets
    (single-page TIFF, multi-page OME-TIFF, NDTIFF)
    and OME-Zarr (v0.1 linear HCS layout and all v0.4 layouts).
    """
    for file in files:
        click.echo(f"Reading file:\t {file}")
        print_info(file, verbose=verbose)


@cli.command()
@click.help_option("-h", "--help")
@click.option(
    "--input",
    "-i",
    required=True,
    type=_DATASET_PATH,
    help="Input Micro-Manager TIFF dataset directory",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False, resolve_path=True),
    help="Output zarr store (/**/converted.zarr)",
)
@click.option(
    "--grid-layout",
    "-g",
    required=False,
    is_flag=True,
    help="Arrange FOVs in a row/column grid layout for tiled acquisition",
)
@click.option(
    "--chunks",
    "-c",
    required=False,
    default="XYZ",
    help="Zarr chunk size given as 'XY', 'XYZ', or a tuple of chunk "
    "dimensions. If 'XYZ', chunk size will be limited to 500 MB.",
)
def convert(input, output, grid_layout, chunks):
    """Converts Micro-Manager TIFF datasets to OME-Zarr"""
    converter = TIFFConverter(
        input_dir=input,
        output_dir=output,
        grid_layout=grid_layout,
        chunks=chunks,
    )
    converter()


@cli.command()
@click.help_option("-h", "--help")
@input_position_dirpaths()
@click.option(
    "--t-scale",
    "-t",
    required=False,
    type=float,
    help="New t scale",
)
@click.option(
    "--z-scale",
    "-z",
    required=False,
    type=float,
    help="New z scale",
)
@click.option(
    "--y-scale",
    "-y",
    required=False,
    type=float,
    help="New y scale",
)
@click.option(
    "--x-scale",
    "-x",
    required=False,
    type=float,
    help="New x scale",
)
def set_scale(
    input_position_dirpaths,
    t_scale=None,
    z_scale=None,
    y_scale=None,
    x_scale=None,
):
    """Update scale metadata in OME-Zarr datasets.

    >> iohub set-scale -i input.zarr/*/*/* -t 1.0 -z 1.0 -y 0.5 -x 0.5

    Supports setting a single axis at a time:

    >> iohub set-scale -i input.zarr/*/*/* -z 2.0
    """
    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(
            input_position_dirpath, layout="fov", mode="a"
        ) as dataset:
            for name, value in zip(
                ["T", "Z", "Y", "X"], [t_scale, z_scale, y_scale, x_scale]
            ):
                if value is None:
                    continue
                old_value = dataset.scale[dataset.get_axis_index(name)]
                print(
                    f"Updating {input_position_dirpath} {name} scale from "
                    f"{old_value} to {value}."
                )
                dataset.set_scale("0", name, value)
