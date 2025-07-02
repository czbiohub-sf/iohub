import pathlib

import click

from iohub import open_ome_zarr
from iohub._version import __version__
from iohub.cli.parsing import input_position_dirpaths
from iohub.convert import TIFFConverter
from iohub.reader import print_info
from iohub.rename_wells import rename_wells

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
@click.option(
    "--image",
    required=False,
    help="Image name to set scale for. Default is '0'",
)
def set_scale(
    input_position_dirpaths,
    t_scale=None,
    z_scale=None,
    y_scale=None,
    x_scale=None,
    image=None,
):
    """Update scale metadata in OME-Zarr datasets.

    >> iohub set-scale -i input.zarr/*/*/* -t 1.0 -z 1.0 -y 0.5 -x 0.5

    Supports setting a single axis at a time:

    >> iohub set-scale -i input.zarr/*/*/* -z 2.0
    """
    if image is None:
        image = "0"
    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(
            input_position_dirpath, layout="fov", mode="r+"
        ) as dataset:
            for name, value in zip(
                ["t", "z", "y", "x"], [t_scale, z_scale, y_scale, x_scale]
            ):
                if value is None:
                    continue
                dataset.set_scale(image, name, value)


@cli.command(name="rename-wells")
@click.help_option("-h", "--help")
@click.option(
    "-i",
    "--input",
    "zarrfile",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    help="Path to the input Zarr file.",
)
@click.option(
    "-c",
    "--csv",
    "csvfile",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the CSV file containing old and new well names.",
)
def rename_wells_command(zarrfile, csvfile):
    """Rename wells in an plate.

    >> iohub rename-wells -i plate.zarr -c names.csv

    The CSV file must have two columns with old and new names in the form:
    ```
    A/1,B/2
    A/2,B/2
    ```
    """
    rename_wells(zarrfile, csvfile)
