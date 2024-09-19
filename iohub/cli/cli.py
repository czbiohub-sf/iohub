import csv
import pathlib

import click

from iohub._version import __version__
from iohub.convert import TIFFConverter
from iohub.ngff import open_ome_zarr
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
    type=click.File("r"),
    required=True,
    help="Path to the CSV file containing old and new well names.",
)
def rename_wells(zarrfile, csvfile):
    """Rename wells in an plate.

    >> iohub rename-wells -i plate.zarr -c names.csv

    The CSV file must have two columns with old and new names in the form:
    ```
    A/1,B/2
    A/2,B/2
    ```
    """

    # read and check csv
    name_pair_list = []
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if len(row) != 2:
            raise ValueError(
                f"Invalid row format: {row}."
                f"Each row must have two columns."
            )
        name_pair_list.append([row[0], row[1]])

    # rename each well while catching errors
    with open_ome_zarr(zarrfile, mode="a") as plate:
        for old, new in name_pair_list:
            print(f"Renaming {old} to {new}")
            try:
                plate.rename_well(old, new)
            except ValueError as e:
                print(f"Error renaming {old} to {new}: {e}")