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
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the input Zarr file.",
)
@click.option(
    "-c",
    "--csv",
    type=click.File("r"),
    required=True,
    help="Path to the CSV file containing well names.",
)
def rename_wells(csvfile, zarrfile):
    """Rename wells based on CSV file

    The CSV file should have two columns: old_well_path and new_well_path.
    """
    names = []

    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if len(row) != 2:
            raise ValueError(
                f"Invalid row format: {row}."
                f"Each row must have two columns."
            )
        names.append([row[0], row[1]])

    plate = open_ome_zarr(zarrfile, mode="a")

    modified = {}
    modified["wells"] = []
    modified["rows"] = []
    modified["columns"] = []

    well_paths = [
        plate.metadata.wells[i].path for i in range(len(plate.metadata.wells))
    ]

    for old_well_path, new_well_path in names:
        if old_well_path not in well_paths:
            raise ValueError(
                f"Old well path '{old_well_path}' not found "
                f"in the plate metadata."
            )
        for well in plate.metadata.wells:
            if well.path == old_well_path and well not in modified["wells"]:
                plate.rename_well(
                    well, old_well_path, new_well_path, modified, False
                )
                modified["wells"].append(well)
