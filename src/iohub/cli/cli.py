import pathlib

import click

from iohub import __version__, open_ome_zarr
from iohub.cli.parsing import input_position_dirpaths
from iohub.convert import TIFFConverter
from iohub.core.ozx import is_ozx_path, pack_ozx, unpack_ozx
from iohub.reader import print_info
from iohub.rename_wells import rename_wells

VERSION = __version__

_DATASET_PATH = click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path)
# Like _DATASET_PATH but also accepts files (e.g. ``.ozx`` archives).
_OME_ZARR_PATH = click.Path(exists=True, resolve_path=True, path_type=pathlib.Path)


@click.group()
@click.help_option("-h", "--help")
@click.version_option(version=VERSION)
def cli():
    """iohub: N-dimensional bioimaging I/O"""


@cli.command()
@click.help_option("-h", "--help")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=_OME_ZARR_PATH,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show usage guide to open dataset in Python and full tree for HCS Plates in OME-Zarr",
)
def info(files, verbose):
    """View metadata for one or more FILES.

    Supports Micro-Manager TIFF datasets (multi-page OME-TIFF, NDTIFF),
    OME-Zarr directory stores (v0.4 and v0.5), and RFC-9 zipped
    OME-Zarr archives (``.ozx``).
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
    type=_OME_ZARR_PATH,
    help="Input dataset: Micro-Manager TIFF dir, OME-Zarr dir, or RFC-9 .ozx archive.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False, resolve_path=True, path_type=pathlib.Path),
    help="Output path. Suffix selects the operation: .zarr (Zarr dir) or .ozx (zipped).",
)
@click.option(
    "--grid-layout",
    "-g",
    required=False,
    is_flag=True,
    help="(TIFF → Zarr only) Arrange FOVs in a row/column grid layout.",
)
@click.option(
    "--chunks",
    "-c",
    required=False,
    default="XYZ",
    help="(TIFF → Zarr only) Zarr chunk size: 'XY', 'XYZ', or a tuple. 'XYZ' caps at 500 MB.",
)
@click.option(
    "--ome-zarr-version",
    "-v",
    required=False,
    default=None,
    type=click.Choice(["0.4", "0.5"]),
    help="OME-NGFF version. TIFF default: 0.4. Pack: sniffed from source if omitted.",
)
def convert(input, output, grid_layout, chunks, ome_zarr_version):
    """Convert datasets between supported formats.

    Routes by suffix: a TIFF directory plus a ``.zarr`` output runs the
    Micro-Manager TIFF → OME-Zarr converter; a ``.zarr`` source plus a
    ``.ozx`` output packs an RFC-9 zip archive; a ``.ozx`` source plus
    a ``.zarr`` output unpacks back to a directory store.
    """
    src = pathlib.Path(input)
    dst = pathlib.Path(output)
    tiff_only = grid_layout or chunks != "XYZ"

    if is_ozx_path(dst):
        # Pack: 1:1 file copy preserving source chunks. Re-chunking would
        # mean a full read+rewrite — a different operation, out of scope.
        if tiff_only:
            raise click.BadParameter(
                "--grid-layout and --chunks apply only to TIFF → Zarr conversion. "
                "Pack copies chunks 1:1 from the source."
            )
        out = pack_ozx(src, dst, version=ome_zarr_version)
        click.echo(f"packed: {out}")
        return
    if is_ozx_path(src):
        # Unpack: archive structure dictates everything; no flags apply.
        if tiff_only or ome_zarr_version is not None:
            raise click.BadParameter(
                "--grid-layout, --chunks, and --ome-zarr-version do not apply to .ozx → .zarr unpack."
            )
        out = unpack_ozx(src, dst)
        click.echo(f"unpacked: {out}")
        return
    # Default: TIFF → OME-Zarr (TIFFConverter sniffs the input format).
    TIFFConverter(
        input_dir=src,
        output_dir=dst,
        grid_layout=grid_layout,
        chunks=chunks,
        version=ome_zarr_version or "0.4",
    )()


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

    ```
    iohub set-scale -i input.zarr/*/*/* -t 1.0 -z 1.0 -y 0.5 -x 0.5
    ```

    Supports setting a single axis at a time:

    ```
    iohub set-scale -i input.zarr/*/*/* -z 2.0
    ```
    """
    if image is None:
        image = "0"
    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(input_position_dirpath, layout="fov", mode="r+") as dataset:
            for name, value in zip(["t", "z", "y", "x"], [t_scale, z_scale, y_scale, x_scale], strict=False):
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

    ```
    iohub rename-wells -i plate.zarr -c names.csv
    ```

    The CSV file must have two columns with old and new names in the form:
    ```
    A / 1, B / 2
    A / 2, B / 2
    ```
    """
    rename_wells(zarrfile, csvfile)
