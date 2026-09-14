import logging
import pathlib
from enum import StrEnum
from importlib.metadata import version
from typing import Annotated

import typer
from typer.core import TyperGroup
from typer.main import get_command

# Import dataset readers inside commands so help and version output stay fast.
from iohub.cli.parsing import (
    InputPositionDirpaths,
    expand_position_dirpaths,
    install_eat_all_positions,
)
from iohub.core.types import NGFFVersion

_logger = logging.getLogger(__name__)

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"iohub, version {version('iohub')}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Show the iohub version and exit.",
        ),
    ] = None,
) -> None:
    """iohub: N-dimensional bioimaging I/O"""


@app.command()
def info(
    files: Annotated[
        list[pathlib.Path],
        typer.Argument(
            exists=True,
            resolve_path=True,
            help="One or more datasets to inspect.",
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show Python usage examples and the full OME-Zarr plate tree.",
        ),
    ] = False,
) -> None:
    """Show dataset metadata.

    Accepts Micro-Manager OME-TIFF and NDTIFF datasets, OME-Zarr 0.4 and 0.5
    directories, and zipped OME-Zarr archives with the .ozx extension.
    """
    from iohub.reader import print_info

    for file in files:
        typer.echo(f"Reading file:\t {file}")
        print_info(file, verbose=verbose)


@app.command()
def convert(
    input: Annotated[
        pathlib.Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            resolve_path=True,
            help="Micro-Manager TIFF directory, OME-Zarr directory, or .ozx archive.",
        ),
    ],
    output: Annotated[
        pathlib.Path,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Output OME-Zarr directory or .ozx archive.",
        ),
    ],
    grid_layout: Annotated[
        bool,
        typer.Option(
            "--grid-layout",
            "-g",
            help="Arrange fields of view in a row/column grid. TIFF conversion only.",
        ),
    ] = False,
    chunks: Annotated[
        str,
        typer.Option(
            "--chunks",
            "-c",
            help="Chunk shape as 'XY', 'XYZ', or a tuple. 'XYZ' caps chunks at 500 MB. TIFF conversion only.",
        ),
    ] = "XYZ",
    ome_zarr_version: Annotated[
        NGFFVersion | None,
        typer.Option(
            "--ome-zarr-version",
            "-v",
            help="OME-NGFF version. Defaults to 0.4 for TIFF conversion or the source version when packing .ozx.",
        ),
    ] = None,
    num_workers: Annotated[
        int | None,
        typer.Option(
            "--num-workers",
            "-n",
            min=1,
            help="Threads copying pixels concurrently. Defaults to 4; NDTiff scales to about 16. TIFF conversion only.",
        ),
    ] = None,
) -> None:
    """Convert Micro-Manager TIFF to OME-Zarr, or pack and unpack .ozx archives.

    An .ozx output packs an OME-Zarr directory without changing its chunks.
    An .ozx input unpacks to a directory. Other inputs use the TIFF converter.

    --grid-layout, --chunks, and --num-workers apply only to TIFF conversion.
    --ome-zarr-version does not apply when unpacking.
    """
    from iohub.convert import DEFAULT_NUM_WORKERS, TIFFConverter
    from iohub.core.ozx import is_ozx_path, pack_ozx, unpack_ozx

    src = pathlib.Path(input)
    dst = pathlib.Path(output)
    tiff_only = grid_layout or chunks != "XYZ" or num_workers is not None

    if is_ozx_path(dst):
        # Packing preserves source chunks.
        if tiff_only:
            raise typer.BadParameter(
                "--grid-layout, --chunks, and --num-workers apply only to TIFF → Zarr conversion. "
                "Pack copies chunks 1:1 from the source."
            )
        out = pack_ozx(src, dst, version=ome_zarr_version)
        typer.echo(f"packed: {out}")
        return
    if is_ozx_path(src):
        if tiff_only or ome_zarr_version is not None:
            raise typer.BadParameter(
                "--grid-layout, --chunks, --num-workers, and --ome-zarr-version do not apply to .ozx → .zarr unpack."
            )
        out = unpack_ozx(src, dst)
        typer.echo(f"unpacked: {out}")
        return
    # TIFFConverter detects the TIFF format.
    TIFFConverter(
        input_dir=src,
        output_dir=dst,
        grid_layout=grid_layout,
        chunks=chunks,
        version=ome_zarr_version or "0.4",
        num_workers=num_workers if num_workers is not None else DEFAULT_NUM_WORKERS,
    )()


@app.command(name="set-scale")
def set_scale(
    input_position_dirpaths: InputPositionDirpaths,
    t_scale: Annotated[float | None, typer.Option("--t-scale", "-t", help="New time-axis scale.")] = None,
    z_scale: Annotated[float | None, typer.Option("--z-scale", "-z", help="New z-axis scale.")] = None,
    y_scale: Annotated[float | None, typer.Option("--y-scale", "-y", help="New y-axis scale.")] = None,
    x_scale: Annotated[float | None, typer.Option("--x-scale", "-x", help="New x-axis scale.")] = None,
    image: Annotated[
        str | None,
        typer.Option("--image", help="Image to update. Defaults to '0'."),
    ] = None,
) -> None:
    """Update axis scale metadata in OME-Zarr positions.

    Omitted axes keep their current scales. Image data is unchanged.

    Example:

    iohub set-scale -i input.zarr/A/1/0 -z 2.0
    """
    from iohub import open_ome_zarr

    if image is None:
        image = "0"
    for input_position_dirpath in expand_position_dirpaths(input_position_dirpaths):
        with open_ome_zarr(input_position_dirpath, layout="fov", mode="r+") as dataset:
            for name, value in zip(["t", "z", "y", "x"], [t_scale, z_scale, y_scale, x_scale], strict=False):
                if value is None:
                    continue
                dataset.set_scale(image, name, value)


class PyramidMethod(StrEnum):
    mean = "mean"
    median = "median"
    mode = "mode"
    min = "min"
    max = "max"
    stride = "stride"


_PYRAMID_DIM_CHOICES = ("t", "z", "y", "x")


def _parse_dims(value: str | None) -> set[str] | None:
    if value is None:
        return None
    tokens = [t.strip().lower() for t in value.split(",") if t.strip()]
    if not tokens:
        return None
    invalid = [t for t in tokens if t not in _PYRAMID_DIM_CHOICES]
    if invalid:
        invalid_dims = ", ".join(dict.fromkeys(invalid))
        valid_dims = ", ".join(_PYRAMID_DIM_CHOICES)
        raise typer.BadParameter(f"Unknown dim(s): {invalid_dims}. Valid choices: {valid_dims}.")
    return set(tokens)


@app.command(name="compute-pyramid")
def compute_pyramid(
    input_position_dirpaths: InputPositionDirpaths,
    levels: Annotated[
        int,
        typer.Option(
            "--levels",
            "-l",
            min=2,
            help="Total number of levels, including the original level 0.",
        ),
    ],
    method: Annotated[
        PyramidMethod,
        typer.Option("--method", "-m", help="Downsampling method."),
    ] = PyramidMethod.mean,
    dims: Annotated[
        str | None,
        typer.Option(
            "--dims",
            "-d",
            help="Comma-separated axes for a new pyramid, such as 'y,x'. Defaults to 'z,y,x'.",
        ),
    ] = None,
) -> None:
    """Create or recompute an OME-Zarr pyramid in place.

    Level 0 is unchanged. Existing pyramids must have the requested number
    of levels. Recomputing overwrites the downsampled levels.

    Example:

    iohub compute-pyramid -i input.zarr -l 3 -m median --dims y,x
    """
    from iohub import open_ome_zarr

    parsed_dims = _parse_dims(dims)
    for input_position_dirpath in expand_position_dirpaths(input_position_dirpaths):
        _logger.info(f"Computing pyramid for {input_position_dirpath}")
        with open_ome_zarr(input_position_dirpath, layout="fov", mode="r+") as dataset:
            dataset.compute_pyramid(levels=levels, method=method.value, dims=parsed_dims)


@app.command(name="rename-wells")
def rename_wells_command(
    zarrfile: Annotated[
        pathlib.Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            help="OME-Zarr plate directory.",
        ),
    ],
    csvfile: Annotated[
        pathlib.Path,
        typer.Option(
            "--csv",
            "-c",
            exists=True,
            dir_okay=False,
            help="CSV file mapping old well names to new names.",
        ),
    ],
) -> None:
    """Rename wells in an OME-Zarr plate.

    Use a CSV with two columns, old name then new name, and no header:

        A/1,B/1
        A/2,B/2

    Example:

    iohub rename-wells -i plate.zarr -c names.csv
    """
    from iohub.rename_wells import rename_wells

    rename_wells(zarrfile, csvfile)


# Installed CLI entry point.
cli = get_command(app)
assert isinstance(cli, TyperGroup)
# Let position options accept multiple paths after one -i.
install_eat_all_positions(cli)

# App name used by mkdocs-typer2 to generate the CLI reference.
iohub = app
