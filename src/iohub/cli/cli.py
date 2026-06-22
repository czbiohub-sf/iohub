import logging
import pathlib
from enum import StrEnum
from typing import Annotated

import typer
from typer.core import TyperGroup
from typer.main import get_command

from iohub import __version__, open_ome_zarr
from iohub.cli.parsing import (
    InputPositionDirpaths,
    expand_position_dirpaths,
    install_eat_all_positions,
)
from iohub.convert import TIFFConverter
from iohub.core.ozx import is_ozx_path, pack_ozx, unpack_ozx
from iohub.reader import print_info
from iohub.rename_wells import rename_wells

_logger = logging.getLogger(__name__)

VERSION = __version__

app = typer.Typer(
    add_completion=True,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        # Match Click's ``version_option`` output ("<prog>, version <x>").
        typer.echo(f"iohub, version {VERSION}")
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
            help="Show usage guide to open dataset in Python and full tree for HCS Plates in OME-Zarr",
        ),
    ] = False,
) -> None:
    """View metadata for one or more FILES.

    Supports Micro-Manager TIFF datasets (multi-page OME-TIFF, NDTIFF),
    OME-Zarr directory stores (v0.4 and v0.5), and RFC-9 zipped
    OME-Zarr archives (``.ozx``).
    """
    for file in files:
        typer.echo(f"Reading file:\t {file}")
        print_info(file, verbose=verbose)


class OMEZarrVersion(StrEnum):
    """OME-NGFF versions accepted by ``convert``."""

    v0_4 = "0.4"
    v0_5 = "0.5"


@app.command()
def convert(
    input: Annotated[
        pathlib.Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            resolve_path=True,
            help="Input dataset: Micro-Manager TIFF dir, OME-Zarr dir, or RFC-9 .ozx archive.",
        ),
    ],
    output: Annotated[
        pathlib.Path,
        typer.Option(
            "--output",
            "-o",
            resolve_path=True,
            help="Output path. Suffix selects the operation: .zarr (Zarr dir) or .ozx (zipped).",
        ),
    ],
    grid_layout: Annotated[
        bool,
        typer.Option(
            "--grid-layout",
            "-g",
            help="(TIFF → Zarr only) Arrange FOVs in a row/column grid layout.",
        ),
    ] = False,
    chunks: Annotated[
        str,
        typer.Option(
            "--chunks",
            "-c",
            help="(TIFF → Zarr only) Zarr chunk size: 'XY', 'XYZ', or a tuple. 'XYZ' caps at 500 MB.",
        ),
    ] = "XYZ",
    ome_zarr_version: Annotated[
        OMEZarrVersion | None,
        typer.Option(
            "--ome-zarr-version",
            "-v",
            help="OME-NGFF version. TIFF default: 0.4. Pack: sniffed from source if omitted.",
        ),
    ] = None,
) -> None:
    """Convert datasets between supported formats.

    Routes by suffix: a TIFF directory plus a ``.zarr`` output runs the
    Micro-Manager TIFF → OME-Zarr converter; a ``.zarr`` source plus a
    ``.ozx`` output packs an RFC-9 zip archive; a ``.ozx`` source plus
    a ``.zarr`` output unpacks back to a directory store.
    """
    src = pathlib.Path(input)
    dst = pathlib.Path(output)
    version = ome_zarr_version.value if ome_zarr_version is not None else None
    tiff_only = grid_layout or chunks != "XYZ"

    if is_ozx_path(dst):
        # Pack: 1:1 file copy preserving source chunks. Re-chunking would
        # mean a full read+rewrite — a different operation, out of scope.
        if tiff_only:
            raise typer.BadParameter(
                "--grid-layout and --chunks apply only to TIFF → Zarr conversion. "
                "Pack copies chunks 1:1 from the source."
            )
        out = pack_ozx(src, dst, version=version)
        typer.echo(f"packed: {out}")
        return
    if is_ozx_path(src):
        # Unpack: archive structure dictates everything; no flags apply.
        if tiff_only or version is not None:
            raise typer.BadParameter(
                "--grid-layout, --chunks, and --ome-zarr-version do not apply to .ozx → .zarr unpack."
            )
        out = unpack_ozx(src, dst)
        typer.echo(f"unpacked: {out}")
        return
    # Default: TIFF → OME-Zarr (TIFFConverter sniffs the input format).
    TIFFConverter(
        input_dir=src,
        output_dir=dst,
        grid_layout=grid_layout,
        chunks=chunks,
        version=version or "0.4",
    )()


@app.command(name="set-scale")
def set_scale(
    input_position_dirpaths: InputPositionDirpaths,
    t_scale: Annotated[float | None, typer.Option("--t-scale", "-t", help="New t scale")] = None,
    z_scale: Annotated[float | None, typer.Option("--z-scale", "-z", help="New z scale")] = None,
    y_scale: Annotated[float | None, typer.Option("--y-scale", "-y", help="New y scale")] = None,
    x_scale: Annotated[float | None, typer.Option("--x-scale", "-x", help="New x scale")] = None,
    image: Annotated[
        str | None,
        typer.Option("--image", help="Image name to set scale for. Default is '0'"),
    ] = None,
) -> None:
    """Update scale metadata in OME-Zarr datasets.

    `iohub set-scale -i input.zarr -t 1.0 -z 1.0 -y 0.5 -x 0.5`

    Supports setting a single axis at a time:

    `iohub set-scale -i input.zarr/A/1/0 -z 2.0`
    """
    if image is None:
        image = "0"
    for input_position_dirpath in expand_position_dirpaths(input_position_dirpaths):
        with open_ome_zarr(input_position_dirpath, layout="fov", mode="r+") as dataset:
            for name, value in zip(["t", "z", "y", "x"], [t_scale, z_scale, y_scale, x_scale], strict=False):
                if value is None:
                    continue
                dataset.set_scale(image, name, value)


class PyramidMethod(StrEnum):
    """Downsampling methods accepted by ``compute-pyramid``."""

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
            help="Total number of pyramid levels including level 0 (e.g. 4 = level 0 + 3 extra).",
        ),
    ],
    method: Annotated[
        PyramidMethod,
        typer.Option("--method", "-m", help="The Downsampling method."),
    ] = PyramidMethod.mean,
    dims: Annotated[
        str | None,
        typer.Option(
            "--dims",
            "-d",
            help="Comma-separated axes to downsample (e.g. 'y,x' for YX-only). Defaults to 'z,y,x'.",
        ),
    ] = None,
) -> None:
    """Compute multiscale pyramid levels in place for OME-Zarr positions.

    The level 0 array is preserved; new downsampled levels are appended.

    `iohub compute-pyramid -i input.zarr --levels 4`

    `iohub compute-pyramid -i input.zarr/A/1/0 -l 3 -m median --dims y,x`
    """
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
            help="Path to the input Zarr file.",
        ),
    ],
    csvfile: Annotated[
        pathlib.Path,
        typer.Option(
            "--csv",
            "-c",
            exists=True,
            dir_okay=False,
            help="Path to the CSV file containing old and new well names.",
        ),
    ],
) -> None:
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


# Entry point (pyproject ``[project.scripts]``); also what the tests invoke.
cli = get_command(app)
assert isinstance(cli, TyperGroup)  # multi-command app -> always a group
# Typer can't express a greedy option; make ``-i`` eat space-separated paths.
install_eat_all_positions(cli)

# mkdocs-typer2 ``:name: iohub`` resolves this attribute and labels the docs.
iohub = app
