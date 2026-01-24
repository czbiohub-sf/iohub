from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import natsort
import tifffile as tiff
import zarr
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from iohub._deprecated.reader_base import ReaderBase
from iohub._deprecated.singlepagetiff import MicromanagerSequenceReader
from iohub._deprecated.zarrfile import ZarrReader
from iohub.fov import BaseFOVMapping
from iohub.mmstack import MMStack
from iohub.ndtiff import NDTiffDataset
from iohub.ngff.models import SpaceAxisMeta
from iohub.ngff.nodes import (
    NGFFNode,
    Plate,
    Position,
    _is_remote_url,
    open_ome_zarr,
)

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath

_logger = logging.getLogger(__name__)


def _find_ngff_version_in_zarr_group(group: zarr.Group) -> str | None:
    for key in ["plate", "well", "ome"]:
        if key in group.attrs:
            if v := group.attrs[key].get("version"):
                return v
    if "multiscales" in group.attrs:
        for ms in group.attrs["multiscales"]:
            if v := ms.get("version"):
                return v
    return None


def _check_zarr_data_type(src: Path):
    try:
        root = zarr.open(src, mode="r")
        if version := _find_ngff_version_in_zarr_group(root):
            return version
        else:
            for _, row in root.groups():
                for _, well in row.groups():
                    if version := _find_ngff_version_in_zarr_group(well):
                        return version
    except Exception:
        return False
    return "unknown"


def _check_single_page_tiff(src: Path):
    if src.is_file():
        src = src.parent
    files = src.glob("*.tif")
    try:
        next(files)
    except StopIteration:
        sub_dirs = _get_sub_dirs(src)
        if sub_dirs:
            files = (src / sub_dirs[0]).glob("*.tif")
            try:
                with tiff.TiffFile(next(files)) as tf:
                    if (
                        len(tf.pages) == 1
                    ):  # and tf.pages[0].is_multipage is False:
                        return True
            except StopIteration:
                pass
    return False


def _check_multipage_tiff(src: Path):
    if src.is_file():
        src = src.parent
    try:
        file = next(src.glob("*.tif"))
    except StopIteration:
        return False
    with tiff.TiffFile(file) as tf:
        if len(tf.pages) > 1:
            return True
        elif tf.is_multipage is False and tf.is_ome is True:
            return True
    return False


def _check_ndtiff(src: Path):
    # select parent directory if a .tif file is selected
    if src.is_file() and "tif" in src.suffixes:
        if _check_ndtiff(src.parent) or _check_ndtiff(src.parent.parent):
            return True
    # ndtiff v2
    if Path(src, "Full resolution", "NDTiff.index").exists():
        return True
    # ndtiff v3
    elif Path(src, "NDTiff.index").exists():
        return True
    return False


def _get_sub_dirs(directory: Path) -> list[str]:
    """ """
    sub_dir_name = [
        subdir.name for subdir in directory.iterdir() if subdir.is_dir()
    ]
    #    assert subDirName, 'No sub directories found'
    return natsort.natsorted(sub_dir_name)


def _infer_format(path: Path):
    extra_info = None
    if ngff_version := _check_zarr_data_type(path):
        data_type = "omezarr"
        extra_info = ngff_version
    elif _check_ndtiff(path):
        data_type = "ndtiff"
    elif _check_multipage_tiff(path):
        data_type = "ometiff"
    elif _check_single_page_tiff(path):
        data_type = "singlepagetiff"
    else:
        raise RuntimeError(
            "Failed to infer data type: "
            f"No compatible data found under {path}."
        )
    return (data_type, extra_info)


def read_images(
    path: StrOrBytesPath,
    data_type: Literal[
        "singlepagetiff", "ometiff", "ndtiff", "omezarr"
    ] = None,
) -> ReaderBase | BaseFOVMapping:
    """Read image arrays and metadata from a Micro-Manager dataset.
    Supported formats are Micro-Manager-acquired TIFF datasets
    (single-page TIFF, multi-page OME-TIFF, NDTIFF),
    and converted OME-Zarr (v0.1/v0.4 HCS layout assuming a linear structure).

    Parameters
    ----------
    path : StrOrBytesPath
        File path, directory path to ome-tiff series, or Zarr root path
    data_type :
    Literal["singlepagetiff", "ometiff", "ndtiff", "omezarr"], optional
        Dataset format, by default None

    Returns
    -------
    ReaderBase | BaseFOVMapping
        Image collection object for the dataset
    """
    path = Path(path).resolve()
    # try to guess data type
    extra_info = None
    if not data_type:
        data_type, extra_info = _infer_format(path)
    _logger.debug(f"Detected data type: {data_type} {extra_info}")
    # identify data structure type
    if data_type == "ometiff":
        return MMStack(path)
    elif data_type == "singlepagetiff":
        return MicromanagerSequenceReader(path)
    elif data_type == "omezarr":
        if extra_info is None:
            _, extra_info = _infer_format(path)
        if extra_info == "0.4":
            # `warnings` instead of `logging` since this can be avoided
            warnings.warn(
                UserWarning(
                    "For NGFF v0.4 datasets, `iohub.open_ome_zarr()` "
                    "is preferred over `iohub.read_images()`. "
                    "Note that `open_ome_zarr()` will return "
                    "an NGFFNode object instead of a ReaderBase instance."
                )
            )
        elif extra_info != "0.1":
            raise ValueError(f"NGFF version {extra_info} is not supported.")
        return ZarrReader(path, version=extra_info)
    elif data_type == "ndtiff":
        return NDTiffDataset(path)
    else:
        raise ValueError(f"Reader of type {data_type} is not implemented")


def _create_summary_table() -> Table:
    """Create a two-column key-value table for summary display."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("Property", style="cyan", no_wrap=True, width=22)
    table.add_column("Value", overflow="fold")
    return table


def _format_axes(axes: list) -> str:
    """Format axes as comma-separated string with types."""
    return ", ".join(f"{a.name} ({a.type})" for a in axes)


def _format_channels(names: list) -> str:
    """Format channel names as comma-separated string."""
    return ", ".join(str(n) for n in names)


_UNIT_ABBREV = {
    "micrometer": "µm",
    "nanometer": "nm",
    "millimeter": "mm",
    "meter": "m",
}


def _get_space_unit(axes: list) -> str | None:
    """Get the unit from the first space axis, abbreviated for display."""
    for ax in axes:
        if isinstance(ax, SpaceAxisMeta):
            return _UNIT_ABBREV.get(ax.unit, ax.unit)
    return None


def print_info(path: StrOrBytesPath | str, verbose=False):
    """Print summary information for a dataset.

    Parameters
    ----------
    path : StrOrBytesPath | str
        Path to the dataset (local path or remote URL)
    verbose : bool, optional
        Show usage guide to open dataset in Python
        and full tree for HCS Plates in OME-Zarr,
        by default False
    """
    is_remote = _is_remote_url(path)
    if not is_remote:
        path = Path(path).resolve()

    try:
        if is_remote:
            # Remote URLs only support OME-Zarr
            reader = open_ome_zarr(path, mode="r")
            fmt, extra_info = "omezarr", reader.zattrs.get("ome", {}).get(
                "version", reader.zattrs.get("plate", {}).get("version")
            )
        else:
            fmt, extra_info = _infer_format(path)
            if fmt == "omezarr" and extra_info in ("0.4", "0.5"):
                reader = open_ome_zarr(path, mode="r", version=extra_info)
            else:
                reader = read_images(path, data_type=fmt)
    except (ValueError, RuntimeError):
        print("Error: No compatible dataset is found.")
        return
    fmt_msg = f"Format:\t\t\t {fmt}"
    if extra_info:
        if extra_info.startswith("0."):
            fmt_msg += " v" + extra_info
    sum_msg = "\n=== Summary ==="
    ch_msg = f"Channel names:\t\t {reader.channel_names}"
    code_msg = "\nThis datset can be opened with iohub in Python code:\n"
    msgs = []
    if isinstance(reader, BaseFOVMapping):
        _, first_fov = next(iter(reader))
        shape_msg = ", ".join(
            [
                f"{a}={s}"
                for s, a in zip(first_fov.shape, ("T", "C", "Z", "Y", "X"))
            ]
        )
        msgs.extend(
            [
                sum_msg,
                fmt_msg,
                f"FOVs:\t\t\t {len(reader)}",
                f"FOV shape:\t\t {shape_msg}",
                ch_msg,
                f"(Z, Y, X) scale (um):\t {first_fov.zyx_scale}",
            ]
        )
        if verbose:
            msgs.extend(
                [
                    code_msg,
                    ">>> from iohub import read_images",
                    f">>> reader = read_images('{path}')",
                ]
            )
        print(str.join("\n", msgs))
    elif isinstance(reader, NGFFNode):
        console = Console()

        # Print tree first (if verbose or Position)
        if verbose or isinstance(reader, Position):
            console.print("[bold]Zarr hierarchy:[/bold]")
            reader.print_tree()
            console.print()

        # Build summary table
        table = _create_summary_table()
        fmt_str = f"{fmt} v{extra_info}" if extra_info else fmt
        table.add_row("Format", fmt_str)
        table.add_row("Axes", _format_axes(reader.axes))
        table.add_row("Channel names", _format_channels(reader.channel_names))

        if isinstance(reader, Plate):
            meta = reader.metadata
            table.add_row("Row names", ", ".join(r.name for r in meta.rows))
            col_names = ", ".join(c.name for c in meta.columns)
            table.add_row("Column names", col_names)
            table.add_row("Wells", str(len(meta.wells)))
            if verbose:
                positions = list(reader.positions())
                first_pos = positions[0][1]
                first_array = list(first_pos._member_names)[0]
                total_bytes = sum(p[first_array].nbytes for _, p in positions)
                table.add_row("Positions", str(len(positions)))
                table.add_row("Chunk size", str(first_pos[first_array].chunks))
                table.add_row(
                    "Bytes (decompressed)",
                    f"{total_bytes} [{sizeof_fmt(total_bytes)}]",
                )
        else:
            first_array = list(reader._member_names)[0]
            total_bytes = reader[first_array].nbytes
            space_unit = _get_space_unit(reader.axes)
            unit_str = f" ({space_unit})" if space_unit else ""
            scale_label = f"(Z, Y, X) scale{unit_str}"
            table.add_row(scale_label, str(tuple(reader.scale[2:])))
            table.add_row("Chunk size", str(reader[first_array].chunks))
            table.add_row(
                "Bytes (decompressed)",
                f"{total_bytes} [{sizeof_fmt(total_bytes)}]",
            )

        # Print summary panel
        console.print(
            Panel(table, title="[bold]Summary[/bold]", border_style="blue")
        )

        # Print usage code (if verbose)
        if verbose:
            console.print()
            console.print("[dim]This dataset can be opened with:[/dim]")
            console.print("[green]>>> from iohub import open_ome_zarr[/green]")
            code = f">>> dataset = open_ome_zarr('{path}', mode='r')"
            console.print(f"[green]{code}[/green]")

        reader.close()


def sizeof_fmt(num: int) -> str:
    """
    Human readable file size
    Adapted form:
    https://web.archive.org/web/20111010015624/
    http://blogmag.net/blog/read/38/Print_human_readable_file_size
    """
    if num < 1024:
        return f"{num} B"
    for x in ["KiB", "MiB", "GiB", "TiB"]:
        num /= 1024
        if num < 1024:
            return f"{num:.1f} {x}"
