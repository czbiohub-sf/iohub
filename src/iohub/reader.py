from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import natsort
import tifffile as tiff
import zarr

from iohub.core.ozx import is_ozx_path, read_ozx_version, summarize_ozx
from iohub.fov import BaseFOVMapping
from iohub.mmstack import MMStack
from iohub.ndtiff import NDTiffDataset
from iohub.ngff.nodes import (
    Bioformats2RawSeries,
    NGFFNode,
    Plate,
    Position,
    TiledPosition,
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


def _has_bf2raw_marker(group: zarr.Group) -> bool:
    """Detect ``bioformats2raw.layout`` marker at root, v0.4 or v0.5."""
    if "bioformats2raw.layout" in group.attrs:
        return True
    ome = group.attrs.get("ome")
    return isinstance(ome, dict) and "bioformats2raw.layout" in ome


def _check_zarr_data_type(src: Path):
    try:
        root = zarr.open(src, mode="r")
        if version := _find_ngff_version_in_zarr_group(root):
            return version
        # bf2raw root carries no version; recurse into first child series
        if _has_bf2raw_marker(root):
            for name, child in root.groups():
                if name == "OME":
                    continue
                if version := _find_ngff_version_in_zarr_group(child):
                    return version
            # fall back to zarr format → ngff version mapping
            zarr_format = getattr(root.metadata, "zarr_format", None)
            return "0.5" if zarr_format == 3 else "0.4"
        for _, row in root.groups():
            for _, well in row.groups():
                if version := _find_ngff_version_in_zarr_group(well):
                    return version
    except Exception:  # noqa: BLE001 — version detection may fail in many ways
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
                    if len(tf.pages) == 1:  # and tf.pages[0].is_multipage is False:
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
    sub_dir_name = [subdir.name for subdir in directory.iterdir() if subdir.is_dir()]
    #    assert subDirName, 'No sub directories found'
    return natsort.natsorted(sub_dir_name)


def _infer_format(path: Path):
    extra_info = None
    if path.is_file() and path.suffix.lower() == ".nd2":
        data_type = "nd2"
    elif is_ozx_path(path):
        data_type = "omezarr"
        extra_info = read_ozx_version(path) or "0.5"
    elif ngff_version := _check_zarr_data_type(path):
        data_type = "omezarr"
        extra_info = ngff_version
    elif _check_ndtiff(path):
        data_type = "ndtiff"
    elif _check_multipage_tiff(path):
        data_type = "ometiff"
    elif _check_single_page_tiff(path):
        data_type = "singlepagetiff"
    else:
        raise RuntimeError(f"Failed to infer data type: No compatible data found under {path}.")
    return (data_type, extra_info)


def read_images(
    path: StrOrBytesPath,
    data_type: Literal["ometiff", "ndtiff", "omezarr", "nd2"] | None = None,
) -> BaseFOVMapping | Plate | Position | TiledPosition:
    """Read image arrays and metadata from a Micro-Manager dataset.

    Supported formats are Micro-Manager-acquired TIFF datasets
    (multi-page OME-TIFF, NDTIFF),
    and converted OME-Zarr (v0.4/v0.5 HCS layout).

    Parameters
    ----------
    path : StrOrBytesPath
        File path, directory path to ome-tiff series, or Zarr root path
    data_type : Literal["ometiff", "ndtiff", "omezarr"], optional
        Dataset format, by default None

    Returns
    -------
    BaseFOVMapping | Plate | Position | TiledPosition
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
        raise NotImplementedError(
            "Single-page TIFF reading has been removed. "
            "Please convert your data to OME-Zarr using a previous version of iohub."
        )
    elif data_type == "omezarr":
        if extra_info is None:
            _, extra_info = _infer_format(path)
        if extra_info in ("0.4", "0.5"):
            return open_ome_zarr(path, mode="r", version=extra_info)
        elif extra_info == "0.1":
            raise NotImplementedError(
                "OME-Zarr v0.1 reading has been removed. Please convert your data to OME-Zarr v0.4 or later."
            )
        else:
            raise ValueError(f"NGFF version {extra_info} is not supported.")
    elif data_type == "ndtiff":
        return NDTiffDataset(path)
    elif data_type == "nd2":
        from iohub.nd2 import ND2Dataset

        return ND2Dataset(path)
    else:
        raise ValueError(f"Reader of type {data_type} is not implemented")


def print_info(path: StrOrBytesPath, verbose=False):
    """Print summary information for a dataset.

    Parameters
    ----------
    path : StrOrBytesPath
        Path to the dataset
    verbose : bool, optional
        Show usage guide to open dataset in Python
        and full tree for HCS Plates in OME-Zarr,
        by default False
    """
    path = Path(path).resolve()

    try:
        fmt, extra_info = _infer_format(path)
        if fmt == "omezarr" and extra_info in ("0.4", "0.5"):
            reader = open_ome_zarr(path, mode="r", version=extra_info)
        else:
            reader = read_images(path, data_type=fmt)
    except (ValueError, RuntimeError, NotImplementedError):
        print("Error: No compatible dataset is found.")
        return
    try:
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
                [f"{a}={s}" for s, a in zip(first_fov.shape, ("T", "C", "Z", "Y", "X"), strict=False)]
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
            msgs.extend(
                [
                    sum_msg,
                    fmt_msg,
                    "".join(["Axes:\t\t\t "] + [f"{a.name} ({a.type}); " for a in reader.axes]),
                    ch_msg,
                ]
            )
            if isinstance(reader, Plate):
                meta = reader.metadata
                msgs.extend(
                    [
                        f"Row names:\t\t {[r.name for r in meta.rows]}",
                        f"Column names:\t\t {[c.name for c in meta.columns]}",
                        f"Wells:\t\t\t {len(meta.wells)}",
                    ]
                )
                if verbose:
                    print("Zarr hierarchy:")
                    reader.print_tree()
                    positions = list(reader.positions())
                    total_bytes_uncompressed = sum(p["0"].nbytes for _, p in positions)
                    msgs.append(f"Positions:\t\t {len(positions)}")
                    msgs.append(f"Chunk size:\t\t {positions[0][1][0].chunks}")
                    msgs.append(
                        f"No. bytes decompressed:\t\t, {total_bytes_uncompressed}"
                        f" [{sizeof_fmt(total_bytes_uncompressed)}]"
                    )
            elif isinstance(reader, Bioformats2RawSeries):
                positions = list(reader.positions())
                msgs.append(f"Positions:\t\t {len(positions)}")
                if positions:
                    name, first = positions[0]
                    msgs.append(f"First position:\t\t '{name}'")
                    if "0" in first:
                        msgs.append(f"Chunk size:\t\t {first['0'].chunks}")
                if verbose:
                    print("Zarr hierarchy:")
                    reader.print_tree()
            else:
                total_bytes_uncompressed = reader["0"].nbytes
                msgs.append(f"(Z, Y, X) scale (um):\t {tuple(reader.scale[2:])}")
                msgs.append(f"Chunk size:\t\t {reader['0'].chunks}")
                msgs.append(
                    f"No. bytes decompressed:\t\t {total_bytes_uncompressed} [{sizeof_fmt(total_bytes_uncompressed)}]"
                )
            if verbose:
                msgs.extend(
                    [
                        code_msg,
                        ">>> from iohub import open_ome_zarr",
                        f">>> dataset = open_ome_zarr('{path}', mode='r')",
                    ]
                )
                if is_ozx_path(path):
                    msgs.extend(_rfc9_info_lines(path))
            if isinstance(reader, Position):
                print("Zarr hierarchy:")
                reader.print_tree()
            print("\n".join(msgs))
    finally:
        if hasattr(reader, "close"):
            reader.close()


def _rfc9_info_lines(path: Path) -> list[str]:
    """Return one-line summary entries for an ``.ozx`` archive's RFC-9 metadata."""
    s = summarize_ozx(path)
    return [
        "\n=== RFC-9 archive ===",
        f"OME version:\t\t {s.version}",
        f"jsonFirst:\t\t {s.json_first}",
        f"Entries:\t\t {s.n_entries:,}  ({s.n_zarr_json} zarr.json, {s.n_duplicates} duplicate names)",
        f"Size on disk:\t\t {sizeof_fmt(path.stat().st_size)}",
    ]


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
