"""Tests for RFC-9 (Zipped OME-Zarr, ``.ozx``) support."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from iohub import open_ome_zarr
from iohub.core.ozx import (
    is_ozx_path,
    pack_ozx,
    read_ozx_json_first,
    read_ozx_version,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [("foo.ozx", True), ("FOO.OZX", True), ("foo.zarr", False)],
)
def test_is_ozx_path(path: str, expected: bool) -> None:
    assert is_ozx_path(path) is expected


def test_fov_roundtrip_v0_5(tmp_path: Path) -> None:
    """End-to-end v0.5 write/read + verify RFC-9 archive shape."""
    path = tmp_path / "fov.ozx"
    data = np.arange(16, dtype=np.uint16).reshape(1, 1, 1, 4, 4)
    with open_ome_zarr(path, layout="fov", mode="w", channel_names=["DAPI"], version="0.5") as pos:
        arr = pos.create_zeros("0", shape=data.shape, dtype=data.dtype, chunks=(1, 1, 1, 2, 2))
        arr[:] = data
    with open_ome_zarr(path, mode="r") as pos:
        assert pos.channel_names == ["DAPI"]
        np.testing.assert_array_equal(pos["0"][:], data)
    # RFC-9: archive comment carries OME version, all entries ZIP_STORED.
    with zipfile.ZipFile(path) as zf:
        assert json.loads(zf.comment) == {"ome": {"version": "0.5"}}
        assert {info.compress_type for info in zf.infolist()} == {zipfile.ZIP_STORED}


def test_fov_roundtrip_v0_4(tmp_path: Path) -> None:
    """v0.4 uses zarr_format=2 — separate code path from v0.5."""
    path = tmp_path / "fov_v04.ozx"
    with open_ome_zarr(path, layout="fov", mode="w", channel_names=["GFP"], version="0.4") as pos:
        pos.create_zeros("0", shape=(1, 1, 1, 2, 2), dtype=np.uint8, chunks=(1, 1, 1, 2, 2))
    assert read_ozx_version(path) == "0.4"
    with open_ome_zarr(path, mode="r") as pos:
        assert pos.metadata.multiscales[0].version == "0.4"


def test_hcs_roundtrip(tmp_path: Path) -> None:
    """Plate layout writes multiple nested groups — different enough from FOV to test."""
    path = tmp_path / "plate.ozx"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["DAPI"], version="0.5") as plate:
        for well in (("A", "1"), ("A", "2")):
            pos = plate.create_position(*well, "0")
            pos.create_zeros("0", shape=(1, 1, 1, 2, 2), dtype=np.uint8, chunks=(1, 1, 1, 2, 2))
    with open_ome_zarr(path, mode="r") as plate:
        assert plate.channel_names == ["DAPI"]
        assert next(iter(plate["A/1"].positions()))[0] == "0"


def test_pack_ozx_from_directory(tmp_path: Path) -> None:
    """pack_ozx walks a .zarr dir, sniffs version, emits ordered .ozx."""
    src_dir = tmp_path / "src.zarr"
    with open_ome_zarr(src_dir, layout="fov", mode="w", channel_names=["c"], version="0.5") as pos:
        arr = pos.create_zeros("0", shape=(1, 1, 1, 4, 4), dtype=np.uint8, chunks=(1, 1, 1, 2, 2))
        arr[:] = np.ones(arr.shape, dtype=np.uint8)

    dst = tmp_path / "out.ozx"
    pack_ozx(src_dir, dst)

    # Version sniffed from the source's zarr.json, jsonFirst flipped on.
    assert read_ozx_version(dst) == "0.5"
    assert read_ozx_json_first(dst) is True

    # Roundtrip via the public API.
    with open_ome_zarr(dst, mode="r") as pos:
        np.testing.assert_array_equal(pos["0"][:], np.ones((1, 1, 1, 4, 4), dtype=np.uint8))

    # Refuse overwriting existing destination.
    with pytest.raises(FileExistsError):
        pack_ozx(src_dir, dst)


def test_pack_ozx_is_byte_reproducible(tmp_path: Path) -> None:
    """Two pack_ozx runs over the same source produce byte-identical archives.

    Without pinning ``date_time`` / ``create_system`` / ``external_attr``
    on each ``ZipInfo``, Python's default ``zipfile`` behaviour stamps
    wall-clock mtimes into every Local File Header, so consecutive packs
    diverge at the byte level — and any sha256-based artifact integrity
    check (Croissant ``cr:FileObject.sha256``, OZX MANIFEST.json) fails.
    """
    src_dir = tmp_path / "src.zarr"
    with open_ome_zarr(
        src_dir, layout="fov", mode="w", channel_names=["c"], version="0.5"
    ) as pos:
        arr = pos.create_zeros(
            "0", shape=(1, 1, 1, 4, 4), dtype=np.uint8, chunks=(1, 1, 1, 2, 2)
        )
        arr[:] = np.arange(16, dtype=np.uint8).reshape(arr.shape)

    a = tmp_path / "a.ozx"
    b = tmp_path / "b.ozx"
    pack_ozx(src_dir, a)
    pack_ozx(src_dir, b)
    assert a.read_bytes() == b.read_bytes()


def test_pack_ozx_pinned_zip_metadata(tmp_path: Path) -> None:
    """Every entry's ZipInfo has the pinned date_time / create_system / external_attr.

    Guards the upper-half of the reproducibility chain — the round-trip
    test confirms the bytes match across two runs, this confirms
    *which* fields were pinned. If a future refactor drops one of the
    pins, that field starts drifting silently across machines but the
    byte-equality test still passes for two same-machine same-second
    runs; this assertion catches that.
    """
    src_dir = tmp_path / "src.zarr"
    with open_ome_zarr(
        src_dir, layout="fov", mode="w", channel_names=["c"], version="0.5"
    ) as pos:
        pos.create_zeros(
            "0", shape=(1, 1, 1, 2, 2), dtype=np.uint8, chunks=(1, 1, 1, 2, 2)
        )
    dst = tmp_path / "out.ozx"
    pack_ozx(src_dir, dst)

    with zipfile.ZipFile(dst) as zf:
        infos = zf.infolist()
    assert infos, "expected at least one zip entry"
    expected_attr = (0o100000 | 0o644) << 16  # stat.S_IFREG | 0o644, shifted
    for info in infos:
        assert info.date_time == (1980, 1, 1, 0, 0, 0), (
            f"{info.filename}: date_time {info.date_time} not pinned"
        )
        assert info.create_system == 3, (
            f"{info.filename}: create_system {info.create_system} not Unix (3)"
        )
        assert info.external_attr == expected_attr, (
            f"{info.filename}: external_attr {oct(info.external_attr)} "
            f"not {oct(expected_attr)}"
        )


def test_tensorstore_rejected_for_ozx(tmp_path: Path) -> None:
    """TS cannot write .ozx (its zip kvstore is read-only). Fail loud, not late."""
    pytest.importorskip("tensorstore")
    with pytest.raises(ValueError, match="does not support RFC-9"):
        open_ome_zarr(
            tmp_path / "ts.ozx",
            layout="fov",
            mode="w",
            channel_names=["c"],
            version="0.5",
            implementation="tensorstore",
        )
