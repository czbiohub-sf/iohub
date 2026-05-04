"""Tests for RFC-9 (Zipped OME-Zarr, ``.ozx``) support."""

from __future__ import annotations

import multiprocessing as mp
import sys
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import zarr

from iohub import open_ome_zarr
from iohub.core.ozx import (
    OzxStore,
    is_ozx_path,
    pack_ozx,
    read_ozx_version,
    summarize_ozx,
    unpack_ozx,
)
from tests.conftest import make_fov_zarr


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("foo.ozx", True),
        ("FOO.OZX", False),  # case-sensitive, matches .zarr precedent
        ("foo.zarr", False),
    ],
)
def test_is_ozx_path(path: str, expected: bool) -> None:
    assert is_ozx_path(path) is expected


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_fov_roundtrip(tmp_path: Path, version: Literal["0.4", "0.5"]) -> None:
    """E2E write/read for both NGFF versions + verify RFC-9 archive shape."""
    path = tmp_path / f"fov_{version}.ozx"
    data = np.arange(16, dtype=np.uint8).reshape(1, 1, 1, 4, 4)
    make_fov_zarr(path, data, version=version)
    assert read_ozx_version(path) == version
    with open_ome_zarr(path, mode="r") as pos:
        np.testing.assert_array_equal(pos["0"][:], data)
    with zipfile.ZipFile(path) as zf:
        assert zf.comment == f'{{"ome":{{"version":"{version}"}}}}'.encode()
        assert {info.compress_type for info in zf.infolist()} == {zipfile.ZIP_STORED}


def test_hcs_roundtrip(tmp_path: Path) -> None:
    """Plate layout exercises a different write path (nested groups)."""
    path = tmp_path / "plate.ozx"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["DAPI"], version="0.5") as plate:
        for well in (("A", "1"), ("A", "2")):
            pos = plate.create_position(*well, "0")
            pos.create_zeros("0", shape=(1, 1, 1, 2, 2), dtype=np.uint8, chunks=(1, 1, 1, 2, 2))
    with open_ome_zarr(path, mode="r") as plate:
        assert plate.channel_names == ["DAPI"]
        assert next(iter(plate["A/1"].positions()))[0] == "0"


def test_pack_unpack_roundtrip(tmp_path: Path) -> None:
    """``pack_ozx`` then ``unpack_ozx`` round-trips data, version, and ordering;
    pack refuses to overwrite an existing destination.
    """
    src = tmp_path / "src.zarr"
    data = np.arange(64, dtype=np.uint8).reshape(1, 1, 1, 8, 8)
    make_fov_zarr(src, data)

    ozx = tmp_path / "src.ozx"
    pack_ozx(src, ozx)
    s = summarize_ozx(ozx)
    assert s.version == "0.5"  # sniffed from src
    assert s.json_first is True  # BFS-ordered output

    restored = tmp_path / "restored.zarr"
    unpack_ozx(ozx, restored)
    with open_ome_zarr(restored, mode="r") as pos:
        np.testing.assert_array_equal(pos["0"][:], data)

    with pytest.raises(FileExistsError):
        pack_ozx(src, ozx)


def test_unpack_ozx_dedupes_shadow_entries(tmp_path: Path) -> None:
    """Duplicate entry names (shadow writes) collapse to the last — matches
    ``ZipFile.read`` resolution."""
    ozx = tmp_path / "shadow.ozx"
    with zipfile.ZipFile(ozx, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("zarr.json", b'{"shadow": true}')
        zf.writestr("zarr.json", b'{"shadow": false}')

    out = tmp_path / "unpacked"
    unpack_ozx(ozx, out)
    assert (out / "zarr.json").read_text() == '{"shadow": false}'


def test_unpack_ozx_rejects_path_traversal(tmp_path: Path) -> None:
    """Entries with ``..`` must not write outside ``dst``; partial output is removed."""
    ozx = tmp_path / "evil.ozx"
    with zipfile.ZipFile(ozx, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("../escape.txt", b"pwn")

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="escapes destination"):
        unpack_ozx(ozx, out)
    # Cleanup-on-failure invariant: no half-written tree left behind.
    assert not out.exists()


def test_pack_ozx_cleans_up_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``_write_ozx_archive`` raises mid-write, the partial ``.ozx`` is removed.

    Otherwise users hit ``FileExistsError`` on retry from leftover output.
    """
    src = tmp_path / "src.zarr"
    make_fov_zarr(src, np.zeros((1, 1, 1, 4, 4), dtype=np.uint8))
    dst = tmp_path / "out.ozx"

    # Force a failure mid-copy by patching shutil.copyfileobj inside the
    # ozx module — _write_ozx_archive's try/except + unlink runs.
    from iohub.core import ozx as ozx_mod

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated mid-copy failure")

    monkeypatch.setattr(ozx_mod.shutil, "copyfileobj", boom)
    with pytest.raises(RuntimeError, match="simulated mid-copy"):
        pack_ozx(src, dst)
    # Cleanup-on-failure invariant: partial output unlinked.
    assert not dst.exists()


def test_unpack_ozx_handles_directory_entries(tmp_path: Path) -> None:
    """Zips from other tools (7z, info-zip) sometimes include directory entries
    (``name/`` with empty content). ``unpack_ozx`` must not crash on them.
    """
    ozx = tmp_path / "with_dirs.ozx"
    with zipfile.ZipFile(ozx, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("zarr.json", b"{}")
        zf.writestr("inner/", b"")  # directory entry — 7z-style
        zf.writestr("inner/zarr.json", b"{}")

    out = tmp_path / "out"
    unpack_ozx(ozx, out)
    assert (out / "zarr.json").is_file()
    assert (out / "inner" / "zarr.json").is_file()


def _build_summarize_fixture(tmp_path: Path, kind: str) -> Path:
    """Make a fixture archive for ``test_summarize_ozx_edges``."""
    path = tmp_path / f"{kind}.zip"
    if kind == "missing_comment":
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("zarr.json", b"{}")
    elif kind == "malformed_comment":
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("zarr.json", b"{}")
            zf.comment = b"not json at all"
    elif kind == "corrupt_zip":
        path.write_bytes(b"\x00" * 1024)
    return path


@pytest.mark.parametrize("kind", ["missing_comment", "malformed_comment", "corrupt_zip"])
def test_summarize_ozx_edges(tmp_path: Path, kind: str) -> None:
    """Summary degrades gracefully for missing/malformed comments and propagates
    ``BadZipFile`` for corrupt archives — no silent ``n_entries=0``.
    """
    path = _build_summarize_fixture(tmp_path, kind)
    if kind == "corrupt_zip":
        with pytest.raises(zipfile.BadZipFile):
            summarize_ozx(path)
        return
    s = summarize_ozx(path)
    assert s.version is None
    assert s.json_first is False
    assert s.n_entries == 1


def _read_many_in_child(ozx_path_str: str, n_iters: int, q: mp.Queue) -> None:
    store = OzxStore(ozx_path_str, mode="r")
    grp = zarr.open_group(store=store, mode="r")
    arr = grp["0"]
    digests = [bytes(arr[:].tobytes()) for _ in range(n_iters)]
    store.close()
    q.put(digests)


@pytest.mark.skipif(sys.platform == "win32", reason="fork only on POSIX")
def test_fork_safety(tmp_path: Path) -> None:
    """Fork-children reading from a parent-opened archive each get a clean fd."""
    src = tmp_path / "src.zarr"
    # 16 chunks total (4x4 grid); each child reads them all ``n_iters``
    # times to maximize the chance a shared-fd race surfaces.
    data = np.arange(256, dtype=np.uint8).reshape(1, 1, 1, 16, 16)
    make_fov_zarr(src, data)
    ozx_path = tmp_path / "src.ozx"
    pack_ozx(src, ozx_path)

    parent_store = OzxStore(ozx_path, mode="r")
    parent_grp = zarr.open_group(store=parent_store, mode="r")
    expected = parent_grp["0"][:].tobytes()

    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    n_workers, n_iters = 4, 5
    procs = [ctx.Process(target=_read_many_in_child, args=(str(ozx_path), n_iters, q)) for _ in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=15)
        assert p.exitcode == 0, f"child crashed: exitcode={p.exitcode}"

    parent_store.close()
    for _ in range(n_workers):
        digests = q.get(timeout=2)
        assert all(d == expected for d in digests)


def test_tensorstore_rejected_for_ozx(tmp_path: Path) -> None:
    """TS cannot write ``.ozx`` (its zip kvstore is read-only). Fail loud, not late."""
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
