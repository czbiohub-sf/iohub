"""RFC-9 (Zipped OME-Zarr, ``.ozx``) support — a thin layer over ``zarr.storage.ZipStore``.

Adds on top of zarr-python's ``ZipStore``:

1. ``OzxStore`` — RFC-9 defaults (``ZIP_STORED``, ``allowZip64=True``) and
   writes the archive comment on close.
2. Comment helpers — read and write the ``{"ome": {"version": ..., "zipFile": ...}}`` JSON.
3. ``pack_ozx`` — emit a BFS-ordered archive from a directory store, satisfying
   the RFC-9 SHOULD ordering clause in one pass.
"""

from __future__ import annotations

import json
import stat
import logging
import shutil
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Literal, NamedTuple

from zarr.storage import ZipStore

from iohub.core.compat import get_ome_attrs

_logger = logging.getLogger(__name__)

OZX_EXTENSION = ".ozx"

# 1 MiB: bounds peak memory on sharded archives where a single chunk
# can be hundreds of MB.
_COPY_BUFFER_BYTES = 1 << 20

_OME = "ome"
_VERSION = "version"
_ZIP_FILE = "zipFile"
_CENTRAL_DIRECTORY = "centralDirectory"
_JSON_FIRST = "jsonFirst"

# Reproducibility: pin every variable field on each ZipInfo so two
# `pack_ozx` runs over the same source produce byte-identical archives.
# The defaults Python's ``zipfile`` would otherwise pick — wall-clock
# mtime, OS-specific create_system, OS-specific external_attr — make
# downstream sha256 verification meaningless across machines and runs.
#
# - 1980-01-01 is the earliest representable zip timestamp.
# - create_system 3 is the canonical Unix creator; using it on every
#   platform ensures Windows-side packs match Linux-side packs.
# - external_attr is a regular-file 0o644 in the upper 16 bits, where
#   external file attributes live for Unix-created entries.
_REPRODUCIBLE_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_REPRODUCIBLE_FILE_ATTR = (stat.S_IFREG | 0o644) << 16
_CREATE_SYSTEM_UNIX = 3


def is_ozx_path(path: str | Path) -> bool:
    """Return True when ``path`` ends with ``.ozx``.

    Parameters
    ----------
    path : str | Path
        Path to check; existence not required.
    """
    return Path(path).suffix.lower() == OZX_EXTENSION


def _build_comment(version: str, *, json_first: bool = False) -> bytes:
    ome: dict = {_VERSION: str(version)}
    if json_first:
        ome[_ZIP_FILE] = {_CENTRAL_DIRECTORY: {_JSON_FIRST: True}}
    return json.dumps({_OME: ome}, separators=(",", ":")).encode("utf-8")


def _parse_comment(raw: bytes) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _logger.warning("OZX archive comment is not valid UTF-8 JSON; ignoring.")
        return None


def _comment_version(comment: dict | None) -> str | None:
    return get_ome_attrs(comment or {}).get(_VERSION)


def _comment_json_first(comment: dict | None) -> bool:
    return bool(get_ome_attrs(comment or {}).get(_ZIP_FILE, {}).get(_CENTRAL_DIRECTORY, {}).get(_JSON_FIRST, False))


def read_ozx_comment(path: str | Path) -> dict | None:
    """Return the parsed RFC-9 archive comment, or ``None`` if absent or invalid.

    Parameters
    ----------
    path : str | Path
        ``.ozx`` archive to read.
    """
    with zipfile.ZipFile(path, mode="r") as zf:
        return _parse_comment(zf.comment)


def read_ozx_version(path: str | Path) -> str | None:
    """Return the OME-NGFF version advertised in the archive comment.

    Parameters
    ----------
    path : str | Path
        ``.ozx`` archive to read.
    """
    return _comment_version(read_ozx_comment(path))


def read_ozx_json_first(path: str | Path) -> bool:
    """Return the ``jsonFirst`` ordering hint (defaults to ``False``).

    Parameters
    ----------
    path : str | Path
        ``.ozx`` archive to read.
    """
    return _comment_json_first(read_ozx_comment(path))


class OzxSummary(NamedTuple):
    """RFC-9 attributes carried in an ``.ozx`` archive comment."""

    version: str | None
    json_first: bool


def summarize_ozx(path: str | Path) -> OzxSummary:
    """Read all RFC-9 attributes in one zip open.

    Prefer this when you need both the version and ``jsonFirst`` —
    one open instead of two.

    Parameters
    ----------
    path : str | Path
        ``.ozx`` archive to inspect.
    """
    comment = read_ozx_comment(path)
    return OzxSummary(_comment_version(comment), _comment_json_first(comment))


def write_ozx_comment(
    path: str | Path,
    *,
    version: str,
    json_first: bool = False,
) -> None:
    """Set the RFC-9 archive comment on an existing zip file.

    Use this to patch archives not written through ``OzxStore``.

    Parameters
    ----------
    path : str | Path
        Existing zip file to patch.
    version : str
        OME-NGFF version to record.
    json_first : bool
        True if all ``zarr.json`` entries precede chunks in BFS order.
    """
    # ``mode="a"`` rewrites the central directory on close — the cheap
    # way to update the comment without touching entry data.
    with zipfile.ZipFile(path, mode="a") as zf:
        zf.comment = _build_comment(version, json_first=json_first)


class OzxStore(ZipStore):
    """RFC-9 (Zipped OME-Zarr) store.

    A :class:`zarr.storage.ZipStore` subclass with RFC-9 defaults:
    ``ZIP_STORED`` (Zarr codecs handle compression), ``allowZip64=True``
    (regardless of size), and the archive comment written on ``close()``.

    The store writes entries in zarr's creation order and leaves
    ``jsonFirst`` unset. For a SHOULD-compliant archive ready for
    HTTP-range publishing, write to a directory store first and run
    :func:`pack_ozx`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: Literal["r", "w", "a"] = "r",
        read_only: bool | None = None,
        ome_version: str = "0.5",
        allowZip64: bool = True,
        compression: int = zipfile.ZIP_STORED,
    ) -> None:
        super().__init__(
            path,
            mode=mode,
            read_only=read_only,
            compression=compression,
            allowZip64=allowZip64,
        )
        self._ome_version = str(ome_version)

    def close(self) -> None:
        """Write the RFC-9 archive comment (if writable) then close."""
        # ZipStore opens lazily via ``_sync_open``; ``_zf``/``_lock`` only
        # exist after first use. The ``_is_open`` guard handles untouched
        # stores — writers that exited before any key write.
        if not self._is_open:
            return
        if not self.read_only:
            try:
                self._zf.comment = _build_comment(self._ome_version)
            except Exception as err:  # noqa: BLE001 — never block close on comment failure
                _logger.warning("Failed to write OZX archive comment: %s", err)
        super().close()


def _bfs_order(names: Iterable[str]) -> list[str]:
    """Order ``names`` for RFC-9 SHOULD compliance.

    All ``zarr.json`` entries first, sorted by depth then name (root
    sorts first because it sits at depth 0). Other entries keep their
    input order.
    """
    meta: list[tuple[int, str]] = []
    chunks: list[str] = []
    for name in names:
        if name.rsplit("/", 1)[-1] == "zarr.json":
            meta.append((name.count("/"), name))
        else:
            chunks.append(name)
    meta.sort()
    return [name for _, name in meta] + chunks


def _reproducible_zip_info(arcname: str) -> zipfile.ZipInfo:
    """Build a ``ZipInfo`` with every variable header field pinned.

    Pass to ``zipfile.ZipFile.open(zinfo, mode="w")`` instead of a
    bare arcname — that path lets Python stamp ``date_time``,
    ``create_system``, and ``external_attr`` from runtime state,
    which is what makes consecutive ``pack_ozx`` runs produce
    different bytes for the same source.
    """
    zinfo = zipfile.ZipInfo(filename=arcname, date_time=_REPRODUCIBLE_DATE_TIME)
    zinfo.compress_type = zipfile.ZIP_STORED
    zinfo.create_system = _CREATE_SYSTEM_UNIX
    zinfo.external_attr = _REPRODUCIBLE_FILE_ATTR
    return zinfo


def _write_ozx_archive(
    out_path: Path,
    ordered: list[str],
    *,
    version: str,
    open_member: Callable[[str], IO[bytes]],
) -> None:
    """Write an RFC-9 ``.ozx`` to ``out_path`` from a callable entry source.

    Owns the destination ``ZipFile``, the archive comment, and the
    unlink-on-failure cleanup. Each entry's header is built from
    :func:`_reproducible_zip_info` so the resulting bytes are stable
    across machines and re-runs given identical input.
    """
    try:
        with zipfile.ZipFile(
            out_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as zout:
            for arcname in ordered:
                zinfo = _reproducible_zip_info(arcname)
                with (
                    open_member(arcname) as src_f,
                    zout.open(zinfo, mode="w", force_zip64=True) as dst_f,
                ):
                    shutil.copyfileobj(src_f, dst_f, length=_COPY_BUFFER_BYTES)
            zout.comment = _build_comment(version, json_first=True)
    except Exception:
        if out_path.exists():
            out_path.unlink()
        raise


def pack_ozx(
    src: str | Path,
    dst: str | Path,
    *,
    version: str | None = None,
) -> Path:
    """Pack an OME-Zarr directory store into an RFC-9 ``.ozx`` archive.

    Writes entries in BFS order with ``jsonFirst:true`` — fully
    SHOULD-compliant in one pass.

    Parameters
    ----------
    src : str | Path
        OME-Zarr directory store.
    dst : str | Path
        Output ``.ozx`` path; must not exist.
    version : str | None
        OME-NGFF version for the archive comment. Sniffs from ``src``
        when omitted.

    Returns
    -------
    Path
        Path of the written archive.
    """
    src_path = Path(src)
    if not src_path.is_dir():
        raise NotADirectoryError(f"pack source must be a directory: {src_path}")
    out_path = Path(dst)
    if out_path.exists():
        raise FileExistsError(out_path)

    if version is None:
        # Local import: ``iohub`` re-exports from ``iohub.core``, which
        # would create a cycle at module load.
        from iohub import open_ome_zarr

        with open_ome_zarr(src_path, mode="r") as node:
            version = node.version

    files = [p for p in src_path.rglob("*") if p.is_file()]
    rel_names = [str(p.relative_to(src_path)).replace("\\", "/") for p in files]
    by_name = dict(zip(rel_names, files, strict=True))
    ordered = _bfs_order(rel_names)

    _write_ozx_archive(out_path, ordered, version=version, open_member=lambda name: by_name[name].open("rb"))
    return out_path
