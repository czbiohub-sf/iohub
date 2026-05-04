"""RFC-9 (Zipped OME-Zarr, ``.ozx``) support — a thin layer over ``zarr.storage.ZipStore``.

Adds on top of zarr-python's ``ZipStore``:

1. ``OzxStore`` — RFC-9 defaults (``ZIP_STORED``, ``allowZip64=True``),
   writes the archive comment on close, fork-safe via ``os.register_at_fork``.
2. Comment helpers — read and write the ``{"ome": {"version": ..., "zipFile": ...}}`` JSON.
3. ``pack_ozx`` / ``unpack_ozx`` — convert directory stores ↔ archives.
   Pack writes BFS-ordered entries with ``jsonFirst:true`` in one pass.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import weakref
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Literal, NamedTuple, override

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


def is_ozx_path(path: str | Path) -> bool:
    """Return True when ``path`` ends with ``.ozx``.

    Case-sensitive to match Zarr's ``.zarr`` precedent within iohub.

    Parameters
    ----------
    path : str | Path
        Path to check; existence not required.
    """
    return Path(path).suffix == OZX_EXTENSION


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


def _read_ozx_comment(path: str | Path) -> dict | None:
    """Return the parsed RFC-9 archive comment, or ``None`` if absent or invalid."""
    with zipfile.ZipFile(path, mode="r") as zf:
        return _parse_comment(zf.comment)


def read_ozx_version(path: str | Path) -> str | None:
    """Return the OME-NGFF version advertised in the archive comment.

    Parameters
    ----------
    path : str | Path
        ``.ozx`` archive to read.
    """
    return get_ome_attrs(_read_ozx_comment(path) or {}).get(_VERSION)


class OzxSummary(NamedTuple):
    """Combined RFC-9 metadata + entry counts read from a ``.ozx``."""

    version: str | None
    json_first: bool
    n_entries: int
    n_zarr_json: int
    n_duplicates: int


def summarize_ozx(path: str | Path) -> OzxSummary:
    """Read all RFC-9 attributes plus entry counts in one zip open.

    Opens the archive once and pulls the comment plus ``namelist`` while
    the file is open. ``zipfile.BadZipFile`` propagates so corrupt
    central directories surface explicitly instead of being masked as
    ``n_entries=0``.

    Parameters
    ----------
    path : str | Path
        ``.ozx`` archive to inspect.
    """
    with zipfile.ZipFile(path, mode="r") as zf:
        comment = _parse_comment(zf.comment)
        names = zf.namelist()
    ome = get_ome_attrs(comment or {})
    version = ome.get(_VERSION)
    json_first = bool(ome.get(_ZIP_FILE, {}).get(_CENTRAL_DIRECTORY, {}).get(_JSON_FIRST, False))
    n_zarr_json = sum(1 for n in names if n.rsplit("/", 1)[-1] == "zarr.json")
    n_duplicates = len(names) - len(set(names))
    return OzxSummary(version, json_first, len(names), n_zarr_json, n_duplicates)


def _write_ozx_comment(
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

    Fork-safe: an ``os.register_at_fork`` hook invalidates the
    inherited ``ZipFile`` fd in child processes, so each child opens
    its own. Spawn-safe via ``ZipStore.__getstate__``/``__setstate__``.
    """

    # Identity-keyed live-instance registry. ``ZipStore`` defines
    # ``__eq__`` (path equality) without ``__hash__``, so ``OzxStore``
    # is unhashable; ``id(self)`` sidesteps that and the eq/hash
    # contract violation a ``__hash__`` override would introduce.
    _live_instances: weakref.WeakValueDictionary[int, OzxStore] = weakref.WeakValueDictionary()

    @classmethod
    def _invalidate_all_after_fork(cls) -> None:
        """Drop the inherited ``ZipFile`` fd on every live instance.

        Called by the module-level ``os.register_at_fork`` hook in each
        forked child so the child re-opens the archive on first read
        instead of sharing the parent's fd.
        """
        for store in list(cls._live_instances.values()):
            store.invalidate_after_fork()

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
        # Track this instance so the at-fork hook can invalidate its
        # inherited fd in any child process.
        type(self)._live_instances[id(self)] = self

    @override
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

    def invalidate_after_fork(self) -> None:
        """Drop the inherited ``ZipFile`` fd after a ``fork()``.

        Called by the module-level ``os.register_at_fork`` hook in every
        forked child so that the child re-opens the archive on its first
        read instead of sharing the parent's fd. No-op if the store has
        not been opened yet.
        """
        self._is_open = False


# Registration runs at module import; POSIX only. Windows has no fork,
# and spawn workers re-open via ``ZipStore.__getstate__``/``__setstate__``.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=OzxStore._invalidate_all_after_fork)


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


def _write_ozx_archive(
    out_path: Path,
    ordered: list[str],
    *,
    version: str,
    open_member: Callable[[str], IO[bytes]],
) -> None:
    """Write an RFC-9 ``.ozx`` to ``out_path`` from a callable entry source.

    Owns the destination ``ZipFile``, the archive comment, and the
    unlink-on-failure cleanup.
    """
    try:
        with zipfile.ZipFile(
            out_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as zout:
            for arcname in ordered:
                with open_member(arcname) as src_f, zout.open(arcname, mode="w") as dst_f:
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


def unpack_ozx(src: str | Path, dst: str | Path) -> Path:
    """Unpack an RFC-9 ``.ozx`` archive into an OME-Zarr directory store.

    Walks the archive, dedupes shadow ``zarr.json`` entries (last-wins,
    matching ``ZipFile.read``), and writes each entry as a file under
    ``dst``. Reverse of :func:`pack_ozx`.

    Parameters
    ----------
    src : str | Path
        Existing ``.ozx`` archive.
    dst : str | Path
        Output directory; must not exist.

    Returns
    -------
    Path
        Path of the written directory store.
    """
    src_path = Path(src)
    if not src_path.is_file():
        raise FileNotFoundError(f"unpack source must be a file: {src_path}")
    out_path = Path(dst)
    if out_path.exists():
        raise FileExistsError(out_path)

    out_path.mkdir(parents=True)
    try:
        with zipfile.ZipFile(src_path, mode="r") as zin:
            # Dedupe by keeping the last ZipInfo per name — matches
            # ZipFile.read semantics for archives with shadow entries.
            infos: dict[str, zipfile.ZipInfo] = {info.filename: info for info in zin.infolist()}
            for name, info in infos.items():
                # Skip directory entries (``name/`` with empty content)
                # that some zip tools (7z, info-zip) emit. They'd
                # otherwise be written as regular files at the directory
                # path and break sibling entry creation.
                if name.endswith("/"):
                    continue
                # Refuse traversal: zip entries with .. or absolute paths
                # could write outside dst. zipfile's extract() does this
                # by default; we replicate the guard since we resolve
                # paths manually.
                target = (out_path / name).resolve()
                if not target.is_relative_to(out_path.resolve()):
                    raise ValueError(f"unsafe entry path escapes destination: {name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zin.open(info, mode="r") as src_f, target.open("wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f, length=_COPY_BUFFER_BYTES)
    except Exception:
        shutil.rmtree(out_path, ignore_errors=True)
        raise
    return out_path
