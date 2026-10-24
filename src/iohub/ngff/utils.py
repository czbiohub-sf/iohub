from __future__ import annotations

import inspect
import itertools
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path
from typing import Any, Literal

import click
import numpy as np
from numpy.typing import DTypeLike, NDArray

from iohub.core.compat import V04_MAX_CHUNK_SIZE_BYTES
from iohub.core.sharding import ShardsLike
from iohub.core.utils import normalize_path
from iohub.ngff import open_ome_zarr
from iohub.ngff._write_units import (
    WriteUnit,
    plan_write_unit,
    progress_dir_for,
    tracking_available,
    unit_is_complete,
)
from iohub.ngff.nodes import TransformationMeta

#: Default ZYX chunk size for OME-Zarr v0.5 stores: ~2 MB at uint16 / ~4 MB at float32.
_V05_DEFAULT_ZYX_CHUNKS: tuple[int, int, int] = (16, 256, 256)

#: zattrs keys the OME-Zarr spec owns ON A POSITION. Excluded when copying custom
#: metadata between positions, and refused as ``extra_metadata`` keys.
#:
#: The entries do different jobs, and which one applies depends on the version:
#:
#: * ``ome`` covers **v0.5**, where every OME key is nested under it, so this
#:   single entry protects the whole block at position level.
#: * ``multiscales``, ``omero`` and ``version`` are top-level in **v0.4**, and
#:   are what protects them there. They are belt-and-braces for the copy — a
#:   freshly created position already carries all three, and the copy skips any
#:   key the destination has — but they are load-bearing for the
#:   ``extra_metadata`` refusal, which has no such fallback.
#: * ``labels`` is the one entry the copy genuinely needs. A destination that
#:   has no label arrays has no ``labels`` key either, so the "already present"
#:   check does not fire, and without this the source's label reference is
#:   copied onto a store where it dangles.
_POSITION_OME_KEYS = {"ome", "multiscales", "omero", "labels", "version"}


def _validated_extra_metadata(
    extra_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Check a caller-supplied provenance mapping and return it as a plain dict.

    Shared by ``create_empty_plate`` and ``process_single_position`` so the two
    seats that write ``extra_metadata`` cannot disagree about what is legal.
    Every entry becomes a TOP-LEVEL zattrs key, so a key the OME-Zarr spec owns
    would corrupt the position rather than annotate it.
    """
    if extra_metadata is None:
        return {}
    if not isinstance(extra_metadata, Mapping):
        raise TypeError(f"extra_metadata must be a mapping, got {type(extra_metadata).__name__}.")
    non_string_keys = [key for key in extra_metadata if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"extra_metadata keys must be strings, got {non_string_keys}.")
    reserved = _POSITION_OME_KEYS.intersection(extra_metadata)
    if reserved:
        raise ValueError(
            f"extra_metadata keys {sorted(reserved)} are reserved OME-Zarr "
            f"keys and cannot be written as top-level zattrs. Use a "
            f"namespaced key (e.g. '<package>-<step>') instead."
        )
    return dict(extra_metadata)


def _selected_metadata_keys(
    source_attrs: Mapping[str, Any],
    metadata_keys: str | Iterable[str] | None,
) -> list[str]:
    """Names of the source zattrs to copy to a destination position.

    OME-owned keys are always excluded. When ``metadata_keys`` is None every
    remaining key is selected; otherwise a key is selected only if it matches
    at least one of the ``fnmatch`` patterns.

    Patterns are shell globs rather than regexes: the strings callers actually
    write, like ``"biahub-*"``, are also valid regexes that mean something
    else, so reading them as regexes would silently select nothing instead of
    raising. The iterable already supplies the alternation a regex would add.
    ``fnmatchcase`` keeps matching case-sensitive on every platform, matching
    how zattrs keys compare.
    """
    candidates = (k for k in source_attrs if k not in _POSITION_OME_KEYS)
    if metadata_keys is None:
        return list(candidates)
    if isinstance(metadata_keys, str):
        patterns = [metadata_keys]
    else:
        patterns = list(metadata_keys)
    return [k for k in candidates if any(fnmatchcase(k, p) for p in patterns)]


def create_empty_plate(
    store_path: Path,
    position_keys: list[tuple[str, str, str]],
    channel_names: list[str],
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | None = None,
    shards: ShardsLike = None,
    shards_ratio: tuple[int, ...] | None = None,
    version: Literal["0.4", "0.5"] = "0.5",
    scale: tuple[float, ...] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
    metadata_sources: Path | str | list[Path | str] | None = None,
    metadata_keys: str | Iterable[str] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    label_names: list[str] | None = None,
    label_dtype: DTypeLike = np.uint32,
    label_chunks: tuple[int, ...] | None = None,
    label_shards: ShardsLike = None,
    label_shards_ratio: tuple[int, ...] | None = None,
) -> None:
    """
    Create a new HCS Plate in OME-Zarr format if the plate does not exist.

    If the plate exists, append positions and channels
    if they are not already in the plate.

    Parameters
    ----------
    store_path : Path
        Path to the HCS plate.
    position_keys : list[tuple[str, str, str]]
        Position keys (row, column, fov) to append if not present in the plate,
        e.g., [("A", "1", "0"), ("A", "1", "1")].
    channel_names : list[str]
        List of channel names. If the store exists,
        append if not present in metadata.
    shape : tuple[int, ...]
        TCZYX shape of the plate.
    chunks : tuple[int, ...], optional
        TCZYX chunk size of the plate. If None, a version-specific default
        is used:
        - "0.4": ``(1, 1, Z, Y, X)`` capped in Z to 500 MB.
        - "0.5": ``(1, 1, 16, 256, 256)`` clamped to ``shape``.
        Defaults to None.
    shards : tuple[int, ...] or str, optional
        Shard geometry of the plate. One of:

        - an explicit TCZYX shard shape in array elements, rounded up to whole
          chunks;
        - a spatial extent, ``"XY"`` for one shard per plane or ``"XYZ"`` for
          one shard per ZYX volume;
        - a target file size such as ``"2GB"`` or ``"512MiB"``, which iohub
          converts to the largest chunk-aligned shard that fits under it. X
          grows first, then Y, then Z, then T, so a torn shard costs a spatial
          region before it costs timepoints, and the channel axis is never
          grown (channel-sharded stores cannot be written a channel at a
          time). The target is compared against the nominal (uncompressed)
          size, so compressed shards land under it.

        If None, a version-specific default is used:

        - "0.4": no sharding (Zarr v2 does not support it).
        - "0.5": ``"XYZ"``, one shard per ZYX volume.

        Defaults to None.
    shards_ratio : tuple[int, ...], optional
        Deprecated TCZYX multiplier over ``chunks``
        (``shards = chunks * shards_ratio``). Use ``shards`` instead, which
        does not require knowing the chunk shape; passing both raises
        ``ValueError``. Defaults to None.
    version : Literal["0.4", "0.5"], optional
        OME-Zarr version to use for the plate.
        Defaults to "0.5".
    scale : tuple[float, ...], optional
        TCZYX scale of the plate. Defaults to (1, 1, 1, 1, 1).
    dtype : DTypeLike, optional
        Data type of the plate. Defaults to np.float32.
    metadata_sources : Path or str or list of Path or str, optional
        Path(s) to one or more source HCS plates from which to copy
        per-position metadata. This and ``metadata_keys`` together define the
        copy: ``metadata_sources`` is *where* metadata comes from,
        ``metadata_keys`` is *which* of it is taken.
        When set, custom (non-OME) zattrs (e.g. provenance keys such as
        ``biahub-flat_field``) are transferred from matching positions in the
        source plate(s). Metadata is only transferred for newly created
        positions, and a given zattrs key is only copied if it does not
        already exist on the destination position (so earlier sources take
        precedence over later ones). Coordinate transforms, axis definitions,
        and label references are **not** copied.
        Defaults to None (no metadata copy).
    metadata_keys : str or iterable of str, optional
        Case-sensitive shell-glob patterns (``fnmatch``, not regex) narrowing
        which zattrs keys ``metadata_sources`` may contribute, e.g.
        ``{"provenance-*", "acquisition"}``. A key is
        copied only if it matches at least one pattern; OME-owned keys are
        excluded either way. This only filters the sources, so it is
        meaningless on its own: passing it without ``metadata_sources``
        raises ``ValueError``.
        Defaults to None (copy every non-OME key the sources provide).
    extra_metadata : Mapping[str, Any], optional
        The caller's *own* per-position metadata, where ``metadata_sources`` is
        metadata from elsewhere. Each entry is written as a TOP-LEVEL zattrs key
        on every position in ``position_keys`` -- the same spelling
        ``process_single_position`` uses -- so a processing step records its
        provenance as a sibling of what it inherited, e.g.
        ``extra_metadata={"biahub-deskew": settings.model_dump()}``.

        Two deliberate differences from ``metadata_sources``:

        - It is written to **every** position named, not only newly created
          ones. Re-running with a changed configuration must refresh the record;
          inherited metadata, by contrast, describes the store as it was made.
        - It **wins** over an inherited key of the same name, since the caller
          is the authority on its own step. Sources still take precedence over
          each other in order.

        Writing the record here rather than after the call is what lets a
        downstream ``create_empty_plate`` inherit it: the source copy reads the
        upstream store's zattrs at plate-creation time, so a pipeline that
        scaffolds all of its stores up front would otherwise chain empty
        provenance. Keys the OME-Zarr spec owns are refused.
        Defaults to None.
    label_names : list[str], optional
        Names of empty label images to create in every position in
        ``position_keys``, at ``<fov>/labels/<name>/``. Each label mirrors
        the position's image pyramid: one level per image level, with the
        same path, axes, scale (and translation, if any), and the image's
        shape with one channel, ``(T, 1, Z, Y, X)``: a label is one
        segmentation, and the DCA array standard wants every array 5D. For a
        newly created position that is ``shape`` with C=1; for an existing
        position it is read from the image, so the label always matches the
        image it annotates. Labels that already exist are left untouched, so
        calling this again is a no-op. No data is written (fill value 0); fill
        a label later with ``position.get_label(name)["0"][...] = data``.
        Defaults to None (no labels).
    label_dtype : DTypeLike, optional
        Integer data type of the label images. Defaults to np.uint32.
    label_chunks : tuple[int, ...], optional
        TCZYX chunk size of the label images (C must be 1), clamped to the
        shape of each level. If None, the version-specific default the image
        uses when ``chunks`` is None, computed for each position and level
        from the image shape at ``label_dtype`` (for "0.4" the Z chunk is
        capped to 500 MB at ``label_dtype``). Explicit image ``chunks`` are
        not inherited. Defaults to None.
    label_shards : tuple[int, ...] or str, optional
        Shard geometry of the label images, in the forms ``shards`` takes
        (an explicit TCZYX shape in elements, ``"XY"``/``"XYZ"``, or a size
        target), resolved per level of each position's label. If None, for
        "0.5" one shard per ZYX volume (``"XYZ"``); no sharding for "0.4".
        The image's ``shards`` are not inherited. Defaults to None.
    label_shards_ratio : tuple[int, ...], optional
        Deprecated TCZYX multiplier over ``label_chunks``; use
        ``label_shards``. Passing both raises ``ValueError``. Defaults to None.

    Raises
    ------
    ValueError
        If ``metadata_keys`` is given without ``metadata_sources``.
    ValueError
        If ``label_dtype`` is not an integer type, or ``label_chunks`` or
        ``label_shards_ratio`` is not TCZYX (5 values) with C 1, or
        ``label_shards`` / ``label_shards_ratio`` is given for version "0.4",
        or both are given.
    ValueError
        If ``label_dtype``, ``label_chunks``, ``label_shards`` or
        ``label_shards_ratio`` is given without ``label_names``: they
        describe labels, so on their own they select nothing.
    FileNotFoundError
        If a ``metadata_sources`` plate root does not exist.

    Examples
    --------
    Create a new plate with positions and channels:
    >>> create_empty_plate(
    ...     store_path=Path("/path/to/store"),
    ...     position_keys=[("A", "1", "0"), ("A", "1", "1")],
    ...     channel_names=["DAPI", "FITC"],
    ...     shape=(1, 1, 256, 256, 256),
    ... )

    Create a plate with custom chunk size and scale:
    >>> create_empty_plate(
    ...     store_path=Path("/path/to/store"),
    ...     position_keys=[("A", "1", "0")],
    ...     channel_names=["DAPI"],
    ...     shape=(1, 1, 256, 256, 256),
    ...     chunks=(1, 1, 128, 128, 128),
    ...     scale=(1, 1, 0.5, 0.5, 0.5),
    ... )

    Create a plate whose shards hold as much as fits in ~2 GB files:
    >>> create_empty_plate(
    ...     store_path=Path("/path/to/store"),
    ...     position_keys=[("A", "1", "0")],
    ...     channel_names=["DAPI"],
    ...     shape=(100, 1, 64, 2048, 2048),
    ...     chunks=(1, 1, 8, 128, 128),
    ...     scale=(1, 1, 0.5, 0.5, 0.5),
    ...     shards="2GB",
    ...     version="0.5",
    ... )

    Create a plate with one shard per ZYX volume:
    >>> create_empty_plate(
    ...     store_path=Path("/path/to/store"),
    ...     position_keys=[("A", "1", "0")],
    ...     channel_names=["DAPI"],
    ...     shape=(1, 1, 64, 2048, 2048),
    ...     chunks=(1, 1, 8, 128, 128),
    ...     shards="XYZ",
    ...     version="0.5",
    ... )

    Create a plate copying metadata from an input plate:
    >>> create_empty_plate(
    ...     store_path=Path("/path/to/output.zarr"),
    ...     position_keys=[("A", "1", "0")],
    ...     channel_names=["DAPI"],
    ...     shape=(1, 1, 256, 256, 256),
    ...     metadata_sources=Path("/path/to/input.zarr"),
    ... )

    Copy only the provenance keys, leaving a bulky instrument blob behind:
    >>> create_empty_plate(
    ...     store_path=Path("/path/to/output.zarr"),
    ...     position_keys=[("A", "1", "0")],
    ...     channel_names=["DAPI"],
    ...     shape=(1, 1, 256, 256, 256),
    ...     metadata_sources=Path("/path/to/input.zarr"),
    ...     metadata_keys={"provenance-*"},
    ... )

    Inherit the input's provenance and record this step's own alongside it:
    >>> create_empty_plate(
    ...     store_path=Path("/path/to/output.zarr"),
    ...     position_keys=[("A", "1", "0")],
    ...     channel_names=["DAPI"],
    ...     shape=(1, 1, 256, 256, 256),
    ...     metadata_sources=Path("/path/to/input.zarr"),
    ...     metadata_keys={"provenance-*"},
    ...     extra_metadata={"provenance-deskew": {"ls_angle_deg": 30}},
    ... )

    Scaffold an empty segmentation label next to the images:
    >>> create_empty_plate(
    ...     store_path=Path("/path/to/output.zarr"),
    ...     position_keys=[("A", "1", "0")],
    ...     channel_names=["DAPI"],
    ...     shape=(1, 1, 256, 256, 256),
    ...     label_names=["nuclei"],
    ... )
    >>> with open_ome_zarr("/path/to/output.zarr/A/1/0", mode="r+") as position:
    ...     position.get_label("nuclei")["0"][0] = segmentation  # ZYX

    Notes
    -----
    - If `chunks` is not provided, a version-specific default is used (see
    parameter description).
    - The function ensures that positions and channels are appended to an
    existing plate if they are not already present.
    """
    if chunks is None:
        chunks = _default_chunks(shape, dtype, version)

    if shards is None and shards_ratio is None and version == "0.5":
        shards = "XYZ"

    extra_metadata = _validated_extra_metadata(extra_metadata)

    label_names = list(dict.fromkeys(label_names or []))
    if label_names:
        _validate_label_layout(label_dtype, label_chunks, label_shards, label_shards_ratio, version)
    else:
        given = [
            arg_name
            for arg_name, value, default in (
                ("label_dtype", np.dtype(label_dtype), np.dtype(np.uint32)),
                ("label_chunks", label_chunks, None),
                ("label_shards", label_shards, None),
                ("label_shards_ratio", label_shards_ratio, None),
            )
            if value != default
        ]
        if given:
            raise ValueError(
                f"{', '.join(given)} describe the label images, so they select nothing on their own. "
                "Pass label_names, or drop them."
            )

    # Normalize to a list of Paths. Fail loudly if any metadata source root
    # is wrong; missing individual positions within them are still skipped
    # gracefully in the loop below.
    if metadata_sources is None:
        if metadata_keys is not None:
            raise ValueError(
                "metadata_keys filters what metadata_sources contributes, "
                "so it selects nothing on its own. Pass metadata_sources, "
                "or drop metadata_keys."
            )
        metadata_sources = []
    elif isinstance(metadata_sources, (str, Path)):
        metadata_sources = [Path(metadata_sources)]
    else:
        metadata_sources = [Path(source) for source in metadata_sources]
    for source in metadata_sources:
        if not source.exists():
            raise FileNotFoundError(f"metadata_sources source plate not found at {source}")

    # Create plate
    output_plate = open_ome_zarr(
        str(store_path),
        layout="hcs",
        mode="a",
        channel_names=channel_names,
        version=version,
    )

    def _finalize_position(position, collected: dict[str, Any]) -> None:
        # Check if channel_names are already in the store, if not append them
        for channel_name in channel_names:
            metadata_channel_names = position.channel_names
            if channel_name not in metadata_channel_names:
                position.append_channel(channel_name, resize_arrays=True)

        # One custom-metadata write per position, and it comes last: both
        # create_zeros and append_channel may have rewritten the OME keys, so
        # the base dict is re-read here rather than reused from before them.
        # `collected` is empty unless this call created the position;
        # `extra_metadata` applies either way and wins on a collision, because
        # the caller is the authority on its own step.
        if collected or extra_metadata:
            position.zattrs.put({**dict(position.zattrs), **collected, **extra_metadata})

        _create_missing_labels(position, label_names, label_dtype, label_chunks, label_shards, label_shards_ratio)

    def _init_new_position(position_key_string: str, position) -> None:
        _ = position.create_zeros(
            name="0",
            shape=shape,
            chunks=chunks,
            shards=shards,
            shards_ratio=shards_ratio,
            dtype=dtype,
            transform=[TransformationMeta(type="scale", scale=scale)],
        )

        # Copy per-position custom (non-OME) zattrs from source plate(s).
        # Only for newly created positions; pre-existing ones are left
        # as-is. A key is only copied if it is not already present on the
        # destination, so earlier sources take precedence over later ones.
        #
        # Collect across all sources first, then write once.
        collected: dict[str, Any] = {}
        existing = dict(position.zattrs)
        for source in metadata_sources:
            # Only the open is guarded: a source that lacks this position
            # is skipped, but a failure while reading one is a real error.
            try:
                src_pos = open_ome_zarr(source / position_key_string, layout="fov", mode="r")
            except FileNotFoundError:
                continue
            with src_pos:
                src_attrs = dict(src_pos.zattrs)
            for k in _selected_metadata_keys(src_attrs, metadata_keys):
                if k not in existing and k not in collected:
                    collected[k] = src_attrs[k]
        _finalize_position(position, collected)

    def _get_position(position_key_string: str):
        """Open an existing position, or return None if it is not in the store."""
        try:
            return output_plate[position_key_string]
        except KeyError:
            return None

    # Normalize, then de-duplicate while keeping order, so keys that resolve
    # to the same group (e.g. "A" and "A/") are created only once.
    keys = list(dict.fromkeys(tuple(normalize_path(str(part)) for part in k) for k in position_keys))
    key_strings = ["/".join(k) for k in keys]

    # Positions are independent once their well metadata exists, so the
    # per-position work (array creation, source metadata reads, zattrs writes)
    # runs on a thread pool. Only well metadata is shared between positions;
    # `create_positions` writes it once per well rather than once per position,
    # which keeps creating thousands of FOVs in one well linear instead of
    # quadratic.
    with ThreadPoolExecutor(max_workers=max(1, min(32, len(keys)))) as pool:
        existing = list(pool.map(_get_position, key_strings))
        new = [(k, s) for k, s, pos in zip(keys, key_strings, existing, strict=True) if pos is None]

        created = output_plate.create_positions([k for k, _ in new]) if new else []
        futures = [pool.submit(_init_new_position, s, pos) for (_, s), pos in zip(new, created, strict=True)]
        futures += [pool.submit(_finalize_position, pos, {}) for pos in existing if pos is not None]
        for future in as_completed(futures):
            future.result()

    output_plate.close()


def _validate_label_layout(
    label_dtype: DTypeLike,
    label_chunks: tuple[int, ...] | None,
    label_shards: ShardsLike,
    label_shards_ratio: tuple[int, ...] | None,
    version: Literal["0.4", "0.5"],
) -> None:
    if not np.issubdtype(label_dtype, np.integer):
        raise ValueError(f"label_dtype must be an integer type, got {np.dtype(label_dtype)}.")
    for arg_name, value in (("label_chunks", label_chunks), ("label_shards_ratio", label_shards_ratio)):
        if value is None:
            continue
        if len(value) != 5:
            raise ValueError(f"{arg_name} must be TCZYX (5 values), got {value}.")
        if value[1] != 1:
            raise ValueError(f"{arg_name} must have 1 on the channel axis (a label has one channel), got {value}.")
    if version == "0.4" and (label_shards is not None or label_shards_ratio is not None):
        raise ValueError("label_shards / label_shards_ratio are not supported for OME-Zarr v0.4.")
    if label_shards is not None and label_shards_ratio is not None:
        raise ValueError("Pass either label_shards or the deprecated label_shards_ratio, not both.")


def _create_missing_labels(
    position,
    label_names: list[str],
    label_dtype: DTypeLike,
    label_chunks: tuple[int, ...] | None,
    label_shards: ShardsLike,
    label_shards_ratio: tuple[int, ...] | None,
) -> None:
    """Create the label images of ``label_names`` the position does not have yet.

    Existing labels are never touched, which also makes a repeated call a
    no-op. Shapes and transforms come from the position's own image, level by
    level, so a label added to an existing plate matches the image pyramid it
    annotates.
    """
    missing = [name for name in label_names if name not in position.label_names()]
    if not missing:
        return

    def _label_transforms(transforms: list[TransformationMeta] | None) -> list[TransformationMeta] | None:
        if transforms is None:
            return None
        # A relative `path` would no longer resolve from labels/<name>. iohub
        # never writes these, so refuse rather than copy a transform that
        # does not describe the label.
        if any(transform.path is not None for transform in transforms):
            raise ValueError(
                f"Cannot add labels to position '{position.zgroup.path}': its image uses "
                "path-backed coordinate transformations, which do not resolve from the label."
            )
        # The label has the image's axes (its channel axis has length 1), so
        # its transforms are the image's.
        return [transform.model_copy(deep=True) for transform in transforms]

    # The two transform scopes are kept apart: dataset transforms apply
    # first and multiscale-level ones after them, so flattening them
    # into the label's dataset transforms would reorder non-commuting
    # operations (a dataset scale and a multiscale translation).
    multiscale = position.metadata.multiscales[0]
    multiscale_transforms = _label_transforms(multiscale.coordinate_transformations)
    levels = []
    for dataset in multiscale.datasets:
        label_transform = _label_transforms(dataset.coordinate_transformations)
        image_shape = position[dataset.path].shape
        label_shape = position.label_shape(dataset.path)
        # Omitted layouts get the image defaults, computed per position
        # and level: an existing position's image need not have this
        # call's shape. The shards follow the label chunks, not the
        # image's ratio, which was sized for the image chunks.
        if label_chunks is None:
            chunks_tzyx = _clamp_chunks_to_shape(
                label_shape, _default_chunks(image_shape, label_dtype, position.version)
            )
        else:
            chunks_tzyx = _clamp_chunks_to_shape(label_shape, label_chunks)
        # The caller's label_shards, else one shard per ZYX volume on v0.5
        # (unless the deprecated ratio was given), else unsharded.
        if label_shards is not None:
            shards_tzyx = label_shards
        else:
            shards_tzyx = "XYZ" if label_shards_ratio is None and position.version == "0.5" else None
        levels.append((dataset.path, label_shape, chunks_tzyx, shards_tzyx, label_shards_ratio, label_transform))
    for name in missing:
        label = None
        for path, label_shape, chunks_tzyx, shards_tzyx, shards_ratio_tzyx, label_transform in levels:
            if label is None:
                label = position.create_label_zeros(
                    name,
                    shape=label_shape,
                    dtype=label_dtype,
                    chunks=chunks_tzyx,
                    shards=shards_tzyx,
                    shards_ratio=shards_ratio_tzyx,
                    transform=label_transform,
                    level=path,
                )
            else:
                label.create_zeros(
                    path,
                    label_shape,
                    label_dtype,
                    chunks=chunks_tzyx,
                    shards=shards_tzyx,
                    shards_ratio=shards_ratio_tzyx,
                    transform=label_transform,
                )
        if multiscale_transforms:
            label.metadata.multiscales[0].coordinate_transformations = multiscale_transforms
            label.dump_meta()


#: Attribute keys the OME-Zarr spec owns on a label image group: ``ome`` (v0.5), and
#: the top-level v0.4 keys. Refused as ``label_metadata`` keys.
_LABEL_OME_KEYS = {"ome", "multiscales", "image-label", "version"}


def create_empty_labels(
    store_path: Path,
    position_keys: list[tuple[str, str, str]],
    label_names: list[str],
    label_dtype: DTypeLike = np.uint32,
    label_chunks: tuple[int, ...] | None = None,
    label_shards: ShardsLike = None,
    label_shards_ratio: tuple[int, ...] | None = None,
    label_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Create empty label images in existing positions of an HCS plate.

    The label-image half of `create_empty_plate` for a plate that already has
    its positions: the step that segments a plate it reads scaffolds its
    ``labels/<name>`` images without restating the plate's channels, shape or
    scale, which each label takes from its own position's image. Labels that
    already exist are left untouched, so calling this again is a no-op.

    Parameters
    ----------
    store_path : Path
        Path to the existing HCS plate.
    position_keys : list[tuple[str, str, str]]
        Positions (row, column, fov) to create the labels in. Every one must
        exist; use `create_empty_plate` to create positions.
    label_names : list[str]
        Names of the label images, at ``<fov>/labels/<name>/``. Each mirrors the
        position's image pyramid: one level per image level, with the image's
        axes, scale (and translation, if any) and shape with one channel,
        ``(T, 1, Z, Y, X)``. No data is written (fill value 0).
    label_dtype : DTypeLike, optional
        Integer data type of the label images. Defaults to np.uint32.
    label_chunks : tuple[int, ...], optional
        TCZYX chunk size (C must be 1), clamped to the shape of each level. If
        None, the version-specific image default at ``label_dtype``. Defaults
        to None.
    label_shards : tuple[int, ...] or str, optional
        Shard geometry of the label images, in the forms ``create_empty_plate``'s
        ``shards`` takes (an explicit TCZYX shape in elements, ``"XY"``/``"XYZ"``,
        or a size target), resolved per level. If None, one shard per ZYX
        volume on "0.5"; no sharding on "0.4". Defaults to None.
    label_shards_ratio : tuple[int, ...], optional
        Deprecated TCZYX multiplier over ``label_chunks``; use ``label_shards``.
        Not supported for "0.4". Defaults to None.
    label_metadata : Mapping[str, Any], optional
        The caller's own metadata for the label images, written as top-level
        attributes of every named label group in every position, existing
        labels included, with the caller's entries winning over what is there:
        the step's provenance, e.g.
        ``{"biahub-segment": settings.model_dump()}``, the same spelling as
        ``extra_metadata`` on a position. Keys the OME-Zarr spec owns on a
        label are refused. Defaults to None.

    Raises
    ------
    FileNotFoundError
        If the plate or one of the positions does not exist.
    ValueError
        If ``label_dtype`` is not an integer type, ``label_chunks`` or
        ``label_shards_ratio`` is not TCZYX with C 1, ``label_shards_ratio`` is
        given for a "0.4" plate, or ``label_metadata`` uses a reserved key.

    Examples
    --------
    >>> create_empty_labels(
    ...     store_path=Path("/path/to/plate.zarr"),
    ...     position_keys=[("A", "1", "0")],
    ...     label_names=["nuclei"],
    ...     label_chunks=(16, 1, 1, 128, 128),
    ...     label_metadata={"provenance-segment": {"model": "cpsam_v2"}},
    ... )
    >>> with open_ome_zarr("/path/to/plate.zarr/A/1/0", mode="r+") as position:
    ...     position.get_label("nuclei")["0"][0] = segmentation  # ZYX
    """
    label_names = list(dict.fromkeys(label_names))
    if not label_names:
        raise ValueError("label_names must name at least one label image.")
    if label_metadata is None:
        label_metadata = {}
    elif not isinstance(label_metadata, Mapping):
        raise TypeError(f"label_metadata must be a mapping, got {type(label_metadata).__name__}.")
    reserved = _LABEL_OME_KEYS.intersection(label_metadata)
    if reserved:
        raise ValueError(
            f"label_metadata keys {sorted(reserved)} are reserved OME-Zarr keys on a label image. "
            "Use a namespaced key (e.g. '<package>-<step>') instead."
        )
    label_metadata = dict(label_metadata)

    keys = list(dict.fromkeys(tuple(normalize_path(str(part)) for part in k) for k in position_keys))
    with open_ome_zarr(str(store_path), layout="hcs", mode="r+") as plate:
        positions = []
        for key in keys:
            try:
                positions.append(plate["/".join(key)])
            except KeyError:
                raise FileNotFoundError(
                    f"Position {'/'.join(key)} does not exist in {store_path}; create_empty_plate creates positions."
                ) from None
        for position in positions:
            _validate_label_layout(label_dtype, label_chunks, label_shards, label_shards_ratio, position.version)

        def _add_labels(position) -> None:
            _create_missing_labels(position, label_names, label_dtype, label_chunks, label_shards, label_shards_ratio)
            if label_metadata:
                for name in label_names:
                    label = position.get_label(name)
                    label.zattrs.put({**dict(label.zattrs), **label_metadata})

        with ThreadPoolExecutor(max_workers=max(1, min(32, len(positions)))) as pool:
            for future in as_completed([pool.submit(_add_labels, position) for position in positions]):
                future.result()


# -- Transform helpers -----------------------------------------------------


def _apply_transform_to_czyx(
    func: Callable[[NDArray, Any], NDArray],
    input_position_path: Path,
    input_channel_indices: list[int] | slice,
    input_time_index: int,
    **kwargs,
) -> NDArray | None:
    all_func_params = inspect.signature(func).parameters.keys()
    if "input_time_index" in all_func_params:
        kwargs["input_time_index"] = input_time_index

    click.echo(f"Processing t={input_time_index}, c={input_channel_indices}")
    with open_ome_zarr(input_position_path, layout="fov", mode="r") as input_dataset:
        czyx_data = input_dataset.data.oindex[input_time_index, input_channel_indices]
    if not _check_nan_n_zeros(czyx_data):
        return func(czyx_data, **kwargs)
    else:
        return None


def _echo_finished(
    time_index: int | list[int] | slice,
    channel_index: int | list[int] | slice,
    skipped: bool,
) -> None:
    if skipped:
        click.echo(f"Skipping t={time_index}, c={channel_index} due to all zeros or nans")
    else:
        click.echo(f"Finished writing t={time_index}, c={channel_index}")


def _contiguous_runs(indices: Sequence[int]) -> list[slice]:
    """Split ``indices`` into maximal runs of consecutive values.

    Returns slices *into* ``indices`` rather than the values themselves, so one
    call carves up both an index list and the data written at those indices.
    """
    runs: list[slice] = []
    start = 0
    for position in range(1, len(indices) + 1):
        if position == len(indices) or indices[position] != indices[position - 1] + 1:
            runs.append(slice(start, position))
            start = position
    return runs


def _output_array(output_dataset, output_label: str | None):
    """The array a write targets: the position's image, or one of its label images.

    A label image has the image's axes with one channel, so a label write spans
    its whole channel axis; `plan_write_unit` and the writes below pass
    ``slice(None)`` where the image write passes its channel indices.
    """
    if output_label is None:
        return output_dataset.data
    if output_label not in output_dataset.label_names():
        raise ValueError(
            f"Label image {output_label!r} does not exist in the output position "
            f"(labels: {output_dataset.label_names()}); create it first, e.g. with "
            "create_empty_plate(label_names=[...])."
        )
    return output_dataset.get_label(output_label)["0"]


def _save_transformed(
    transformed: list[NDArray],
    output_position_path: Path,
    output_channel_indices: list[int] | slice,
    output_time_indices: list[int],
    write_unit: WriteUnit | None = None,
    output_label: str | None = None,
) -> None:
    """Write the transformed timepoints, one run of consecutive timepoints at a time.

    The write has to be split because a *gapped* time selection is not
    expressible as a slice, and a sharded write cannot be done without one:
    the zarrs (Rust) codec pipeline rejects the selection with
    ``DiscontiguousArrayError`` and falls back to a zarr-python path that
    mismaps the value buffer onto the shard — indexing it with coordinates from
    the shard's own space, which raises ``IndexError`` or, on a large array,
    asks numpy for a preposterous allocation.

    Gaps arise routinely once the output is sharded along T, because
    `process_single_position` then batches a whole shard's worth of timepoints
    into one write and `apply_transform_to_tczyx_and_save` drops the ones whose
    input was
    all zeros or NaNs from the middle of that batch. They also arise from a
    caller asking for non-consecutive ``output_time_indices`` outright.

    With ``output_label`` the target is the one-channel label image of that
    name: each transformed timepoint must then have exactly one channel, which
    lands on the label's channel axis.
    """
    if output_label is not None:
        channels = {np.shape(czyx)[0] for czyx in transformed}
        if channels != {1}:
            raise ValueError(
                f"Writing into label image {output_label!r} needs func to return one channel "
                f"per timepoint (a label has one channel), got {sorted(channels)} channels."
            )
    if output_label is not None:
        # The label's one channel, whatever a direct caller passed: this is the
        # selection `_plan_output_write` planned the write unit for.
        output_channel_indices = slice(None)
    with open_ome_zarr(output_position_path, layout="fov", mode="r+") as output_dataset:
        arr = _output_array(output_dataset, output_label)
        if write_unit is not None:
            write_unit.begin()
        for run in _contiguous_runs(output_time_indices):
            arr._impl.write_oindex(
                arr.native,
                (output_time_indices[run], output_channel_indices),
                transformed[run],
            )
    # Recorded only after the store is closed, so the codec pipeline has
    # released the shards and anything it buffered in user space has reached
    # the kernel before `WriteUnit.complete` syncs it to disk. Marking the
    # unit finished inside the block above would leave that buffered tail
    # outside the barrier, which is the whole point of the barrier.
    if write_unit is not None:
        write_unit.complete()


def _plan_output_write(
    output_position_path: Path,
    output_channel_indices: list[int] | slice,
    output_time_indices: list[int],
    resume: bool,
    resume_token: str,
    output_label: str | None = None,
) -> tuple[WriteUnit | None, bool]:
    """Describe the files this write owns, and whether it already finished."""
    with open_ome_zarr(output_position_path, layout="fov", mode="r") as output_dataset:
        array = _output_array(output_dataset, output_label)
        progress_dir = progress_dir_for(output_position_path)
        if output_label is not None and progress_dir is not None:
            # Markers of a label image live beside the image's, under the label's name.
            progress_dir = progress_dir / "labels" / output_label
        unit = plan_write_unit(
            array,
            output_time_indices,
            slice(None) if output_label is not None else output_channel_indices,
            token=resume_token,
            progress_dir=progress_dir,
        )
        if unit is None:
            return None, False
        return unit, bool(resume and unit_is_complete(unit, array))


def apply_transform_to_tczyx_and_save(
    func: Callable[[NDArray, Any], NDArray],
    input_position_path: Path,
    output_position_path: Path,
    input_channel_indices: list[int] | slice,
    output_channel_indices: list[int] | slice,
    input_time_indices: list[int] | slice,
    output_time_indices: list[int] | slice,
    *,
    resume: bool = False,
    resume_token: str = "",
    output_label: str | None = None,
    **kwargs,
) -> None:
    """Load a TCZYX array from a position store.

    Apply a transformation and save the result.

    When the write owns its output files outright (see
    `iohub.ngff._write_units`) they are cleared before writing rather
    than written over, and completion is recorded in a marker file. Pass
    ``resume=True`` to skip a unit that a previous run already finished.
    With ``output_label`` the result goes into that label image of the output
    position instead of its image (see `process_single_position`).
    """
    unit, already_written = _plan_output_write(
        output_position_path,
        output_channel_indices,
        _slice_to_list(output_time_indices),
        resume,
        resume_token,
        output_label=output_label,
    )
    if already_written:
        click.echo(f"Skipping t={output_time_indices}, c={output_channel_indices}: already written")
        return

    input_time_indices = _slice_to_list(input_time_indices)
    results = {}
    for i, input_time_index in enumerate(input_time_indices):
        result = _apply_transform_to_czyx(
            func,
            input_position_path=input_position_path,
            input_channel_indices=input_channel_indices,
            input_time_index=input_time_index,
            **kwargs,
        )
        if result is not None:
            results[i] = result
        else:
            _echo_finished(
                time_index=input_time_index,
                channel_index=input_channel_indices,
                skipped=True,
            )
    if results:
        output_time_indices = _slice_to_list(output_time_indices)
        output_time_indices = [output_time_indices[i] for i in results.keys()]
        _save_transformed(
            transformed=list(results.values()),
            output_position_path=output_position_path,
            output_channel_indices=output_channel_indices,
            output_time_indices=output_time_indices,
            write_unit=unit,
            output_label=output_label,
        )
        _echo_finished(input_time_indices, input_channel_indices, skipped=False)
    elif unit is not None:
        # Every timepoint was skipped, so there is nothing to write. Clear the
        # unit's shards anyway, then record it as finished claiming none, so a
        # re-run leaves what a fresh run would: a unit whose input has become
        # all-zero must not keep serving an earlier run's output. Skipping the
        # clear would make that outcome depend on whether a *sibling* timepoint
        # in the same shard happened to produce data, since the writing branch
        # clears unconditionally.
        unit.begin()
        unit.complete(wrote=False)
    del results


# -- Shard-aligned batching helpers ----------------------------------------


def _indices_to_shard_aligned_batches(indices: Sequence[int], shard_size: int) -> list[list[int]]:
    """Split indices into batches that are in the same shards."""
    indices = sorted(indices)
    batches = defaultdict(list)
    for index in indices:
        if index < 0:
            raise ValueError(f"Negative indices are not supported: {indices}")
        batches[index // shard_size].append(index)
    return list(batches.values())


def _match_indices_to_batches(
    flat_indices: Sequence[int],
    original_reference: Sequence[int],
    batched_reference: list[list[int]],
) -> list[list[int]]:
    """Match flat indices to batches based on a reference pair."""
    matched_batches = []
    for batch in batched_reference:
        matched_batch = [flat_indices[original_reference.index(index)] for index in batch]
        matched_batches.append(matched_batch)
    return matched_batches


def _slice_to_list(indices: list[int] | slice) -> list[int]:
    if isinstance(indices, slice):
        start = indices.start or 0
        step = indices.step or 1
        return list(range(start, indices.stop, step))
    return indices


def _available_cpus() -> int:
    """Return the CPU count the current process is allowed to use.

    Slurm exports ``SLURM_CPUS_PER_TASK`` for tasks that ask for more than
    one CPU, which reflects the cgroup CPU allocation rather than the
    host's total CPU count. Honouring it here prevents oversubscribing
    the cgroup when ``os.cpu_count()`` reports the whole node (e.g. 128)
    while slurm only granted us a few cores.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return int(slurm_cpus)
    return os.cpu_count() or 1


def process_single_position(
    func: Callable[[NDArray, Any], NDArray],
    input_position_path: Path,
    output_position_path: Path,
    input_channel_indices: list[slice] | list[list[int]] | None = None,
    output_channel_indices: list[slice] | list[list[int]] | None = None,
    input_time_indices: list[int] | None = None,
    output_time_indices: list[int] | None = None,
    num_workers: int = 1,
    use_threads: bool = False,
    resume: bool = False,
    resume_token: str = "",
    output_label: str | None = None,
    **kwargs,
) -> None:
    """
    Apply function to data in an `iohub` position store.

    Parallelize over time and channel indices and save result in an output position store.

    Parameters
    ----------
    func : Callable[[NDArray, Any], NDArray]
        The function to be applied to the data.
        func must take the CZYX NDArray as the first argument and return
        a CZXY NDArray. Additional arguments are passed through **kwargs.
    input_position_path : Path
        The path to the input OME-Zarr position store
        (e.g., input_store_path.zarr/A/1/0).
    output_position_path : Path
        The path to the output OME-Zarr position store
        (e.g., output_store_path.zarr/A/1/0). May be the input position
        when writing into one of its label images (``output_label``).
    output_label : str, optional
        Write into the label image ``labels/<output_label>`` of the output
        position (created beforehand, e.g. by ``create_empty_plate``'s
        ``label_names``) instead of its image. A label image has one channel, so
        ``func`` must return one channel per timepoint and
        ``input_channel_indices`` must hold a single channel group;
        ``output_channel_indices`` is ignored. Resume and repair work as for
        the image, with the markers kept under the label's name.
        Defaults to None, the image.
    input_time_indices : list[int], optional
        If not provided, all timepoints will be processed.
    output_time_indices : list[int], optional
        The time indices to write to. Must match length of input_time_indices.
        Typically used for stabilization, which needs per timepoint processing.
        If not provided, input_time_indices will be used.
    input_channel_indices : Union[list[slice], list[list[int]]], optional
        The channel indices to process. Acceptable values:
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A list of lists of integers: [[0, 1, 2, 3, 4]].
        If empty, process all channels.
        Must match output_channel_indices if not empty.
        Defaults to None.
    output_channel_indices : Union[list[slice], list[list[int]]], optional
        The channel indices to write to. Acceptable values:
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A list of lists of integers: [[0, 1, 2, 3, 4]].
        If empty, write to all channels.
        Must match input_channel_indices if not empty.
        Defaults to None.
    num_workers : int, optional
        Number of simultaneous workers (processes or threads) per position.
        If <= 1, the work is performed serially in the calling process.
        Defaults to 1.
    use_threads : bool, optional
        If True, parallelize across threads via ``ThreadPoolExecutor``;
        otherwise spawn worker processes via ``ProcessPoolExecutor``.
        Defaults to False.
    resume : bool, optional
        If True, skip shard-aligned ``(time, channel)`` units that a previous
        run already finished, as recorded by the markers described in
        `iohub.ngff._write_units`. Intended for retrying a run that was
        interrupted, for example by Slurm preemption. Leave False when
        re-running with changed inputs or settings,
        since a finished unit is skipped without checking whether it would
        now produce different data. Defaults to False.
    resume_token : str, optional
        Fingerprint of whatever determines the output, typically the resolved
        settings of the calling step. Mixed into each unit's identity so that
        ``resume=True`` after a parameter change recomputes rather than
        skipping work that would now produce different data. Defaults to "".
    kwargs : dict, optional
        Additional arguments to pass to the function.
        An ``extra_metadata`` kwarg (a dict) can be passed to record per-step
        provenance at the FOV level, typically a single namespaced entry keyed
        by ``"<package>-<step>"``, e.g.
        ``extra_metadata={"biahub-deskew": settings.model_dump()}``.
        Each item is written as a separate top-level key on the output
        position's zattrs (*not* nested under an ``extra_metadata`` key), so
        successive steps accumulate sibling provenance keys instead of
        overwriting one shared key. If a key is already present on the output
        position (e.g. copied from the input by ``create_empty_plate``'s
        ``metadata_sources``) it is overwritten and a warning is raised.
        Reserved OME-Zarr keys (``ome``, ``multiscales``, ``omero``,
        ``labels``, ``version``) are rejected with a ``ValueError``.
    """
    click.echo(f"Function to be applied: \t{func}")
    click.echo(f"Input data path:\t{input_position_path}")
    click.echo(f"Output data path:\t{output_position_path}")
    if output_label is not None:
        click.echo(f"Output label image:\t{output_label}")

    # Get data shape and shard info
    with open_ome_zarr(input_position_path, layout="fov", mode="r") as input_dataset:
        input_data_shape = input_dataset.data.shape
    with open_ome_zarr(output_position_path, layout="fov", mode="r") as output_dataset:
        output_array = _output_array(output_dataset, output_label)
        output_shards = output_array.shards
        if resume and not tracking_available(output_array):
            warnings.warn(
                f"resume=True was requested but progress cannot be tracked for {output_position_path}, "
                "so every unit will be recomputed. Tracking requires a local Zarr v3 (OME-Zarr v0.5) store.",
                stacklevel=2,
            )

    # Process time indices
    if input_time_indices is None:
        input_time_indices = list(range(input_data_shape[0]))
    assert type(input_time_indices) is list, "input_time_indices must be a list"
    if output_time_indices is None:
        output_time_indices = input_time_indices
    if output_shards is not None:
        batched_output_time_indices = _indices_to_shard_aligned_batches(output_time_indices, output_shards[0])
        batched_input_time_indices = _match_indices_to_batches(
            flat_indices=input_time_indices,
            original_reference=output_time_indices,
            batched_reference=batched_output_time_indices,
        )
    else:
        batched_input_time_indices = [[i] for i in input_time_indices]
        batched_output_time_indices = [[i] for i in output_time_indices]

    # Process channel indices
    if input_channel_indices is None:
        input_channel_indices = [[c] for c in range(input_data_shape[1])]
        output_channel_indices = input_channel_indices
    assert type(input_channel_indices) is list, "input_channel_indices must be a list"
    if output_label is not None:
        # One label image takes one result per timepoint, whatever channels feed it.
        if len(input_channel_indices) != 1:
            raise ValueError(
                f"output_label={output_label!r} takes a single input channel group, "
                f"got {len(input_channel_indices)}: {input_channel_indices}."
            )
        output_channel_indices = [slice(None)]
    elif output_channel_indices is None:
        output_channel_indices = input_channel_indices
    if output_label is None and output_shards is not None and output_shards[1] != 1:
        raise ValueError("Sharding along the channel dimension is not supported.")

    # Check for invalid times
    time_ubound = input_data_shape[0] - 1
    if np.max(input_time_indices) > time_ubound:
        raise ValueError(f"""input_time_indices = {input_time_indices} includes
            a time index beyond the maximum index of
            the dataset = {time_ubound}""")

    # Write extra metadata to the output store. Each entry is stored as a
    # top-level zattrs key rather than nested under a single "extra_metadata"
    # key, so successive processing steps record their provenance as sibling
    # keys (e.g. "biahub-flat_field", "biahub-deskew") instead of overwriting
    # one shared key. Overwriting a pre-existing key (e.g. metadata copied by
    # create_empty_plate's metadata_sources) warns rather than silently
    # clobbering upstream provenance.
    extra_metadata = _validated_extra_metadata(kwargs.pop("extra_metadata", None))
    if extra_metadata:
        with open_ome_zarr(output_position_path, layout="fov", mode="r+") as output_dataset:
            for key, value in extra_metadata.items():
                if key in output_dataset.zattrs:
                    warnings.warn(
                        f"extra_metadata key {key!r} already exists on {output_position_path} and will be overwritten.",
                        stacklevel=2,
                    )
                output_dataset.zattrs[key] = value

    # Loop through (T, C), applying transform and writing as we go
    iterable = itertools.product(
        zip(input_channel_indices, output_channel_indices, strict=False),
        zip(batched_input_time_indices, batched_output_time_indices, strict=False),
    )
    flat_iterable = tuple((*c, *t) for c, t in iterable)

    partial_apply_transform_to_czyx_and_save = partial(
        apply_transform_to_tczyx_and_save,
        func,
        input_position_path,
        output_position_path,
        resume=resume,
        resume_token=resume_token,
        output_label=output_label,
        **kwargs,
    )
    num_workers = min(num_workers, len(flat_iterable), _available_cpus())
    if num_workers <= 1:
        click.echo("\nRunning serially in the calling process (num_workers <= 1)")
        for args in flat_iterable:
            partial_apply_transform_to_czyx_and_save(*args)
        click.echo("Done")
    elif use_threads:
        click.echo(f"\nStarting thread pool with {num_workers} threads")
        with ThreadPoolExecutor(max_workers=num_workers) as p:
            futures = [p.submit(partial_apply_transform_to_czyx_and_save, *args) for args in flat_iterable]
            for fut in as_completed(futures):
                fut.result()
        click.echo("Shut down thread pool")
    else:
        click.echo(f"\nStarting multiprocess pool with {num_workers} processes")
        # NOTE: spawn (not fork) — tensorstore runs internal C++ threads
        # that are not fork-safe, so a forked worker can deadlock or
        # segfault before our code runs. See google/tensorstore#61.
        # NOTE: ProcessPoolExecutor (not mp.Pool) so silent worker death
        # (e.g. cgroup OOM-kill) surfaces as BrokenProcessPool instead
        # of hanging indefinitely on pool.starmap.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as p:
            futures = [p.submit(partial_apply_transform_to_czyx_and_save, *args) for args in flat_iterable]
            for fut in as_completed(futures):
                fut.result()
        click.echo("Shut down multiprocess pool")


# -- Pure utility functions ------------------------------------------------


def _check_nan_n_zeros(input_array) -> bool:
    """Checks if any of the channels are all zeros or nans."""
    if input_array.ndim == 3:
        return _zyx_is_all_zero_or_nan(input_array)
    elif input_array.ndim == 4:
        return any(_zyx_is_all_zero_or_nan(input_array[c]) for c in range(input_array.shape[0]))
    else:
        raise ValueError("Input array must be 3D or 4D")


def _zyx_is_all_zero_or_nan(zyx_array) -> bool:
    """All-zero or all-NaN test that short-circuits on the first counter-example.

    `np.any(arr)` returns False iff every element is 0/False, and short-circuits
    in C as soon as it finds a truthy value. The previous `np.all(arr == 0)`
    materialised a full boolean mask of the input volume before reducing it.
    """
    if not np.any(zyx_array):
        return True  # all zeros
    # NaN is truthy in numpy bool context, so the explicit NaN check is only
    # needed when np.any returned True (otherwise the array is all zeros and
    # would not reach here).
    if zyx_array.dtype.kind == "f" and np.isnan(zyx_array).all():
        return True  # all NaN
    return False


def _limit_zyx_chunk_size(
    shape: tuple[int, ...],
    bytes_per_pixel: int,
    max_chunk_size_bytes: float,
    chunks: tuple[int, ...] | None = None,
) -> tuple[int, int, int]:
    """Calculate chunk size for ZYX dimensions to stay under max bytes."""
    chunk_zyx_shape = list(chunks[-3:] if chunks else shape[-3:])

    # Reduce Z chunk size until total chunk is within limit
    while chunk_zyx_shape[-3] > 1 and np.prod(chunk_zyx_shape) * bytes_per_pixel > max_chunk_size_bytes:
        chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2)
    chunk_zyx_shape = tuple(map(int, chunk_zyx_shape))
    return chunk_zyx_shape


def _clamp_chunks_to_shape(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    """Clamp chunk sizes so they don't exceed dimension sizes.

    Parameters
    ----------
    shape : tuple[int, ...]
        Dimension sizes for each dimension.
    chunks : tuple[int, ...]
        Chunk sizes for each dimension.

    Returns
    -------
    tuple[int, ...]
        Chunk sizes clamped to at most the dimension size.
    """
    return tuple(min(c, d) for c, d in zip(chunks, shape, strict=False))


def _default_chunks(
    shape: tuple[int, ...],
    dtype: DTypeLike,
    version: Literal["0.4", "0.5"],
) -> tuple[int, ...]:
    """Version-specific default TCZYX chunk size for a plate array."""
    if version == "0.4":
        chunk_zyx_shape = _limit_zyx_chunk_size(
            shape,
            np.dtype(dtype).itemsize,
            V04_MAX_CHUNK_SIZE_BYTES,
        )
        return (1, 1, *chunk_zyx_shape)
    # v0.5: DCA-aligned small chunks, clamped to the array shape.
    return _clamp_chunks_to_shape(shape, (1, 1, *_V05_DEFAULT_ZYX_CHUNKS))
