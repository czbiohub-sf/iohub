"""Unit tests for iohub.core.sharding (shard auto-sizing)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iohub.core.sharding import parse_shard_size, resolve_shard_shape, resolve_shards

#: The mantis-v2 assemble output from issue #458: the shape whose shard
#: geometry callers were computing by hand.
_MANTIS_SHAPE = (5, 6, 86, 1664, 1193)
_MANTIS_CHUNKS = (1, 1, 16, 256, 256)
_TCZYX = ["t", "c", "z", "y", "x"]


def _resolve(shards, shape=_MANTIS_SHAPE, chunks=_MANTIS_CHUNKS, dtype=np.uint16, names=_TCZYX):
    return resolve_shards(shards, shape=shape, chunks=chunks, dtype=dtype, dimension_names=names)


# ---------------------------------------------------------------------------
# Size string parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("1B", 1),
        ("4096", 4096),
        ("2KB", 2_000),
        ("2KiB", 2_048),
        ("512MB", 512_000_000),
        ("512MiB", 536_870_912),
        ("2GB", 2_000_000_000),
        ("2GiB", 2_147_483_648),
        ("1.5GB", 1_500_000_000),
        ("1TB", 10**12),
        ("  2 gb  ", 2_000_000_000),
        ("2gIb", 2_147_483_648),
    ],
)
def test_parse_shard_size(target, expected):
    assert parse_shard_size(target) == expected


@pytest.mark.parametrize("target", ["", "GB", "2 gigabytes", "-2GB", "2PB", "1e9", "0.4B", "2GB2"])
def test_parse_shard_size_rejects_garbage(target):
    with pytest.raises(ValueError, match=r"Cannot parse shard size|cannot hold a chunk"):
        parse_shard_size(target)


# ---------------------------------------------------------------------------
# Spatial extent keywords
# ---------------------------------------------------------------------------


def test_extent_xyz_covers_the_volume_in_whole_chunks():
    """The keyword reproduces the ratio callers were computing by hand:
    ceil(86/16), ceil(1664/256), ceil(1193/256) chunks in Z, Y, X."""
    assert _resolve("XYZ") == (1, 1, 96, 1792, 1280)


def test_extent_xy_covers_one_plane_stack():
    """Z stays at one chunk, so a shard is a chunk-thick YX slab."""
    assert _resolve("XY") == (1, 1, 16, 1792, 1280)


@pytest.mark.parametrize("keyword", ["XYZ", "xyz", "ZYX", "zyx", "xZy"])
def test_extent_keywords_are_order_and_case_insensitive(keyword):
    assert _resolve(keyword) == _resolve("XYZ")


def test_extent_is_exact_when_chunks_divide_the_shape():
    assert _resolve("XYZ", shape=(4, 2, 32, 512, 512), chunks=(1, 1, 16, 256, 256)) == (1, 1, 32, 512, 512)


def test_extent_needs_enough_axes():
    with pytest.raises(ValueError, match="needs at least 3 dimensions"):
        _resolve("XYZ", shape=(64, 64), chunks=(16, 16), names=["y", "x"])


def test_extent_keeps_channel_at_one_chunk():
    """C is never grown, even when its chunk size is 1 and the array has many."""
    assert _resolve("XYZ")[1] == 1


# ---------------------------------------------------------------------------
# Size targets
# ---------------------------------------------------------------------------


def _nbytes(shards, dtype=np.uint16):
    return math.prod(shards) * np.dtype(dtype).itemsize


def test_size_target_fills_space_then_time():
    """A 2 GB target buys the whole volume (440 MB) plus 3 more timepoints."""
    shards = _resolve("2GB")
    assert shards == (4, 1, 96, 1792, 1280)
    assert _nbytes(shards) <= 2_000_000_000


def test_size_target_stops_at_the_volume_when_time_does_not_fit():
    """512 MiB holds one 440 MB volume; a second timepoint would overshoot."""
    assert _resolve("512MiB") == (1, 1, 96, 1792, 1280)


def test_size_target_below_one_volume_grows_only_x_and_y():
    """A target under one volume fills X, then as much of Y as fits, and
    leaves Z at one chunk."""
    shards = _resolve("64MB")
    assert shards == (1, 1, 16, 1536, 1280)  # 6 of the 7 Y chunks
    assert _nbytes(shards) <= 64_000_000


def test_size_target_partially_fills_the_innermost_axis():
    """When not even a full X row fits, X grows to the multiple that does."""
    shards = _resolve("8MB")
    assert shards == (1, 1, 16, 256, 768)  # 3 of the 5 X chunks
    assert _nbytes(shards) <= 8_000_000


@pytest.mark.parametrize("target", ["4MB", "64MB", "512MB", "2GB", "1TB"])
def test_size_target_is_never_exceeded_and_is_chunk_aligned(target):
    shards = _resolve(target)
    assert _nbytes(shards) <= parse_shard_size(target)
    assert all(s % c == 0 for s, c in zip(shards, _MANTIS_CHUNKS, strict=True))
    assert all(s <= math.ceil(d / c) * c for s, d, c in zip(shards, _MANTIS_SHAPE, _MANTIS_CHUNKS, strict=True))


def test_size_target_never_grows_channels():
    """Channels are unshardable, so the budget is spent on other axes."""
    assert _resolve("1TB")[1] == 1


def test_size_target_larger_than_the_array_covers_it_whole():
    shards = _resolve("1TB")
    assert shards == (5, 1, 96, 1792, 1280)


def test_size_target_smaller_than_a_chunk_falls_back_to_one_chunk(caplog):
    with caplog.at_level("WARNING", logger="iohub.core.sharding"):
        shards = _resolve("1KB")
    assert shards == _MANTIS_CHUNKS
    assert "over the" in caplog.text


def test_size_target_accounts_for_dtype():
    """The same target buys half as many float32 elements as uint16 ones."""
    small = _resolve("2GB", dtype=np.float32)
    large = _resolve("2GB", dtype=np.uint16)
    assert small[0] * 2 == large[0]


def test_size_target_without_names_assumes_tczyx():
    """A 5D array without names is read as TCZYX, so C is still protected."""
    assert _resolve("2GB", names=None) == _resolve("2GB")


def test_size_target_on_a_named_tzyx_array_grows_time_last():
    """Label arrays are TZYX: axis 1 is Z, and only the named T grows last."""
    shards = resolve_shards(
        "2GB",
        shape=(5, 86, 1664, 1193),
        chunks=(1, 16, 256, 256),
        dtype=np.uint16,
        dimension_names=["t", "z", "y", "x"],
    )
    assert shards == (4, 96, 1792, 1280)


# ---------------------------------------------------------------------------
# Explicit shard shapes
# ---------------------------------------------------------------------------


def test_explicit_shape_passes_through():
    assert _resolve((2, 1, 96, 1792, 1280)) == (2, 1, 96, 1792, 1280)


def test_explicit_shape_rounds_up_to_whole_chunks(caplog):
    with caplog.at_level("WARNING", logger="iohub.core.sharding"):
        shards = _resolve((1, 1, 20, 300, 300))
    assert shards == (1, 1, 32, 512, 512)
    assert "whole number of" in caplog.text


def test_explicit_shape_may_exceed_the_array():
    """An explicit shape is taken as given; the trailing shard is just padded."""
    assert _resolve((16, 1, 96, 1792, 1280))[0] == 16


def test_explicit_shape_length_must_match():
    with pytest.raises(ValueError, match="Shard shape length 3 does not match shape length 5"):
        _resolve((1, 1, 16))


@pytest.mark.parametrize("shards", [(1, 1, 0, 256, 256), (1, 1, -16, 256, 256)])
def test_explicit_shape_must_be_positive(shards):
    with pytest.raises(ValueError, match="must be positive"):
        _resolve(shards)


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def test_none_means_no_sharding():
    assert _resolve(None) is None


def test_unknown_string_lists_the_accepted_forms():
    with pytest.raises(ValueError, match="Cannot interpret shards='volume'"):
        _resolve("volume")


def test_chunk_length_must_match_shape():
    with pytest.raises(ValueError, match="does not match shape length"):
        _resolve("XYZ", chunks=(1, 1, 16))


def test_lists_are_accepted_as_shard_shapes():
    assert _resolve([1, 1, 96, 1792, 1280]) == (1, 1, 96, 1792, 1280)


def test_a_bare_number_is_rejected():
    """An int is neither a shape nor a size target; say so instead of failing
    on iteration."""
    with pytest.raises(TypeError, match="got int"):
        _resolve(2_000_000_000)


# ---------------------------------------------------------------------------
# Deprecated shards_ratio
# ---------------------------------------------------------------------------


def _resolve_both(shards, shards_ratio):
    return resolve_shard_shape(
        shards,
        shards_ratio,
        shape=_MANTIS_SHAPE,
        chunks=_MANTIS_CHUNKS,
        dtype=np.uint16,
        dimension_names=_TCZYX,
    )


def test_shards_ratio_still_multiplies_chunks():
    with pytest.deprecated_call(match="shards_ratio is deprecated"):
        assert _resolve_both(None, (1, 1, 6, 7, 5)) == (1, 1, 96, 1792, 1280)


def test_shards_ratio_may_exceed_the_array():
    """The legacy form is unclamped, so existing geometries are preserved."""
    with pytest.deprecated_call():
        assert _resolve_both(None, (10, 1, 8, 8, 8)) == (10, 1, 128, 2048, 2048)


def test_shards_and_shards_ratio_conflict():
    with pytest.raises(ValueError, match="not both"):
        _resolve_both("2GB", (1, 1, 6, 7, 5))


def test_shards_ratio_length_must_match():
    with pytest.deprecated_call(), pytest.raises(ValueError, match="Sharding ratio length"):
        _resolve_both(None, (1, 1, 6))


def test_neither_argument_means_no_sharding():
    assert _resolve_both(None, None) is None


def test_empty_shards_ratio_is_ignored():
    """The legacy falsy-ratio path means "no sharding", without a warning."""
    assert _resolve_both(None, ()) is None
