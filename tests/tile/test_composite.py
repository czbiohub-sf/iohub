"""Tests for _composite_fovs sweep-line FOV compositing."""

import dask.array
import numpy as np
import xarray as xr

from iohub.tile._composite import _composite_fovs
from iohub.tile._compositors import MaxCompositor, MeanCompositor


def _make_fov(data, y_offset=0.0, x_offset=0.0, pixel_size=1.0):
    """Create a 5D FOV xr.DataArray with physical coords.

    Uses pixel_size=1.0 by default to avoid float alignment issues
    in xr.concat (the compositors stack slices by coord values).
    """
    t, c, z, h, w = data.shape
    return xr.DataArray(
        data,
        dims=("t", "c", "z", "y", "x"),
        coords={
            "t": np.arange(t),
            "c": np.arange(c),
            "z": np.arange(z, dtype=np.float64) * pixel_size,
            "y": np.arange(h, dtype=np.float64) * pixel_size + y_offset,
            "x": np.arange(w, dtype=np.float64) * pixel_size + x_offset,
        },
    )


def test_single_fov_passthrough():
    """Single FOV returns the same object."""
    data = np.ones((1, 1, 1, 8, 8), dtype=np.float32)
    fov = _make_fov(data)
    result = _composite_fovs([fov], MeanCompositor())
    assert result is fov


def test_two_overlapping_fovs_mean():
    """Two FOVs overlapping in X, MeanCompositor averages the overlap."""
    # FOV 0: x=[0..7], FOV 1: x=[4..11] → overlap at x=[4..7]
    data_a = np.full((1, 1, 1, 8, 8), 2.0, dtype=np.float32)
    data_b = np.full((1, 1, 1, 8, 8), 6.0, dtype=np.float32)
    fov_a = _make_fov(data_a, x_offset=0.0)
    fov_b = _make_fov(data_b, x_offset=4.0)

    result = _composite_fovs([fov_a, fov_b], MeanCompositor())

    assert result.shape == (1, 1, 1, 8, 12)  # 8 + 8 - 4 overlap = 12 wide
    assert isinstance(result.data, dask.array.Array)

    vals = result.values
    # Left region (FOV A only): columns 0..3
    np.testing.assert_allclose(vals[..., :4], 2.0)
    # Overlap region: columns 4..7 → mean of 2 and 6 = 4
    np.testing.assert_allclose(vals[..., 4:8], 4.0)
    # Right region (FOV B only): columns 8..11
    np.testing.assert_allclose(vals[..., 8:], 6.0)


def test_two_overlapping_fovs_max():
    """Two FOVs overlapping in X, MaxCompositor takes the max."""
    data_a = np.full((1, 1, 1, 8, 8), 2.0, dtype=np.float32)
    data_b = np.full((1, 1, 1, 8, 8), 6.0, dtype=np.float32)
    fov_a = _make_fov(data_a, x_offset=0.0)
    fov_b = _make_fov(data_b, x_offset=4.0)

    result = _composite_fovs([fov_a, fov_b], MaxCompositor())

    vals = result.values
    np.testing.assert_allclose(vals[..., :4], 2.0)
    np.testing.assert_allclose(vals[..., 4:8], 6.0)  # max(2, 6) = 6
    np.testing.assert_allclose(vals[..., 8:], 6.0)


def test_non_overlapping_fovs_gap():
    """Two FOVs with a gap produce NaN in the gap region."""
    data_a = np.full((1, 1, 1, 4, 4), 1.0, dtype=np.float32)
    data_b = np.full((1, 1, 1, 4, 4), 2.0, dtype=np.float32)
    # FOV A at x=0..3, FOV B at x=8..11 → gap at x=4..7
    fov_a = _make_fov(data_a, x_offset=0.0)
    fov_b = _make_fov(data_b, x_offset=8.0)

    result = _composite_fovs([fov_a, fov_b], MeanCompositor())

    assert result.shape == (1, 1, 1, 4, 12)
    vals = result.values
    np.testing.assert_allclose(vals[..., :4], 1.0)
    assert np.all(np.isnan(vals[..., 4:8]))
    np.testing.assert_allclose(vals[..., 8:], 2.0)


def test_overlapping_fovs_y_direction():
    """Two FOVs overlapping in Y."""
    data_a = np.full((1, 1, 1, 8, 4), 10.0, dtype=np.float32)
    data_b = np.full((1, 1, 1, 8, 4), 20.0, dtype=np.float32)
    fov_a = _make_fov(data_a, y_offset=0.0)
    fov_b = _make_fov(data_b, y_offset=4.0)

    result = _composite_fovs([fov_a, fov_b], MeanCompositor())

    assert result.shape == (1, 1, 1, 12, 4)
    vals = result.values
    np.testing.assert_allclose(vals[..., :4, :], 10.0)
    np.testing.assert_allclose(vals[..., 4:8, :], 15.0)  # mean(10, 20)
    np.testing.assert_allclose(vals[..., 8:, :], 20.0)
