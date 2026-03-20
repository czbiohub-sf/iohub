"""Tests for blender protocol and built-in implementations."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from iohub.tile import GaussianBlender, UniformBlender, get_blender


@given(
    h=st.integers(4, 128),
    w=st.integers(4, 256),
    blender=st.sampled_from([UniformBlender(), GaussianBlender()]),
)
def test_weight_shape_matches_input(h, w, blender):
    """Blender output shape must equal the requested tile shape."""
    w_arr = blender.weights((h, w), {})
    assert w_arr.shape == (h, w)
    assert w_arr.dtype == np.float64
    assert (w_arr > 0).all()


def test_gaussian_center_weighted():
    """Gaussian kernel has its maximum at the center."""
    g = GaussianBlender()
    w = g.weights((32, 64), {})
    assert w[16, 32] >= w[0, 0]
    assert w[16, 32] >= w[-1, -1]


def test_resolve_by_name_and_unknown():
    """get_blender resolves by name, passes through instances, rejects unknowns."""
    assert isinstance(get_blender("uniform"), UniformBlender)
    assert isinstance(get_blender("gaussian"), GaussianBlender)

    b = UniformBlender()
    assert get_blender(b) is b

    with pytest.raises(ValueError, match="Unknown blender"):
        get_blender("nonexistent")


# ---------------------------------------------------------------------------
# 3D weight kernel tests
# ---------------------------------------------------------------------------


@given(
    d=st.integers(4, 32),
    h=st.integers(4, 64),
    w=st.integers(4, 128),
    blender=st.sampled_from([UniformBlender(), GaussianBlender()]),
)
def test_3d_weight_shape(d, h, w, blender):
    """Blender output shape matches 3D tile shape."""
    w_arr = blender.weights((d, h, w), {})
    assert w_arr.shape == (d, h, w)
    assert w_arr.dtype == np.float64
    assert (w_arr > 0).all()


def test_gaussian_3d_center_weighted():
    """3D Gaussian kernel has its maximum at the center."""
    g = GaussianBlender()
    w = g.weights((8, 32, 64), {})
    assert w[4, 16, 32] >= w[0, 0, 0]
    assert w[4, 16, 32] >= w[-1, -1, -1]


def test_3d_gaussian_separability():
    """3D Gaussian is the outer product of three 1D gaussians."""
    g = GaussianBlender()
    # Use odd sizes so center is unambiguous
    w3d = g.weights((9, 17, 33), {})
    # Each axis slice through center should be a 1D gaussian
    center_z = w3d[:, 8, 16]
    center_y = w3d[4, :, 16]
    center_x = w3d[4, 8, :]
    # Max should be at center for each
    assert np.argmax(center_z) == 4
    assert np.argmax(center_y) == 8
    assert np.argmax(center_x) == 16
