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
