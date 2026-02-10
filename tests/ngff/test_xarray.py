"""Tests for Position.to_xarray() and Position.write_xarray()."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from typing import get_args

import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_allclose, assert_array_equal

from iohub.ngff.models import SpaceAxisMeta, TimeAxisMeta
from iohub.ngff.nodes import open_ome_zarr
from tests.ngff.test_ngff import (
    c_dim_st,
    t_dim_st,
    x_dim_st,
    y_dim_st,
    z_dim_st,
)

# channel_names_st from test_ngff allows any unicode; restrict to
# printable ASCII so names survive zarr serialization roundtrips.
_printable_text_st = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "S", "Z"), codec="ascii"),
    min_size=1,
    max_size=8,
)
channel_names_st = c_dim_st.flatmap(lambda c: st.lists(_printable_text_st, min_size=c, max_size=c, unique=True))

SHAPE_SMALL = (1, 1, 1, 8, 8)
DEFAULT_SCALES = {"t": 1.0, "z": 1.0, "y": 0.65, "x": 0.65}
DEFAULT_TRANSLATIONS = {"t": 0.0, "z": 0.0, "y": 0.0, "x": 0.0}
DEFAULT_COORD_UNITS = {
    "t": "second",
    "z": "micrometer",
    "y": "micrometer",
    "x": "micrometer",
}

positive_scale_st = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
translation_st = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Derive valid unit strings from the OME-NGFF model Literal types
_space_units = get_args(get_args(SpaceAxisMeta.model_fields["unit"].annotation)[0])
_time_units = get_args(get_args(TimeAxisMeta.model_fields["unit"].annotation)[0])
space_unit_st = st.sampled_from(_space_units)
time_unit_st = st.sampled_from(_time_units)
json_value_st = st.one_of(
    st.text(min_size=1, max_size=16),
    st.integers(-1000, 1000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
)
json_attrs_st = st.dictionaries(
    st.text(min_size=1, max_size=8),
    json_value_st,
    min_size=1,
    max_size=4,
)


@st.composite
def _xarray_roundtrip_params(draw):
    """Draw a valid (channel_names, shape, scales, translations) tuple."""
    channel_names = draw(channel_names_st)
    shape = (
        draw(t_dim_st),
        len(channel_names),
        draw(z_dim_st),
        draw(y_dim_st),
        draw(x_dim_st),
    )
    scales = {d: draw(positive_scale_st) for d in "tzyx"}
    translations = {d: draw(translation_st) for d in "tzyx"}
    return channel_names, shape, scales, translations


def _build_xarray(
    rng,
    channels,
    shape=SHAPE_SMALL,
    scales=None,
    translations=None,
    coord_units=None,
    attrs=None,
):
    """Build a tczyx DataArray from shape, scales, and translations."""
    T, C, Z, Y, X = shape[0], len(channels), shape[2], shape[3], shape[4]
    scales = scales or DEFAULT_SCALES
    translations = translations or DEFAULT_TRANSLATIONS
    coord_units = coord_units or DEFAULT_COORD_UNITS
    data = rng.random((T, C, Z, Y, X)).astype(np.float32)
    coords = {"c": ("c", channels)}
    for dim, idx in [("t", 0), ("z", 2), ("y", 3), ("x", 4)]:
        values = np.arange(shape[idx]) * scales[dim] + translations[dim]
        unit_attrs = {"units": coord_units[dim]} if dim in coord_units else {}
        coords[dim] = (dim, values, unit_attrs)
    return xr.DataArray(
        data,
        dims=("t", "c", "z", "y", "x"),
        coords=coords,
        attrs=attrs or {},
    )


def _make_position(rng, tmp_dir, channel_names, shape, scales=None, name="test.zarr"):
    """Create a Position with random float32 data and optional scales."""
    pos = open_ome_zarr(
        os.path.join(tmp_dir, name),
        layout="fov",
        mode="w-",
        channel_names=channel_names,
    )
    data = rng.random(shape).astype(np.float32)
    pos.create_image("0", data)
    for axis, val in (scales or {}).items():
        pos.set_scale("0", axis, val)
    return pos, data


@given(params=_xarray_roundtrip_params())
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_roundtrip_data_and_scales(rng, params):
    """write_xarray -> to_xarray roundtrips data and coordinates."""
    channel_names, shape, scales, translations = params
    xa = _build_xarray(rng, channel_names, shape, scales, translations)
    with TemporaryDirectory() as tmp:
        with open_ome_zarr(
            os.path.join(tmp, "rt.zarr"),
            layout="fov",
            mode="w-",
            channel_names=channel_names,
        ) as pos:
            pos.write_xarray(xa)
            xa2 = pos.to_xarray()
        assert_array_equal(xa.values, xa2.values)
        assert list(xa2.coords["c"].values) == channel_names
        for dim in "tzyx":
            assert_allclose(
                xa.coords[dim].values,
                xa2.coords[dim].values,
                atol=1e-6,
            )


@given(time_unit=time_unit_st, space_unit=space_unit_st)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_roundtrip_coordinate_units(rng, time_unit, space_unit):
    """Coordinate units survive write_xarray -> to_xarray for any valid OME unit."""
    coord_units = {"t": time_unit, "z": space_unit, "y": space_unit, "x": space_unit}
    xa = _build_xarray(rng, ["ch1"], SHAPE_SMALL, coord_units=coord_units)
    with TemporaryDirectory() as tmp:
        with open_ome_zarr(
            os.path.join(tmp, "u.zarr"),
            layout="fov",
            mode="w-",
            channel_names=["ch1"],
        ) as pos:
            pos.write_xarray(xa)
            xa2 = pos.to_xarray()
        assert xa2.coords["t"].attrs["units"] == time_unit
        for dim in "zyx":
            assert xa2.coords[dim].attrs["units"] == space_unit


@given(attrs=json_attrs_st)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_roundtrip_value_attrs(rng, attrs):
    """DataArray attrs survive write -> close -> reopen -> read."""
    xa = _build_xarray(rng, ["ch1"], (1, 1, 1, 4, 4), attrs=attrs)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v.zarr")
        with open_ome_zarr(
            path,
            layout="fov",
            mode="w-",
            channel_names=["ch1"],
        ) as pos:
            pos.write_xarray(xa)
        with open_ome_zarr(path, mode="r") as pos:
            xa2 = pos.to_xarray()
        for k, v in attrs.items():
            if isinstance(v, float):
                assert_allclose(xa2.attrs[k], v)
            else:
                assert xa2.attrs[k] == v


@given(params=_xarray_roundtrip_params())
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_write_channel_subset(rng, params):
    """Writing channels one at a time matches writing all at once."""
    channel_names, shape, scales, translations = params
    xa_all = _build_xarray(rng, channel_names, shape, scales, translations)
    with TemporaryDirectory() as tmp:
        with open_ome_zarr(
            os.path.join(tmp, "all.zarr"),
            layout="fov",
            mode="w-",
            channel_names=channel_names,
        ) as pos_all:
            pos_all.write_xarray(xa_all)
            result_all = pos_all.to_xarray()

        with open_ome_zarr(
            os.path.join(tmp, "per_ch.zarr"),
            layout="fov",
            mode="w-",
            channel_names=channel_names,
        ) as pos_ch:
            for ch in channel_names:
                pos_ch.write_xarray(xa_all.sel(c=[ch]))
            result_ch = pos_ch.to_xarray()

        assert_array_equal(result_all.values, result_ch.values)


@given(params=_xarray_roundtrip_params())
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_dims_shape_and_data(rng, params):
    """to_xarray output has correct dims, shape, and data for any valid shape."""
    channel_names, shape, scales, _ = params
    data = rng.random(shape).astype(np.float32)
    with TemporaryDirectory() as tmp:
        with open_ome_zarr(
            os.path.join(tmp, "d.zarr"),
            layout="fov",
            mode="w-",
            channel_names=channel_names,
        ) as pos:
            pos.create_image("0", data)
            for axis, dim in [("T", "t"), ("Z", "z"), ("Y", "y"), ("X", "x")]:
                pos.set_scale("0", axis, scales[dim])
            xa = pos.to_xarray()
        assert xa.dims == ("t", "c", "z", "y", "x")
        assert xa.shape == shape
        assert isinstance(xa.data, da.Array)
        assert_array_equal(xa.values, data)


def test_sel_by_channel_and_time(rng):
    """xarray .sel() works on to_xarray output for channel and time."""
    with TemporaryDirectory() as tmp:
        pos, data = _make_position(
            rng,
            tmp,
            ["BF", "GFP", "RFP"],
            (4, 3, 1, 8, 8),
            scales={"T": 0.5},
        )
        xa = pos.to_xarray()
        assert_array_equal(xa.sel(c="BF").values, data[:, 0])
        assert list(xa.sel(c=["BF", "RFP"]).coords["c"].values) == ["BF", "RFP"]
        assert_array_equal(xa.sel(t=1.0).values, data[2])
        pos.close()


class TestWriteXarray:
    def test_unknown_channel_raises(self, rng):
        with TemporaryDirectory() as tmp:
            xa = _build_xarray(rng, ["GFP"])
            with open_ome_zarr(
                os.path.join(tmp, "e.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["RFP"],
            ) as pos:
                with pytest.raises(ValueError, match="Channel 'GFP' not in"):
                    pos.write_xarray(xa)

    def test_wrong_dims_raises(self):
        bad = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        with TemporaryDirectory() as tmp:
            with open_ome_zarr(
                os.path.join(tmp, "e.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["ch1"],
            ) as pos:
                with pytest.raises(ValueError, match="dims must be"):
                    pos.write_xarray(bad)

    def test_uppercase_dims_raises(self):
        """Uppercase TCZYX dims (the old convention) should be rejected."""
        bad = xr.DataArray(
            np.zeros((1, 1, 1, 4, 4)),
            dims=("T", "C", "Z", "Y", "X"),
        )
        with TemporaryDirectory() as tmp:
            with open_ome_zarr(
                os.path.join(tmp, "e.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["ch1"],
            ) as pos:
                with pytest.raises(ValueError, match="dims must be"):
                    pos.write_xarray(bad)

    def test_4d_dims_raises(self):
        """4D array missing a dimension should be rejected."""
        bad = xr.DataArray(
            np.zeros((1, 1, 4, 4)),
            dims=("t", "c", "y", "x"),
        )
        with TemporaryDirectory() as tmp:
            with open_ome_zarr(
                os.path.join(tmp, "e.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["ch1"],
            ) as pos:
                with pytest.raises(ValueError, match="dims must be"):
                    pos.write_xarray(bad)

    def test_translation(self):
        with TemporaryDirectory() as tmp:
            xa = xr.DataArray(
                np.zeros((1, 1, 1, 4, 4), dtype=np.float32),
                dims=("t", "c", "z", "y", "x"),
                coords={
                    "t": ("t", [10.0], {"units": "second"}),
                    "c": ("c", ["ch1"]),
                    "z": ("z", [5.0], {"units": "micrometer"}),
                    "y": ("y", [100.0, 100.65, 101.3, 101.95], {"units": "micrometer"}),
                    "x": ("x", [200.0, 200.65, 201.3, 201.95], {"units": "micrometer"}),
                },
            )
            with open_ome_zarr(
                os.path.join(tmp, "t.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["ch1"],
            ) as pos:
                pos.write_xarray(xa)
                xa2 = pos.to_xarray()
                assert_allclose(xa2.coords["t"].values, [10.0])
                assert_allclose(
                    xa2.coords["y"].values,
                    xa.coords["y"].values,
                    atol=1e-6,
                )

    def test_write_time_subset(self, rng):
        """Write timepoints into a pre-allocated array."""
        with TemporaryDirectory() as tmp:
            t0 = rng.random(SHAPE_SMALL).astype(np.float32)
            t1 = rng.random(SHAPE_SMALL).astype(np.float32)
            coords = {
                "c": ("c", ["ch1"]),
                "z": ("z", [0.0]),
                "y": ("y", np.arange(8.0)),
                "x": ("x", np.arange(8.0)),
            }
            with open_ome_zarr(
                os.path.join(tmp, "t.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["ch1"],
            ) as pos:
                pos.create_zeros(
                    "0",
                    shape=(2, 1, 1, 8, 8),
                    dtype=np.float32,
                )
                for t_coord, data in [(0.0, t0), (1.0, t1)]:
                    xa = xr.DataArray(
                        data,
                        dims=("t", "c", "z", "y", "x"),
                        coords={"t": ("t", [t_coord]), **coords},
                    )
                    pos.write_xarray(xa)
                result = pos.to_xarray()
                assert_array_equal(
                    result.sel(t=0.0, c="ch1").values,
                    t0[0, 0],
                )
                assert_array_equal(
                    result.sel(t=1.0, c="ch1").values,
                    t1[0, 0],
                )

    def test_hcs_plate(self, rng):
        with TemporaryDirectory() as tmp:
            channels = ["GFP", "RFP"]
            xa = _build_xarray(rng, channels, shape=(2, 2, 3, 16, 16))
            with open_ome_zarr(
                os.path.join(tmp, "p.zarr"),
                layout="hcs",
                mode="w-",
                channel_names=channels,
            ) as plate:
                pos = plate.create_position("A", "1", "0")
                pos.write_xarray(xa)
                xa2 = pos.to_xarray()
            assert_array_equal(xa.values, xa2.values)
            assert list(xa2.coords["c"].values) == channels

    def test_waveorder_pattern(self, rng):
        """Read BF, compute Phase+Retardance, write channel-by-channel."""
        with TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "in.zarr")
            in_data = rng.random((2, 1, 4, 16, 16)).astype(np.float32)
            with open_ome_zarr(
                in_path,
                layout="fov",
                mode="w-",
                channel_names=["BF"],
            ) as p:
                p.create_image("0", in_data)
                p.set_scale("0", "Z", 2.0)
                p.set_scale("0", "Y", 0.65)
                p.set_scale("0", "X", 0.65)

            with open_ome_zarr(in_path, mode="r") as p:
                bf = p.to_xarray().sel(c=["BF"])
                phase = np.sin(bf.values).astype(np.float32)
                ret = np.cos(bf.values).astype(np.float32)

            out_path = os.path.join(tmp, "out.zarr")
            with open_ome_zarr(
                out_path,
                layout="fov",
                mode="w-",
                channel_names=["Phase", "Ret"],
            ) as p:
                for name, vals in [("Phase", phase), ("Ret", ret)]:
                    xa = xr.DataArray(
                        vals,
                        dims=("t", "c", "z", "y", "x"),
                        coords={
                            "t": bf.coords["t"],
                            "c": ("c", [name]),
                            "z": bf.coords["z"],
                            "y": bf.coords["y"],
                            "x": bf.coords["x"],
                        },
                    )
                    p.write_xarray(xa)

            with open_ome_zarr(out_path, mode="r") as p:
                result = p.to_xarray()
            assert result.shape == (2, 2, 4, 16, 16)
            assert_array_equal(
                result.sel(c="Phase").values,
                phase[:, 0],
            )
            assert_allclose(
                result.coords["z"].values,
                np.arange(4) * 2.0,
            )
