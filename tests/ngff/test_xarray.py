"""Tests for Position.to_xarray() and Position.write_xarray()."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from iohub.ngff.nodes import open_ome_zarr

RNG = np.random.default_rng(42)
SHAPE_SMALL = (1, 1, 1, 8, 8)


def _make_position(tmp_dir, channel_names, shape, scales=None, name="test.zarr"):
    """Create a Position with random float32 data and optional scales."""
    pos = open_ome_zarr(
        os.path.join(tmp_dir, name),
        layout="fov",
        mode="w-",
        channel_names=channel_names,
    )
    data = RNG.random(shape).astype(np.float32)
    pos.create_image("0", data)
    for axis, val in (scales or {}).items():
        pos.set_scale("0", axis, val)
    return pos, data


def _make_xarray(channels, shape=SHAPE_SMALL, scales=None, coord_units=None, attrs=None):
    """Build a TCZYX DataArray with CF-convention coordinate units."""
    T, _, Z, Y, X = shape[0], None, shape[2], shape[3], shape[4]
    C = len(channels)
    scales = scales or {}
    data = RNG.random((T, C, Z, Y, X)).astype(np.float32)
    coord_units = coord_units or {"T": "second", "Z": "micrometer", "Y": "micrometer", "X": "micrometer"}
    dims_info = {
        "T": (T, scales.get("T", 1.0)),
        "Z": (Z, scales.get("Z", 1.0)),
        "Y": (Y, scales.get("Y", 0.65)),
        "X": (X, scales.get("X", 0.65)),
    }
    coords = {"C": ("C", channels)}
    for dim, (size, sc) in dims_info.items():
        unit_attrs = {"units": coord_units[dim]} if dim in coord_units else {}
        coords[dim] = (dim, np.arange(size) * sc, unit_attrs)
    return xr.DataArray(data, dims=("T", "C", "Z", "Y", "X"), coords=coords, attrs=attrs or {})


class TestToXarray:
    def test_dims_shape_and_data(self):
        with TemporaryDirectory() as tmp:
            pos, data = _make_position(tmp, ["GFP", "RFP"], (2, 2, 3, 16, 16))
            xa = pos.to_xarray()
            assert xa.dims == ("T", "C", "Z", "Y", "X")
            assert xa.shape == (2, 2, 3, 16, 16)
            assert isinstance(xa.data, da.Array)
            assert_array_equal(xa.values, data)
            pos.close()

    def test_channel_and_physical_coordinates(self):
        with TemporaryDirectory() as tmp:
            pos, _ = _make_position(
                tmp,
                ["BF", "GFP"],
                (2, 2, 4, 16, 16),
                scales={"T": 0.5, "Z": 2.0, "Y": 0.65, "X": 0.65},
            )
            xa = pos.to_xarray()
            assert list(xa.coords["C"].values) == ["BF", "GFP"]
            assert_allclose(xa.coords["T"].values, [0.0, 0.5])
            assert_allclose(xa.coords["Z"].values, np.arange(4) * 2.0)
            assert_allclose(xa.coords["Y"].values, np.arange(16) * 0.65)
            pos.close()

    def test_coordinate_units_cf(self):
        with TemporaryDirectory() as tmp:
            pos, _ = _make_position(tmp, ["ch1"], SHAPE_SMALL)
            xa = pos.to_xarray()
            assert xa.coords["T"].attrs["units"] == "second"
            assert xa.coords["Z"].attrs["units"] == "micrometer"
            assert "units" not in xa.coords["C"].attrs
            pos.close()

    def test_sel_by_channel_and_time(self):
        with TemporaryDirectory() as tmp:
            pos, data = _make_position(
                tmp,
                ["BF", "GFP", "RFP"],
                (4, 3, 1, 8, 8),
                scales={"T": 0.5},
            )
            xa = pos.to_xarray()
            assert_array_equal(xa.sel(C="BF").values, data[:, 0])
            assert list(xa.sel(C=["BF", "RFP"]).coords["C"].values) == ["BF", "RFP"]
            assert_array_equal(xa.sel(T=1.0).values, data[2])
            pos.close()


class TestWriteXarray:
    def test_roundtrip_data_and_scales(self):
        with TemporaryDirectory() as tmp:
            pos, data = _make_position(
                tmp,
                ["GFP", "RFP"],
                (2, 2, 3, 16, 16),
                scales={"T": 0.5, "Z": 2.0, "Y": 0.65, "X": 0.65},
                name="orig.zarr",
            )
            xa1 = pos.to_xarray()
            pos.close()

            with open_ome_zarr(
                os.path.join(tmp, "rt.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["GFP", "RFP"],
            ) as pos2:
                pos2.write_xarray(xa1)
                xa2 = pos2.to_xarray()

            assert_array_equal(xa1.values, xa2.values)
            for dim in ("T", "Z", "Y", "X"):
                assert_allclose(xa1.coords[dim].values, xa2.coords[dim].values)

    def test_roundtrip_coordinate_units(self):
        with TemporaryDirectory() as tmp:
            xa = _make_xarray(
                ["ch1"],
                coord_units={
                    "T": "millisecond",
                    "Z": "nanometer",
                    "Y": "nanometer",
                    "X": "nanometer",
                },
            )
            with open_ome_zarr(
                os.path.join(tmp, "u.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["ch1"],
            ) as pos:
                pos.write_xarray(xa)
                xa2 = pos.to_xarray()
            assert xa2.coords["T"].attrs["units"] == "millisecond"
            assert xa2.coords["Z"].attrs["units"] == "nanometer"

    def test_roundtrip_value_attrs(self):
        with TemporaryDirectory() as tmp:
            xa = _make_xarray(["ch1"], attrs={"units": "nanometer", "quantity": "OPL"})
            path = os.path.join(tmp, "v.zarr")
            with open_ome_zarr(path, layout="fov", mode="w-", channel_names=["ch1"]) as pos:
                pos.write_xarray(xa)
            with open_ome_zarr(path, mode="r") as pos:
                xa2 = pos.to_xarray()
            assert xa2.attrs["units"] == "nanometer"
            assert xa2.attrs["quantity"] == "OPL"

    def test_unknown_channel_raises(self):
        with TemporaryDirectory() as tmp:
            xa = _make_xarray(["GFP"])
            with open_ome_zarr(
                os.path.join(tmp, "e.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["RFP"],
            ) as pos:
                with pytest.raises(ValueError, match="Channel 'GFP' not in"):
                    pos.write_xarray(xa)

    def test_wrong_dims_raises(self):
        bad = xr.DataArray(np.zeros((2, 3)), dims=("X", "Y"))
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
                dims=("T", "C", "Z", "Y", "X"),
                coords={
                    "T": ("T", [10.0], {"units": "second"}),
                    "C": ("C", ["ch1"]),
                    "Z": ("Z", [5.0], {"units": "micrometer"}),
                    "Y": ("Y", [100.0, 100.65, 101.3, 101.95], {"units": "micrometer"}),
                    "X": ("X", [200.0, 200.65, 201.3, 201.95], {"units": "micrometer"}),
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
                assert_allclose(xa2.coords["T"].values, [10.0])
                assert_allclose(xa2.coords["Y"].values, xa.coords["Y"].values, atol=1e-6)

    def test_write_channel_subset(self):
        """Write channels one at a time into a multi-channel position."""
        with TemporaryDirectory() as tmp:
            channels = ["Phase", "Ret", "Ori"]
            per_ch = {ch: RNG.random(SHAPE_SMALL).astype(np.float32) for ch in channels}

            with open_ome_zarr(
                os.path.join(tmp, "p.zarr"),
                layout="fov",
                mode="w-",
                channel_names=channels,
            ) as pos:
                for ch, vals in per_ch.items():
                    xa = xr.DataArray(
                        vals,
                        dims=("T", "C", "Z", "Y", "X"),
                        coords={
                            "T": ("T", [0.0]),
                            "C": ("C", [ch]),
                            "Z": ("Z", [0.0]),
                            "Y": ("Y", np.arange(8.0)),
                            "X": ("X", np.arange(8.0)),
                        },
                    )
                    pos.write_xarray(xa)
                result = pos.to_xarray()
                for ch, vals in per_ch.items():
                    assert_array_equal(result.sel(C=ch).values, vals[:, 0])

    def test_write_time_subset(self):
        """Write timepoints into a pre-allocated array."""
        with TemporaryDirectory() as tmp:
            t0 = RNG.random(SHAPE_SMALL).astype(np.float32)
            t1 = RNG.random(SHAPE_SMALL).astype(np.float32)
            coords = {"C": ("C", ["ch1"]), "Z": ("Z", [0.0]), "Y": ("Y", np.arange(8.0)), "X": ("X", np.arange(8.0))}

            with open_ome_zarr(
                os.path.join(tmp, "t.zarr"),
                layout="fov",
                mode="w-",
                channel_names=["ch1"],
            ) as pos:
                pos.create_zeros("0", shape=(2, 1, 1, 8, 8), dtype=np.float32)
                for t_coord, data in [(0.0, t0), (1.0, t1)]:
                    xa = xr.DataArray(
                        data,
                        dims=("T", "C", "Z", "Y", "X"),
                        coords={"T": ("T", [t_coord]), **coords},
                    )
                    pos.write_xarray(xa)
                result = pos.to_xarray()
                assert_array_equal(result.sel(T=0.0, C="ch1").values, t0[0, 0])
                assert_array_equal(result.sel(T=1.0, C="ch1").values, t1[0, 0])

    def test_hcs_plate(self):
        with TemporaryDirectory() as tmp:
            channels = ["GFP", "RFP"]
            xa = _make_xarray(channels, shape=(2, 2, 3, 16, 16))
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
            assert list(xa2.coords["C"].values) == channels

    def test_waveorder_pattern(self):
        """Read BF, compute Phase+Retardance, write channel-by-channel."""
        with TemporaryDirectory() as tmp:
            # Create input
            in_path = os.path.join(tmp, "in.zarr")
            in_data = RNG.random((2, 1, 4, 16, 16)).astype(np.float32)
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

            # Read, process, write
            with open_ome_zarr(in_path, mode="r") as p:
                bf = p.to_xarray().sel(C=["BF"])
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
                        dims=("T", "C", "Z", "Y", "X"),
                        coords={
                            "T": bf.coords["T"],
                            "C": ("C", [name]),
                            "Z": bf.coords["Z"],
                            "Y": bf.coords["Y"],
                            "X": bf.coords["X"],
                        },
                    )
                    p.write_xarray(xa)

            with open_ome_zarr(out_path, mode="r") as p:
                result = p.to_xarray()
            assert result.shape == (2, 2, 4, 16, 16)
            assert_array_equal(result.sel(C="Phase").values, phase[:, 0])
            assert_allclose(result.coords["Z"].values, np.arange(4) * 2.0)
