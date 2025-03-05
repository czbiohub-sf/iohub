import re

import pytest
from xarray import DataArray

from iohub.mmstack import MMOmeTiffFOV, MMStack
from tests.conftest import (
    mm2gamma_ome_tiffs,
    mm2gamma_ome_tiffs_incomplete,
    mm1422_ome_tiffs,
)


def pytest_generate_tests(metafunc):
    if "ome_tiff" in metafunc.fixturenames:
        metafunc.parametrize("ome_tiff", mm2gamma_ome_tiffs + mm1422_ome_tiffs)


def test_mmstack_ctx(ome_tiff):
    with MMStack(ome_tiff) as mmstack:
        assert isinstance(mmstack, MMStack)
        assert len(mmstack) > 0
        assert "MMStack" in mmstack.__repr__()


def test_mmstack_nonexisting(tmpdir):
    with pytest.raises(FileNotFoundError):
        MMStack(tmpdir / "nonexisting")


def test_mmstack_getitem(ome_tiff):
    mmstack = MMStack(ome_tiff)
    assert isinstance(mmstack["0"], MMOmeTiffFOV)
    assert isinstance(mmstack[0], MMOmeTiffFOV)
    for key, fov in mmstack:
        assert isinstance(key, str)
        assert isinstance(fov, MMOmeTiffFOV)
        assert key in mmstack.__repr__()
        assert key in fov.__repr__()
    mmstack.close()


def test_mmstack_num_positions(ome_tiff):
    with MMStack(ome_tiff) as mmstack:
        p_match = re.search(r"_(\d+)p_", str(ome_tiff))
        assert len(mmstack) == int(p_match.group(1)) if p_match else 1


def test_mmstack_num_timepoints(ome_tiff):
    with MMStack(ome_tiff) as mmstack:
        t_match = re.search(r"_(\d+)t_", str(ome_tiff))
        for _, fov in mmstack:
            assert fov.shape[0] == int(t_match.group(1)) if t_match else 1


def test_mmstack_num_timepoints_incomplete():
    with MMStack(mm2gamma_ome_tiffs_incomplete) as mmstack:
        for name, fov in mmstack:
            assert fov.shape[0] == 20
            if int(name) >= 11:
                assert not fov[:].any()
            else:
                assert fov[:].any()


def test_mmstack_metadata(ome_tiff):
    with MMStack(ome_tiff) as mmstack:
        assert isinstance(mmstack.micromanager_metadata, dict)
        assert mmstack.micromanager_metadata["Summary"]
        assert mmstack.micromanager_summary
        if mmstack.stage_positions:
            assert "DefaultXYStage" in mmstack.stage_positions[0]


def test_fov_axes_names(ome_tiff):
    mmstack = MMStack(ome_tiff)
    for _, fov in mmstack:
        axes_names = fov.axes_names
        assert isinstance(axes_names, list)
        assert len(axes_names) == 5
        assert all([isinstance(name, str) for name in axes_names])
    mmstack.close()


def test_fov_getitem(ome_tiff):
    mmstack = MMStack(ome_tiff)
    for _, fov in mmstack:
        img: DataArray = fov[:]
        assert isinstance(img, DataArray)
        assert img.ndim == 5
        assert img[0, 0, 0, 0, 0] >= 0
        for ch in fov.channel_names:
            assert img.sel(T=0, Z=0, C=ch, Y=0, X=0) >= 0
    mmstack.close()


def test_fov_equal(ome_tiff):
    ref1 = MMStack(ome_tiff)
    ref2 = MMStack(ome_tiff)
    for (_, fov1), (_, fov2) in zip(ref1, ref2):
        assert fov1 == fov2
    ref1.close()
    ref2.close()


def test_fov_not_equal(ome_tiff):
    with MMStack(ome_tiff) as mmstack:
        if len(mmstack) < 2:
            # pass
            return
        assert mmstack["0"] != mmstack["1"]
