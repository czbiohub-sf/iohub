import re

import pytest
from xarray import DataArray

from iohub.mmstack import MMOmeTiffFOV, MMStack
from tests.conftest import mm2gamma_ome_tiffs


def pytest_generate_tests(metafunc):
    if "mm2gamma_ome_tiff" in metafunc.fixturenames:
        metafunc.parametrize("mm2gamma_ome_tiff", mm2gamma_ome_tiffs)


def test_mmstack_ctx(mm2gamma_ome_tiff):
    with MMStack(mm2gamma_ome_tiff) as mmstack:
        assert isinstance(mmstack, MMStack)
        assert len(mmstack) > 0


def test_mmstack_nonexisting(tmpdir):
    with pytest.raises(FileNotFoundError):
        MMStack(tmpdir / "nonexisting")


def test_mmstack_getitem(mm2gamma_ome_tiff):
    mmstack = MMStack(mm2gamma_ome_tiff)
    assert isinstance(mmstack["0"], MMOmeTiffFOV)
    assert isinstance(mmstack[0], MMOmeTiffFOV)
    for key, fov in mmstack:
        assert isinstance(key, str)
        assert isinstance(fov, MMOmeTiffFOV)
    mmstack.close()


def test_mmstack_num_positions(mm2gamma_ome_tiff):
    with MMStack(mm2gamma_ome_tiff) as mmstack:
        p_match = re.search(r"_(\d+)p_", str(mm2gamma_ome_tiff))
        assert len(mmstack) == int(p_match.group(1)) if p_match else 1


def test_mmstack_metadata(mm2gamma_ome_tiff):
    with MMStack(mm2gamma_ome_tiff) as mmstack:
        assert isinstance(mmstack.mm_meta, dict)
        assert mmstack.mm_meta["Summary"]


def test_fov_axes_names(mm2gamma_ome_tiff):
    mmstack = MMStack(mm2gamma_ome_tiff)
    for _, fov in mmstack:
        axes_names = fov.axes_names
        assert isinstance(axes_names, list)
        assert len(axes_names) == 5
        assert all([isinstance(name, str) for name in axes_names])
    mmstack.close()


def test_fov_getitem(mm2gamma_ome_tiff):
    mmstack = MMStack(mm2gamma_ome_tiff)
    for _, fov in mmstack:
        img: DataArray = fov[:]
        assert isinstance(img, DataArray)
        assert img.ndim == 5
        assert img[0, 0, 0, 0, 0] >= 0
        assert img.sel(T=0, Z=0, C=0, Y=0, X=0) >= 0
    mmstack.close()
