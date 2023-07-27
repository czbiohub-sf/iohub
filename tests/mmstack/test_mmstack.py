import pytest
from iohub.multipagetiff import MMStack, MMOmeTiffFOV
from xarray import DataArray


@pytest.fixture(scope="function")
def random_ome_tiff_path(setup_mm2gamma_ome_tiffs):
    _, _, random_ome_tiff_path = setup_mm2gamma_ome_tiffs
    return random_ome_tiff_path


def test_mmstack_open(random_ome_tiff_path):
    mmstack = MMStack(random_ome_tiff_path)
    assert isinstance(mmstack, MMStack)
    assert len(mmstack) > 0


def test_mmstack_getitem(random_ome_tiff_path):
    mmstack = MMStack(random_ome_tiff_path)
    assert isinstance(mmstack["0"], MMOmeTiffFOV)
    assert isinstance(mmstack[0], MMOmeTiffFOV)


def test_fov_axes_names(random_ome_tiff_path):
    for _, fov in MMStack(random_ome_tiff_path):
        axes_names = fov.axes_names
        assert isinstance(axes_names, list)
        assert len(axes_names) == 5
        assert all([isinstance(name, str) for name in axes_names])
