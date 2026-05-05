import pytest

from iohub.mmstack import MMStack
from iohub.ndtiff import NDTiffDataset
from iohub.ngff.nodes import Bioformats2RawSeries
from iohub.reader import print_info, read_images, sizeof_fmt
from tests.conftest import (
    mm2gamma_ome_tiffs,
    mm2gamma_singlepage_tiffs,
    mm1422_ome_tiffs,
    ndtiff_v2_datasets,
    ndtiff_v3_labeled_positions,
)
from tests.ngff.test_ngff import _make_bf2raw_fixture


def test_unsupported_datatype(tmpdir):
    with pytest.raises(ValueError, match=r"."):
        _ = read_images(tmpdir, data_type="unsupportedformat")


@pytest.mark.parametrize("data_path", mm2gamma_ome_tiffs + mm1422_ome_tiffs)
def test_detect_ome_tiff(data_path):
    reader = read_images(data_path)
    assert isinstance(reader, MMStack)
    reader.close()


@pytest.mark.parametrize("data_path", [*ndtiff_v2_datasets, ndtiff_v3_labeled_positions])
def test_detect_ndtiff(data_path):
    reader = read_images(data_path)
    assert isinstance(reader, NDTiffDataset)
    reader.close()


@pytest.mark.parametrize("data_path", mm2gamma_singlepage_tiffs)
def test_detect_single_page_tiff(data_path):
    with pytest.raises(NotImplementedError, match="Single-page TIFF"):
        read_images(data_path)


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [(3, "3 B"), (2.234 * 2**20, "2.2 MiB"), (3.456 * 2**40, "3.5 TiB")],
)
def test_sizeof_fmt(num_bytes, expected):
    assert sizeof_fmt(num_bytes) == expected


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_read_images_bf2raw(tmp_path, version):
    """``read_images`` returns a Bioformats2RawSeries for bf2raw stores."""
    store_path = _make_bf2raw_fixture(tmp_path, version)
    reader = read_images(store_path)
    try:
        assert isinstance(reader, Bioformats2RawSeries)
        assert len(reader) == 2
    finally:
        reader.close()


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_print_info_bf2raw(tmp_path, capsys, version):
    """``print_info`` runs and reports series count for bf2raw stores."""
    store_path = _make_bf2raw_fixture(tmp_path, version)
    print_info(store_path)
    captured = capsys.readouterr().out
    assert "Positions:" in captured
    assert f"v{version}" in captured
