import pytest

from iohub._deprecated.singlepagetiff import MicromanagerSequenceReader
from iohub.mmstack import MMStack
from iohub.ndtiff import NDTiffDataset
from iohub.reader import read_images
from tests.conftest import (
    mm2gamma_ome_tiffs,
    mm2gamma_singlepage_tiffs,
    mm1422_ome_tiffs,
    ndtiff_v2_datasets,
    ndtiff_v3_labeled_positions,
)


def test_unsupported_datatype(tmpdir):
    with pytest.raises(ValueError):
        _ = read_images(tmpdir, data_type="unsupportedformat")


@pytest.mark.parametrize("data_path", mm2gamma_ome_tiffs + mm1422_ome_tiffs)
def test_detect_ome_tiff(data_path):
    reader = read_images(data_path)
    assert isinstance(reader, MMStack)


@pytest.mark.parametrize(
    "data_path", ndtiff_v2_datasets + [ndtiff_v3_labeled_positions]
)
def test_detect_ndtiff(data_path):
    reader = read_images(data_path)
    assert isinstance(reader, NDTiffDataset)


@pytest.mark.parametrize("data_path", mm2gamma_singlepage_tiffs)
def test_detect_single_page_tiff(data_path):
    reader = read_images(data_path)
    assert isinstance(reader, MicromanagerSequenceReader)
