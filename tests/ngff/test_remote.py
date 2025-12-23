"""Tests for remote URL access (HTTP/HTTPS, S3, etc.)."""

import numpy as np
import pytest

from iohub.ngff import open_ome_zarr
from iohub.ngff.nodes import _is_remote_url, _open_store

# Test datasets
WAVEORDER_URL = (
    "https://public.czbiohub.org/comp.micro/neurips_demos/waveorder/20x.zarr/"
)
VISCY_URL = (
    "https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/DENV/test/"
    "20240204_A549_DENV_ZIKV_timelapse/cropped_registered_test.zarr/"
)


class TestRemoteURLDetection:
    """Test URL scheme detection."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://example.com/data.zarr", True),
            ("http://example.com/data.zarr", True),
            ("s3://bucket/data.zarr", True),
            ("gs://bucket/data.zarr", True),
            ("az://container/data.zarr", True),
            ("/local/path/data.zarr", False),
            ("relative/path/data.zarr", False),
            ("C:\\Windows\\path\\data.zarr", False),
        ],
    )
    def test_is_remote_url(self, url, expected):
        assert _is_remote_url(url) == expected


class TestRemoteWriteRestriction:
    """Test that write modes are blocked for remote URLs."""

    @pytest.mark.parametrize("mode", ["w", "w-", "a", "r+"])
    def test_remote_write_not_supported(self, mode):
        with pytest.raises(NotImplementedError, match="not supported for remote"):
            _open_store("https://example.com/data.zarr", mode=mode, version="0.5")


@pytest.mark.network
class TestHTTPAccess:
    """Tests requiring network access. Run with: pytest -m network"""

    @pytest.fixture(scope="class")
    def waveorder_fov(self):
        """v0.5 single FOV (A/1/005028)."""
        return open_ome_zarr(WAVEORDER_URL, mode="r")

    @pytest.fixture(scope="class")
    def viscy_plate(self):
        """v0.4 plate with wells A/3 and B/4, position 9."""
        return open_ome_zarr(VISCY_URL, mode="r")

    # v0.5 tests
    def test_v05_plate_metadata(self, waveorder_fov):
        """Test v0.5 plate structure and metadata."""
        ome = waveorder_fov.zattrs["ome"]
        assert ome["version"] == "0.5"
        assert len(ome["plate"]["wells"]) == 1
        assert ome["plate"]["rows"][0]["name"] == "A"
        assert ome["plate"]["columns"][0]["name"] == "1"

    def test_v05_hierarchy_navigation(self, waveorder_fov):
        """Test navigating plate -> well -> position -> array."""
        well = waveorder_fov["A/1"]
        assert "well" in well.zattrs["ome"]

        pos = waveorder_fov["A/1/005028"]
        assert "multiscales" in pos.zattrs["ome"]

        arr = pos["0"]
        assert arr.shape == (1, 1, 7, 512, 512)
        assert arr.dtype == np.uint16

    def test_v05_read_data(self, waveorder_fov):
        """Test reading pixel data."""
        data = waveorder_fov["A/1/005028"].data[0, 0, 0, :5, :5]
        assert data.shape == (5, 5)
        assert data.sum() > 0

    # v0.4 tests
    def test_v04_plate_metadata(self, viscy_plate):
        """Test v0.4 plate structure and metadata."""
        plate_meta = viscy_plate.zattrs["plate"]
        assert plate_meta["version"] == "0.4"
        assert len(plate_meta["wells"]) == 2
        assert len(plate_meta["rows"]) == 2
        assert len(plate_meta["columns"]) == 2

    def test_v04_hierarchy_navigation(self, viscy_plate):
        """Test navigating plate -> well -> position -> array."""
        well = viscy_plate["A/3"]
        assert "well" in well.zattrs

        pos = viscy_plate["A/3/9"]
        assert "multiscales" in pos.zattrs

        arr = pos["0"]
        assert arr.shape == (48, 4, 74, 1022, 1020)
        assert arr.dtype == np.float32

    def test_v04_multiple_wells(self, viscy_plate):
        """Test accessing multiple wells."""
        well_paths = [w["path"] for w in viscy_plate.zattrs["plate"]["wells"]]
        assert set(well_paths) == {"A/3", "B/4"}

        assert "well" in viscy_plate["A/3"].zattrs
        assert "well" in viscy_plate["B/4"].zattrs

    def test_v04_read_data(self, viscy_plate):
        """Test reading pixel data."""
        data = viscy_plate["A/3/9"].data[0, 0, 0, :5, :5]
        assert data.shape == (5, 5)
        assert data.dtype == np.float32
