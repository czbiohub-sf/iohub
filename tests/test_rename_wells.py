from iohub import open_ome_zarr
from iohub.rename_wells import rename_wells
from tests.conftest import hcs_ref
from tests.ngff.test_ngff import _temp_copy


def test_cli_rename_wells(csv_data_file_1, csv_data_file_2):
    with _temp_copy(hcs_ref) as store_path:
        rename_wells(store_path, csv_data_file_1)
        with open_ome_zarr(store_path, mode="r") as plate:
            well_names = [well[0] for well in plate.wells()]
            assert "D/4" in well_names
            assert "B/03" not in well_names
            assert len(plate.metadata.wells) == 1
            assert len(plate.metadata.rows) == 1
            assert len(plate.metadata.columns) == 1
            assert plate.metadata.wells[0].path == "D/4"
            assert plate.metadata.rows[0].name == "D"
            assert plate.metadata.columns[0].name == "4"

        # Test round trip
        rename_wells(store_path, csv_data_file_2)
        with open_ome_zarr(store_path, mode="r") as plate:
            well_names = [well[0] for well in plate.wells()]
            assert "D/4" not in well_names
            assert "B/03" in well_names
            assert len(plate.metadata.wells) == 1
            assert len(plate.metadata.rows) == 1
            assert len(plate.metadata.columns) == 1
            assert plate.metadata.wells[0].path == "B/03"
            assert plate.metadata.rows[0].name == "B"
            assert plate.metadata.columns[0].name == "03"
