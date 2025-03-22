import csv
from pathlib import Path

from iohub.ngff import open_ome_zarr


def rename_wells(zarr_store_path: str | Path, csv_file_path: str | Path):
    """
    Rename wells in a Zarr store based on a CSV file containing old and new
    well names.

    Parameters
    ----------
    zarr_store_path : str or Path
        Path to the Zarr store.
    csv_file_path : str or Path
        Path to the CSV file containing the old and new well names.

    Raises
    ------
    ValueError
        If a row in the CSV file does not have exactly two columns.
        If there is an error renaming a well in the Zarr store.

    Notes
    -----
    The CSV file should have two columns:
        - The first column contains the old well names.
        - The second column contains the new well names.

    Examples
    --------
    CSV file content:
    A/1,B/2
    A/2,B/2

    """

    # read and check csv
    name_pair_list = []
    with open(csv_file_path, mode="r", encoding="utf-8-sig") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) != 2:
                raise ValueError(
                    f"Invalid row format: {row}."
                    f"Each row must have two columns."
                )
            name_pair_list.append([row[0].strip(), row[1].strip()])

    # rename each well while catching errors
    with open_ome_zarr(zarr_store_path, mode="a") as plate:
        for old, new in name_pair_list:
            print(f"Renaming {old} to {new}")
            try:
                plate.rename_well(old, new)
            except ValueError as e:
                print(f"Error renaming {old} to {new}: {e}")
