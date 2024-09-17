from typing import List

from iohub import open_ome_zarr
from iohub.ngff.models import TransformationMeta


def update_scale_metadata(
    input_position_dirpaths: List[str],
    z_scale: float,
    y_scale: float,
    x_scale: float,
):
    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(
            input_position_dirpath, layout="fov", mode="a"
        ) as input_dataset:
            print(
                f"Updating {input_position_dirpath} scale from "
                f"{tuple(input_dataset.scale[-3:])} to "
                f"{z_scale, y_scale, x_scale}."
            )
            input_dataset.zattrs["old_scale"] = input_dataset.scale[2:]
            transform = [
                TransformationMeta(
                    type="scale",
                    scale=(len(input_dataset.scale) - 3) * (1,)
                    + (z_scale, y_scale, x_scale),
                )
            ]
            input_dataset.set_transform("0", transform=transform)
