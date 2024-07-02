from typing import List

from iohub import open_ome_zarr
from iohub.ngff_meta import TransformationMeta


def update_scale_metadata(
    input_position_dirpaths: List[str],
    x_scale: float = None,
    y_scale: float = None,
    z_scale: float = None,
):
    with open_ome_zarr(input_position_dirpaths[0]) as input_dataset:
        print(
            f"The first dataset in the list you provided has (z, y, x) scale {input_dataset.scale[2:]}"
        )

    print(
        "Please enter the new z, y, and x scales that you would like to apply to all of the positions in the list."
    )
    print(
        "The old scale will be saved in a metadata field named 'old_scale', and the new scale will adhere to the NGFF spec."
    )
    new_scale = [z_scale, y_scale, x_scale]
    for i, character in enumerate("zyx"):
        if new_scale[i] is None:
            new_scale[i] = (float(input(f"Enter a new {character} scale: ")))

    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(input_position_dirpath, layout="fov", mode="a") as input_dataset:
            input_dataset.zattrs['old_scale'] = input_dataset.scale[2:]
            transform = [
                TransformationMeta(
                    type="scale",
                    scale=(
                        1,
                        1,
                    )
                    + tuple(new_scale),
                )
            ]
            input_dataset.set_transform("0", transform=transform)

    print(f"The dataset now has (z, y, x) scale {tuple(new_scale)}.")