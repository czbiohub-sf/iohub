import click
import numpy as np

from iohub import open_ome_zarr
from iohub.cli.parsing import _validate_and_process_paths


def test_validate_and_process_paths(tmpdir):
    plate_path = tmpdir / "dataset.zarr"

    position_list = [("A", "1", "0"), ("B", "2", "0"), ("X", "4", "1")]
    with open_ome_zarr(
        plate_path, mode="w", layout="hcs", channel_names=["1", "2"]
    ) as dataset:
        for position in position_list:
            pos = dataset.create_position(*position)
            pos.create_zeros("0", shape=(1, 1, 1, 1, 1), dtype=np.uint8)

    cmd = click.Command("test")
    ctx = click.Context(cmd)
    opt = click.Option(["--path"], type=click.Path(exists=True))

    # Check plate expansion
    processed = _validate_and_process_paths(ctx, opt, [str(plate_path)])
    assert len(processed) == len(position_list)
    for i, position in enumerate(position_list):
        assert processed[i].parts[-3:] == position

    # Check single position
    processed = _validate_and_process_paths(
        ctx, opt, [str(plate_path / "A" / "1" / "0")]
    )
    assert len(processed) == 1

    # Check two positions
    processed = _validate_and_process_paths(
        ctx,
        opt,
        [str(plate_path / "A" / "1" / "0"), str(plate_path / "B" / "2" / "0")],
    )
    assert len(processed) == 2
