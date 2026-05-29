# Reading OME-Zarr in a PyTorch DataLoader

iohub auto-selects the [zarrs](https://github.com/zarrs/zarrs-python) (Rust)
codec pipeline whenever the `zarrs` package is installed. It is fast, but its
parallel codec runs on a process-global thread pool that is **not fork-safe**.
PyTorch's `DataLoader` forks worker processes on Linux when `num_workers > 0`,
and the first chunk decode inside a forked worker deadlocks.

## A tiny dataset

```python title="dataset.py"
import torch
from torch.utils.data import Dataset

from iohub import open_ome_zarr


class FOVDataset(Dataset):
    def __init__(self, store_path: str):
        self.store_path = store_path

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Open lazily and read a single index from the "0" array.
        with open_ome_zarr(self.store_path, layout="fov", mode="r") as fov:
            return torch.from_numpy(fov["0"][idx])
```

## What breaks

!!! danger "Deadlocks on Linux"

    ```python title="train.py" hl_lines="3 7"
    from torch.utils.data import DataLoader

    sample = FOVDataset(store_path)[0]  # (1)!

    loader = DataLoader(FOVDataset(store_path), num_workers=4)  # (2)!
    for batch in loader:  # hangs forever on the first batch
        ...
    ```

    1.  Any decode in the **main process** before the workers start (a sanity
        read like this, normalization stats, a previous epoch) initializes the
        zarrs codec's thread pool here. This is what arms the deadlock.
    2.  `num_workers > 0` forks worker processes. Each child inherits the pool's
        locks but **none of its threads**, so the first decode blocks on a latch
        no worker will ever release.

!!! warning "Why it's intermittent"

    The hang needs **both** a `fork` start method **and** a main-process decode
    before iterating. A pipeline that reads *only* inside workers may never see it,
    while the same code that first reads an array in the main process deadlocks
    every time. That is exactly why this bug is easy to miss in testing.

## How to do it correctly

The simplest fix is to read in the main process, with no workers and no `fork()`:

!!! success "Load in the main process"

    ```python title="train.py" hl_lines="3"
    from torch.utils.data import DataLoader

    loader = DataLoader(FOVDataset(store_path), num_workers=0)  # (1)!
    for batch in loader:
        ...  # decodes on the calling process, safe
    ```

    1.  `num_workers=0` does all reading in the calling process. zarrs keeps
        its fast Rust codec and there is no `fork()` to corrupt the pool.

!!! tip "If you need parallel workers"

    Force **spawned** (not forked) workers, so each re-initializes zarrs in a
    fresh process:

    ```python title="train.py"
    loader = DataLoader(
        FOVDataset(store_path),
        num_workers=4,
        multiprocessing_context="spawn",
    )
    ```

    Spawn re-imports your modules and pickles the `Dataset` for each worker,
    so startup is slower and throughput drops by roughly 10-20%.

!!! note "Why this happens"

    This is not specific to OME-Zarr or `iohub`. Any `zarr` array read through the
    `zarrs-python` (Rust) codec pipeline inside a `torch` DataLoader hits it: the
    parallel codec uses a process-global thread pool that cannot survive `fork()`.
    [annbatch](https://annbatch.readthedocs.io/en/stable/) documents the same
    failure and recommends the same `spawn` workaround.
