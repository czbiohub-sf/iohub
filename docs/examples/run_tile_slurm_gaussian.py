# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "iohub @ file:///hpc/mydata/sricharan.varra/repos/iohub.tile",
#     "submitit",
#     "scipy",
#     "numpy",
#     "networkx",
#     "zarr",
#     "xarray",
# ]
# ///
"""
Tiled Gaussian Blur via SLURM
==============================

Demonstrates the three-phase tile-stitch pipeline with SLURM parallelism:

  1. ``create_tile_store`` — partition a FOV into overlapping tiles
  2. ``process_tiles``     — applied via SLURM array jobs (one job per batch)
  3. ``stitch_from_store`` — blend tile results into a final OME-Zarr

The processing function (Gaussian blur) is defined at module level so
submitit can pickle it for SLURM workers.

Usage::

    uv run docs/examples/run_tile_slurm_gaussian.py
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np
import submitit
from scipy.ndimage import gaussian_filter

from iohub.ngff import open_ome_zarr
from iohub.tile import create_tile_store, process_tiles, stitch_from_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and parameters — edit these for your dataset
# ---------------------------------------------------------------------------

INPUT = Path("/hpc/mydata/sricharan.varra/data/tile-testing/deskewed_t0_c0_pyramid_3.zarr")
TEMP_STORE = Path("/hpc/mydata/sricharan.varra/data/tile-testing/slurm_gaussian_tiles.zarr")
OUTPUT = Path("/hpc/mydata/sricharan.varra/data/tile-testing/slurm_gaussian_output.zarr")
SLURM_LOGS = Path("/hpc/mydata/sricharan.varra/data/tile-testing/slurm_gaussian_logs")

TILE_SIZE = {"z": 32, "y": 128, "x": 128}
OVERLAP = {"z": 16, "y": 32, "x": 32}
TILE_BATCH_SIZE = 16  # tiles per SLURM job
SIGMA = 2.0  # Gaussian blur sigma in pixels


# ---------------------------------------------------------------------------
# Processing function
# Must be module-level (not a lambda or closure) to be picklable by submitit.
# ---------------------------------------------------------------------------


def gaussian_blur(tile):
    """Apply a 3D Gaussian blur to each (T, C) slice of a tile."""
    data = tile.values.astype(np.float32)  # scipy requires float32+
    out = np.zeros_like(data)
    for t in range(data.shape[0]):
        for c in range(data.shape[1]):
            out[t, c] = gaussian_filter(data[t, c], sigma=SIGMA)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main():
    for p in (TEMP_STORE, OUTPUT):
        if p.exists():
            shutil.rmtree(p)
    SLURM_LOGS.mkdir(parents=True, exist_ok=True)

    pos = open_ome_zarr(str(INPUT), layout="fov")
    logger.info("Input: %s  shape=%s  dtype=%s", INPUT.name, pos.data.shape, pos.data.dtype)

    # Phase 1: partition into tiles
    batches = create_tile_store(
        pos,
        tile_size=TILE_SIZE,
        store=str(TEMP_STORE),
        overlap=OVERLAP,
        tile_batch_size=TILE_BATCH_SIZE,
    )
    logger.info("%d tiles → %d batches", sum(len(b) for b in batches), len(batches))

    # Phase 2: submit one SLURM job per batch
    executor = submitit.AutoExecutor(folder=str(SLURM_LOGS), cluster="slurm")
    executor.update_parameters(
        slurm_job_name="tile-gaussian",
        slurm_partition="cpu",
        slurm_mem_per_cpu="4G",
        slurm_cpus_per_task=2,
        slurm_array_parallelism=100,
        slurm_time=15,
    )

    t0 = time.time()
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for batch in batches:
            jobs.append(executor.submit(process_tiles, pos, gaussian_blur, str(TEMP_STORE), batch))

    logger.info("Waiting for %d jobs...", len(jobs))
    for job in submitit.helpers.as_completed(jobs):
        job.result()  # raises immediately if a job failed
        logger.info("  Job %s done", job.job_id)
    logger.info("All jobs complete in %.1fs", time.time() - t0)

    # Phase 3: stitch and blend
    stitch_from_store(str(TEMP_STORE), str(OUTPUT), pos, weights="gaussian")
    logger.info("Output: %s  (%.1fs total)", OUTPUT.name, time.time() - t0)

    # Verify
    result = open_ome_zarr(str(OUTPUT), layout="fov").data[:].astype(np.float32)
    original = pos.data[:].astype(np.float32)
    assert result.shape == original.shape
    assert not np.allclose(result, original), "Blur should change values"
    assert np.isfinite(result).all(), "Output contains NaN or inf"
    logger.info("Max diff from input: %.4e  PASSED", np.abs(result - original).max())

    # Cleanup
    shutil.rmtree(TEMP_STORE)
    shutil.rmtree(OUTPUT)


if __name__ == "__main__":
    main()
