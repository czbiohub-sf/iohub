from iohub import open_ome_zarr
import numpy as np
from pathlib import Path
from shrimPy.mantis.cli.utils import create_empty_hcs_zarr

class OMEZarrHandler:
    def __init__(self, base_path, store_path, channel_names, dtype, time_index=slice(None), z_index=slice(None), y_index=slice(None), x_index=slice(None), create_store=False):
        self.base_path = base_path
        self.store_path = store_path
        self.channel_names = channel_names
        self.dtype = dtype
        self.time_index = time_index
        self.z_index = z_index
        self.y_index = y_index
        self.x_index = x_index
        self.create_store = create_store

    def open_zarr_store(self, path, layout="hcs", mode="r"):
        print(f"Opening Zarr store at {path} with layout '{layout}' and mode '{mode}'")
        return open_ome_zarr(path, layout=layout, mode=mode)

    def get_positions(self, ds):
        print("Retrieving positions from Zarr store")
        return list(ds.positions())

    def get_channel_indices(self, ds):
        print(f"Retrieving indices for channels: {self.channel_names}")
        return [ds.channel_names.index(name) for name in self.channel_names]

    def calculate_shape(self, array_shape):
        def get_range_len(index_slice, max_len):
            start, stop, step = index_slice.indices(max_len)
            return len(range(start, stop, step))

        time_len = get_range_len(self.time_index, array_shape[0])
        z_len = get_range_len(self.z_index, array_shape[2])
        y_len = get_range_len(self.y_index, array_shape[3])
        x_len = get_range_len(self.x_index, array_shape[4])

        return (time_len, len(self.channel_names), z_len, y_len, x_len)

    def create_new_zarr_store(self, positions, shape, chunks, scale, max_chunk_size_bytes=500e6):
        print(f"Creating new Zarr store at {self.store_path}")
        create_empty_hcs_zarr(
            store_path=Path(self.store_path),
            position_keys=positions,
            channel_names=self.channel_names,
            shape=shape,
            chunks=chunks,
            scale=scale,
            dtype=self.dtype,
            max_chunk_size_bytes=max_chunk_size_bytes
        )

    def validate_position_key(self, key):
        parts = key.split('/')
        if len(parts) == 3 and all(part.isalnum() for part in parts):
            return tuple(parts)
        return None

    def slice_and_assign(self):
        print("Starting slice and assign process")
        
        # Open OME-Zarr store
        ds = self.open_zarr_store(self.base_path)
        
        # Get positions and channel indices
        positions = self.get_positions(ds)
        channel_indices = self.get_channel_indices(ds)

        # Set shape, chunks, and scale 
        example_position = positions[0][1]  
        array_shape = example_position['0'].shape
        scale = example_position.scale
        shape = self.calculate_shape(array_shape)
        chunks = (1, 1, 1, shape[3], shape[4])

        # position keys
        position_keys = [self.validate_position_key(path) for path, _ in positions]
        position_keys = [key for key in position_keys if key is not None]

        # Create or open the Zarr store based on the create_store flag
        if self.create_store:
            self.create_new_zarr_store(position_keys, shape, chunks, scale)
            new_ds = self.open_zarr_store(self.store_path, mode="r+")
        else:
            new_ds = self.open_zarr_store(self.store_path, mode="r+")

        # Iterate over the positions and slice the 5D array
        for path, position in positions:
            try:
                array = position['0']
                # Slice the array and select channels 
                sliced_array = array[self.time_index, channel_indices, self.z_index, self.y_index, self.x_index]

                # Assign the sliced array to the corresponding position in the new store
                new_position = new_ds[path]
                new_position['0'][:] = sliced_array
                print(f"Updated position {path} with shape {sliced_array.shape}")

            except KeyError:
                print(f"Position path: {path} does not contain an array")

if __name__ == "__main__":
    base_path = '/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/2-register/registered.zarr'
    store_path = '/hpc/mydata/alishba.imran/test1.zarr'
    channel_names = ['RFP', 'Phase3D']  # list of channel names
    dtype = np.int32  # np.int32 or np.float32
    time_index = slice(0, 10)  
    z_index = slice(5, 15)  
    y_index = slice(100, 500)  
    x_index = slice(200, 800)  
    create_store = True  # set to False to use an existing store, True to create a new one

    handler = OMEZarrHandler(base_path, store_path, channel_names, dtype, time_index, z_index, y_index, x_index, create_store)
    handler.slice_and_assign()
