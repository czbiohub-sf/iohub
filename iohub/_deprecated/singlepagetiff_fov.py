"""FOV wrapper for single-page TIFF data."""
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import dask.array as da
from iohub.fov import BaseFOV

logger = logging.getLogger(__name__)


class SinglePageTiffFOV(BaseFOV):
    """FOV wrapper for MicromanagerSequenceReader positions.
    
    This provides the interface expected by TIFFConverter for accessing
    single-page TIFF data organized by MicromanagerSequenceReader.
    """
    
    def __init__(self, reader: 'MicromanagerSequenceReader', position: int):
        """Initialize FOV for a specific position.
        
        Parameters
        ----------
        reader : MicromanagerSequenceReader
            Parent reader instance
        position : int
            Position index
        """
        self.reader = reader
        self.position = position
        self._data = None
        
    @property
    def root(self) -> Path:
        """Return root path."""
        return Path(self.reader._folder)
        
    @property
    def axes_names(self) -> list[str]:
        """Return axis names in TCZYX order."""
        return ["T", "C", "Z", "Y", "X"]
        
    @property
    def channel_names(self) -> list[str]:
        """Return channel names."""
        return self.reader.channel_names
        
    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """Return data shape (T, C, Z, Y, X)."""
        return (
            self.reader.frames if self.reader.frames > 0 else 1,
            self.reader.channels if self.reader.channels > 0 else 1,
            self.reader.slices if self.reader.slices > 0 else 1,
            self.reader.height,
            self.reader.width
        )
        
    @property
    def dtype(self):
        """Return data type."""
        return self.reader.dtype
        
    @property
    def t_scale(self) -> float:
        """Return time scale (default 1.0)."""
        # Could extract from metadata if available
        return 1.0
        
    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        """Return ZYX voxel scale."""
        z_scale = self.reader.z_step_size if self.reader.z_step_size else 1.0
        # Default to 1.0 for XY if not specified in metadata
        return (z_scale, 1.0, 1.0)
        
    def _load_data(self):
        """Load position data if not already loaded."""
        if self._data is None:
            # This triggers loading via _create_stores in the reader
            zarr_data = self.reader.get_zarr(self.position)
            # Convert to dask array for compatibility with converter
            self._data = da.from_zarr(zarr_data)
            
    @property
    def xdata(self):
        """Return data wrapped with a .data attribute for converter compatibility."""
        if self._data is None:
            self._load_data()
            
        class DataWrapper:
            def __init__(self, data):
                self.data = data
                
        return DataWrapper(self._data)
        
    def frame_metadata(self, t: int, c: int | str, z: int) -> Optional[dict]:
        """Get metadata for a specific frame.
        
        Parameters
        ----------
        t : int
            Time index
        c : int or str
            Channel index or name
        z : int
            Z slice index
            
        Returns
        -------
        dict or None
            Frame metadata if available
        """
        # Convert channel name to index if needed
        if isinstance(c, str):
            try:
                c = self.channel_names.index(c)
            except ValueError:
                return None
                
        # Check if coordinate exists
        coord = (self.position, t, c, z)
        if coord in self.reader.coord_to_filename:
            # Could extract more metadata from individual TIFF files if needed
            return {
                "Position": self.position,
                "Time": t,
                "Channel": c,
                "Z": z,
                "Filename": self.reader.coord_to_filename[coord]
            }
        return None
        
    def __getitem__(self, key):
        """Get a slice of the data.
        
        Parameters
        ----------
        key : int, slice, or tuple of int/slice
            Indexing key for the 5D array (T, C, Z, Y, X)
            
        Returns
        -------
        ArrayLike
            The indexed data
        """
        if self._data is None:
            self._load_data()
            
        # Handle indexing into the dask array
        return self._data[key]