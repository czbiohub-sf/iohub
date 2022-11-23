# iohub

This library provides io utilites for ND image data. 

Supported formats: 

Read: 
- single-page TIFF, OME-TIFF, NDTiff written by micro-manager, 
- custom data formats used by Biohub microscopes (e.g., PTI, DaXi).
- all the formats writte by this library.

Write: 
- OME-TIFF, 
- OME-zarr, 
- TIFF stacks that mimic OME-zarr structure. This provide benefits of a chunked format like zarr for visualizaion tools and [analysis pipelines that only support TIFF](https://github.com/mehta-lab/recOrder/issues/276).

Data access API (under discussion):
- utilities to visualize data output of iohub in napari and Fiji.
- utilities to ship data to deconvolution and DL pipelines.
 
