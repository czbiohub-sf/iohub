iohub
=====

N-dimensional bioimaging produces data and metadata in various formats,
and iohub aims to become a unified Python interface to the most common formats
used at the Biohub and in the broader imaging community.


Read
----

- OME-Zarr (`OME-NGFF v0.4 <https://ngff.openmicroscopy.org/0.4/>`_)

- Micro-Manager TIFF sequence, OME-TIFF (MMStack), and NDTiff datasets

- Custom data formats generated by Biohub microscopes

  - Supported: Falcon (PTI), Dorado (ClearControl), Dragonfly (OpenCell OME-TIFF), Mantis (NDTiff)

  - WIP: DaXi

Write
-----

- OME-Zarr
- Multi-page TIFF stacks organized in a directory hierarchy that mimics OME-NGFF (WIP)



.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   auto_examples/index
   api
   contact_us
