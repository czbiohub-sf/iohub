iohub
~~~~~~~~~~~~~

N-dimensional bioimaging produces data and metadata in various formats,
and iohub aims to become a unified Python interface to the most common formats
used at the Biohub and in the broader imaging community.


Read
====

- OME-Zarr (`OME-NGFF v0.4 <https://ngff.openmicroscopy.org/0.4/>`_)
- Single-page TIFF, OME-TIFF, and NDTiff images written by Micro-Manager/pycro-manager
- Custom data formats used by Biohub microscopes (e.g., PTI, Mantis (WIP), DaXi (TBD))

Write
======

- OME-Zarr
- Multi-page TIFF stacks organized in a directory hierarchy that mimics OME-NGFF (WIP)



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   Install <getting_started/install.rst>
   Why iohub? <getting_started/why.rst>


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   Tutorials Home <tutorials/home.rst>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   OME-NGFF (OME-Zarr) <api/ngff>
   Read Micro-Manager datasets <api/mm_reader>
   Read MM TIFF sequence <api/mm_sequence_reader>
   Read MMStack OME-TIFF <api/mm_ometiff_reader>
   Read NDTiff <api/ndtiff>
   Read PTI TIFF <api/upti>
   Convert TIFF to OME-Zarr <api/mm_converter>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contact Us

   On Github <contact_us/github>
