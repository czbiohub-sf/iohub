iohub
~~~~~~~~~~~~~

N-dimensional bioimaging produces data and metadata in various formats,
and iohub aims to become a unified Python interface to the most common formats
used at the Biohub and in the broader imaging community.


Read
====

- OME-Zarr
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

   NGFF <api/ngff>
   MicromanagerSequenceReader <api/mm_sequence_reader>
   UPTIReader <api/upti>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contact Us

   On Github <contact_us/github>
