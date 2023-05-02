Getting Started
===============

Install
-------

Install iohub from PyPI:

.. code-block:: shell

    pip install iohub

Command-line interface
----------------------

To check if iohub works for a dataset:

.. code-block:: shell

    iohub info /path/to/data/

The CLI can show a summary of the dataset,
point to relevant Python calls,
and convert other data formats to the latest OME-Zarr.
See the full CLI help message by typing ``iohub`` or ``iohub [command] --help`` in the terminal.

Why iohub?
----------

This project is inspired by the existing Python libraries for bioimaging data I/O,
including `ome-zarr-py <https://github.com/ome/ome-zarr-py>`_, `tifffile <https://github.com/cgohlke/tifffile>`_ and
`aicsimageio <https://github.com/AllenCellModeling/aicsimageio>`_.
They support some of the most widely adopted and/or promising formats in microscopy,
such as OME-Zarr and OME-Tiff.

iohub bridges the gaps among them with the following features:

- Efficient reading of data in various TIFF-based formats produced by the Micro-Manager/Pycro-Manager acquisition stack.

- Efficient and customizable conversion of data and metadata from Tiff to OME-Zarr.

- Pythonic and atomic access of OME-Zarr data with parallelized analysis in mind.

- OME-Zarr metadata is automatically constructed and updated for writing, and verified against the specification when reading.

- Adherence to the latest OME-NGFF specification (v0.4) whenever possible.
