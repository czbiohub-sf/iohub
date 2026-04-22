"""Zarr I/O implementation backends.

Implementations are discovered via the ``iohub.zarr_implementations``
entry point group. See :mod:`iohub.core.registry` for the discovery mechanism.

For direct imports::

    from iohub.core.implementations.zarr_python import ZarrPythonImplementation
    from iohub.core.implementations.tensorstore import TensorStoreImplementation
"""
