"""Custom exceptions for iohub core."""


class IohubError(Exception):
    """Base exception for all iohub errors."""


class ImplementationNotFoundError(IohubError, KeyError):
    """Raised when a requested implementation is not registered."""

    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        super().__init__(f"Unknown implementation: {name!r}. Available: {available}")


class StoreOpenError(IohubError, OSError):
    """Raised when a zarr store cannot be opened."""


class ArraySpecError(IohubError, ValueError):
    """Raised when an array spec is invalid."""


class PathNormalizationError(IohubError, ValueError):
    """Raised for invalid zarr paths (e.g., containing '..')."""
