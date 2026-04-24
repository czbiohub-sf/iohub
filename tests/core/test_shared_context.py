"""Tests for TensorStoreConfig.shared_context passthrough."""

from __future__ import annotations

import pytest


def test_shared_context_defaults_to_none() -> None:
    """Default construction has shared_context=None for backwards compat."""
    from iohub.core.config import TensorStoreConfig

    cfg = TensorStoreConfig()
    assert cfg.shared_context is None


def test_shared_context_accepts_ts_context() -> None:
    """Can assign a tensorstore.Context to shared_context."""
    ts = pytest.importorskip("tensorstore")

    from iohub.core.config import TensorStoreConfig

    ctx = ts.Context({"cache_pool": {"total_bytes_limit": 1_000_000}})
    cfg = TensorStoreConfig(shared_context=ctx)
    assert cfg.shared_context is ctx


def test_shared_context_is_returned_by_implementation_context() -> None:
    """TensorStoreImplementation._context() returns the shared Context when set."""
    ts = pytest.importorskip("tensorstore")

    from iohub.core.config import TensorStoreConfig
    from iohub.core.implementations.tensorstore import TensorStoreImplementation

    shared = ts.Context({"cache_pool": {"total_bytes_limit": 2_000_000}})
    cfg = TensorStoreConfig(shared_context=shared)
    impl = TensorStoreImplementation(config=cfg)
    assert impl._context() is shared


def test_two_implementations_share_one_context() -> None:
    """Two implementations built with the same shared_context return the same Context."""
    ts = pytest.importorskip("tensorstore")

    from iohub.core.config import TensorStoreConfig
    from iohub.core.implementations.tensorstore import TensorStoreImplementation

    shared = ts.Context({"cache_pool": {"total_bytes_limit": 500_000}})
    cfg_a = TensorStoreConfig(shared_context=shared)
    cfg_b = TensorStoreConfig(shared_context=shared)
    impl_a = TensorStoreImplementation(config=cfg_a)
    impl_b = TensorStoreImplementation(config=cfg_b)
    assert impl_a._context() is impl_b._context()


def test_no_shared_context_falls_back_to_per_instance() -> None:
    """Without shared_context, each implementation builds its own Context."""
    pytest.importorskip("tensorstore")

    from iohub.core.config import TensorStoreConfig
    from iohub.core.implementations.tensorstore import TensorStoreImplementation

    cfg = TensorStoreConfig()
    impl_a = TensorStoreImplementation(config=cfg)
    impl_b = TensorStoreImplementation(config=cfg)
    assert impl_a._context() is not impl_b._context()
