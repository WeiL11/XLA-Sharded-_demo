"""Optional profiling helpers for XLA traces."""

from __future__ import annotations

from contextlib import contextmanager

import jax.profiler


@contextmanager
def trace_to(logdir: str):
    """Context manager for jax.profiler tracing."""
    jax.profiler.start_trace(logdir)
    try:
        yield
    finally:
        jax.profiler.stop_trace()

