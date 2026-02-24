"""
Optional profiling helpers for XLA traces and HLO IR dumps.

TensorBoard usage
-----------------
    from benchmark.profiler import trace_to

    with trace_to("/tmp/xla_trace"):
        result = generate_xla(prompt)

    # View with:  tensorboard --logdir /tmp/xla_trace

Requires: pip install "xla-sharded[profile]"
    i.e.  pip install tensorboard tensorboard-plugin-profile

HLO dump usage
--------------
    from benchmark.profiler import dump_hlo

    with dump_hlo("/tmp/hlo_dump"):
        jitted_fn(args)   # .txt files appear in /tmp/hlo_dump

Combined usage
--------------
    with profile_and_dump("/tmp/run1") as dirs:
        result = generate_xla(prompt)
    # dirs == {"trace": "...", "hlo": "..."}
"""

from __future__ import annotations

import os
from contextlib import contextmanager


@contextmanager
def trace_to(logdir: str = "/tmp/xla_trace"):
    """
    Context manager that records an XLA profiler trace to *logdir*.

    The trace can be visualised with TensorBoard::

        tensorboard --logdir <logdir>

    Requires the ``profile`` optional dependency group::

        pip install "xla-sharded[profile]"
    """
    try:
        import jax.profiler as _prof  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "XLA profiling requires the 'profile' extra: "
            "pip install 'xla-sharded[profile]'"
        ) from exc

    import jax.profiler as _prof  # re-import for use below

    os.makedirs(logdir, exist_ok=True)
    _prof.start_trace(logdir)
    try:
        yield logdir
    finally:
        _prof.stop_trace()


@contextmanager
def dump_hlo(outdir: str = "/tmp/hlo_dump"):
    """
    Context manager that instructs XLA to dump HLO IR for every compilation
    that occurs inside the ``with`` block.

    Files such as ``*.before_optimizations.txt`` and
    ``*.after_optimizations.txt`` are written to *outdir*.

    This works by temporarily setting ``XLA_FLAGS``.  For best results, wrap
    the very first JIT call before any other JAX initialisation::

        with dump_hlo("/tmp/hlo"):
            jax.jit(my_fn)(args)
    """
    os.makedirs(outdir, exist_ok=True)
    prev  = os.environ.get("XLA_FLAGS", "")
    flags = f"{prev} --xla_dump_to={outdir} --xla_dump_hlo_as_text".strip()
    os.environ["XLA_FLAGS"] = flags
    try:
        yield outdir
    finally:
        os.environ["XLA_FLAGS"] = prev


@contextmanager
def profile_and_dump(
    logdir: str = "/tmp/xla_trace",
    hlo_dir: str = "/tmp/hlo_dump",
):
    """
    Convenience wrapper that combines :func:`trace_to` and :func:`dump_hlo`.

    Example::

        with profile_and_dump("/tmp/run1") as dirs:
            result = generate_xla(prompt)
        # dirs == {"trace": "/tmp/run1/trace", "hlo": "/tmp/run1/hlo"}
        # View with: tensorboard --logdir /tmp/run1/trace
    """
    trace_path = os.path.join(logdir, "trace")
    hlo_path   = os.path.join(hlo_dir, "hlo")
    with dump_hlo(hlo_path):
        with trace_to(trace_path):
            yield {"trace": trace_path, "hlo": hlo_path}
