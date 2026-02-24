"""Sharding helpers for tensor-parallel placement with JAX NamedSharding."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding


@dataclass(frozen=True)
class ShardSpecs:
    """Canonical partition specs used by the project."""

    wq_wk_wv: P = P(None, "model")
    wo_wdown: P = P("model", None)
    wgate_wup: P = P(None, "model")
    embed: P = P("model", None)
    lm_head: P = P(None, "model")
    replicated: P = P()
    kv_cache: P = P(None, None, None, None, "model", None)


def create_mesh(num_devices: int = 2) -> Mesh:
    """Create a 1D mesh on axis 'model' from the first N devices."""
    devices = jax.devices()
    if len(devices) < num_devices:
        raise ValueError(f"Need at least {num_devices} devices, found {len(devices)}")
    return Mesh(devices[:num_devices], axis_names=("model",))


def create_single_device_sharding(device_idx: int = 0) -> SingleDeviceSharding:
    """Create SingleDeviceSharding for draft model placement."""
    return SingleDeviceSharding(jax.devices()[device_idx])


def named_sharding(mesh: Mesh, spec: P) -> NamedSharding:
    return NamedSharding(mesh, spec)


def shard_array(arr: jnp.ndarray, mesh: Mesh, spec: P) -> jnp.ndarray:
    """Place a dense array according to a NamedSharding PartitionSpec."""
    return jax.device_put(arr, NamedSharding(mesh, spec))


def shard_kv_cache(cache: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """Place KV cache with heads axis sharded on 'model'."""
    return shard_array(cache, mesh, ShardSpecs().kv_cache)

