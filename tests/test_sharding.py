import pytest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from engine.sharder import ShardSpecs, create_mesh, shard_array, shard_kv_cache


@pytest.mark.skipif(len(jax.devices()) < 2, reason="Need >=2 devices for mesh sharding tests")
def test_create_mesh_and_shard_array():
    mesh = create_mesh(2)
    arr = jnp.ones((8, 8), dtype=jnp.float32)
    sharded = shard_array(arr, mesh, ShardSpecs().wq_wk_wv)
    assert isinstance(sharded.sharding, NamedSharding)
    assert sharded.sharding.spec == ShardSpecs().wq_wk_wv


@pytest.mark.skipif(len(jax.devices()) < 2, reason="Need >=2 devices for mesh sharding tests")
def test_shard_kv_cache_uses_expected_spec():
    mesh = create_mesh(2)
    cache = jnp.zeros((2, 2, 1, 16, 4, 8), dtype=jnp.bfloat16)
    sharded = shard_kv_cache(cache, mesh)
    assert isinstance(sharded.sharding, NamedSharding)
    assert sharded.sharding.spec == ShardSpecs().kv_cache

