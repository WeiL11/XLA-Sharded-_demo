"""
Draft model factory. The draft model uses the same Transformer architecture as
the target, but with DraftConfig (2 layers instead of 12). It lives entirely on
GPU 0 (unsharded) — placement is handled in engine/sharder.py, not here.
"""

from flax import nnx

from configs.model_config import DraftConfig
from models.transformer import Transformer


def create_draft_model(rngs: nnx.Rngs | None = None) -> Transformer:
    """Instantiate the 2-layer draft model with default DraftConfig."""
    if rngs is None:
        rngs = nnx.Rngs(params=0)
    return Transformer(DraftConfig(), rngs=rngs)
