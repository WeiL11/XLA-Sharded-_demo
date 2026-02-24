from configs.model_config import DraftConfig, ModelConfig
from engine.spec_dec import speculative_decode


def _tiny_target() -> ModelConfig:
    return ModelConfig(
        n_layers=3,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=256,
        max_seq_len=128,
        mlp_hidden=128,
    )


def _tiny_draft() -> DraftConfig:
    return DraftConfig(
        n_layers=1,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=256,
        max_seq_len=128,
        mlp_hidden=128,
    )


def test_spec_dec_generates_tokens_and_metrics():
    result = speculative_decode(
        "abc",
        max_new_tokens=10,
        k=3,
        seed=7,
        target_config=_tiny_target(),
        draft_config=_tiny_draft(),
    )
    assert 1 <= len(result.generated_ids) <= 10
    assert result.proposed_tokens >= result.accepted_tokens
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.rounds >= 1


def test_spec_dec_stops_at_max_tokens():
    result = speculative_decode(
        "xyz",
        max_new_tokens=5,
        k=4,
        seed=3,
        target_config=_tiny_target(),
        draft_config=_tiny_draft(),
    )
    assert len(result.generated_ids) <= 5

