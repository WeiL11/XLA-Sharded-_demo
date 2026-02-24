"""Phase 2 tests for naive generation baseline."""

from engine.generate_naive import generate_naive
from engine.generate_xla import generate_xla
from configs.model_config import ModelConfig
from tokenizer.tokenizer import DummyTokenizer


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=256,
        max_seq_len=128,
        mlp_hidden=128,
    )


def test_generate_naive_returns_expected_lengths():
    tok = DummyTokenizer()
    prompt = "hello"
    max_new = 4

    result = generate_naive(
        prompt,
        max_new_tokens=max_new,
        seed=0,
        tokenizer=tok,
        config=_tiny_config(),
    )

    assert len(result.prompt_ids) == 1 + len(prompt)
    assert 1 <= len(result.generated_ids) <= max_new
    assert len(result.all_ids) == len(result.prompt_ids) + len(result.generated_ids)


def test_generate_naive_decodes_output():
    result = generate_naive("abc", max_new_tokens=3, seed=1, config=_tiny_config())
    assert isinstance(result.decoded_generated_text, str)
    assert isinstance(result.decoded_all_text, str)


def test_generate_xla_matches_naive_ids():
    cfg = _tiny_config()
    prompt = "abc"
    naive = generate_naive(prompt, max_new_tokens=8, seed=7, config=cfg)
    xla = generate_xla(prompt, max_new_tokens=8, seed=7, config=cfg)
    assert naive.generated_ids == xla.generated_ids
