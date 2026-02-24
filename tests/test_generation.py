"""Phase 2 tests for naive generation baseline."""

from engine.generate_naive import generate_naive
from tokenizer.tokenizer import DummyTokenizer


def test_generate_naive_returns_expected_lengths():
    tok = DummyTokenizer()
    prompt = "hello"
    max_new = 4

    result = generate_naive(prompt, max_new_tokens=max_new, seed=0, tokenizer=tok)

    assert len(result.prompt_ids) == 1 + len(prompt)
    assert 1 <= len(result.generated_ids) <= max_new
    assert len(result.all_ids) == len(result.prompt_ids) + len(result.generated_ids)


def test_generate_naive_decodes_output():
    result = generate_naive("abc", max_new_tokens=3, seed=1)
    assert isinstance(result.decoded_generated_text, str)
    assert isinstance(result.decoded_all_text, str)

