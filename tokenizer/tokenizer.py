"""
V1: DummyTokenizer — maps characters to integer token IDs via hashing.
    Used for pipeline correctness testing without a real vocabulary.

V2: GemmaTokenizer — SentencePiece tokenizer via HuggingFace transformers.
    Requires: pip install xla-sharded[gemma]
    Note: switching to V2 requires updating vocab_size=256_000 in both
    ModelConfig and DraftConfig (Gemma has 256k tokens vs 32k here).
"""


class DummyTokenizer:
    """
    V1 tokenizer for pipeline testing.

    Encodes text by hashing each character to a token ID in [0, vocab_size).
    Decoded output is placeholder "[ID]" tokens — not meaningful text.
    """

    vocab_size: int = 32_000
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0

    def encode(self, text: str) -> list[int]:
        # Use a stable character mapping so token IDs are reproducible across runs.
        return [self.bos_id] + [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: list[int]) -> str:
        special = {self.bos_id, self.eos_id, self.pad_id}
        return " ".join(f"[{i}]" for i in ids if i not in special)


class GemmaTokenizer:
    """
    V2 tokenizer using the Gemma SentencePiece model via HuggingFace transformers.

    Requires: pip install xla-sharded[gemma]
    Important: vocab_size=256_000 — update ModelConfig and DraftConfig before use.
    """

    def __init__(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "GemmaTokenizer requires the 'gemma' extra: "
                "pip install 'xla-sharded[gemma]'"
            ) from e
        self._tok = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.vocab_size = self._tok.vocab_size
        self.bos_id = self._tok.bos_token_id
        self.eos_id = self._tok.eos_token_id
        self.pad_id = self._tok.pad_token_id or 0

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
