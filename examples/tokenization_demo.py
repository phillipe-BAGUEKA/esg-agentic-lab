"""Minimal word-level tokenization example for LLM foundations."""


def tokenize_words(text: str) -> list[str]:
    """Lowercase text, isolate periods, and split it into words."""
    return text.lower().replace(".", " . ").split()


def build_vocabulary(tokens: list[str]) -> dict[str, int]:
    """Build a deterministic vocabulary with a reserved unknown token."""
    vocabulary = {"<UNK>": 0}
    for token in sorted(set(tokens)):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encode(
    tokens: list[str],
    vocabulary: dict[str, int],
) -> list[int]:
    """Convert tokens to IDs, using the unknown ID when needed."""
    unknown_id = vocabulary["<UNK>"]
    return [vocabulary.get(token, unknown_id) for token in tokens]


def decode(
    token_ids: list[int],
    vocabulary: dict[str, int],
) -> list[str]:
    """Convert token IDs back to tokens."""
    id_to_token = {token_id: token for token, token_id in vocabulary.items()}
    return [id_to_token[token_id] for token_id in token_ids]


def detokenize(tokens: list[str]) -> str:
    """Approximately reconstruct text from tokens."""
    return " ".join(tokens).replace(" .", ".")
