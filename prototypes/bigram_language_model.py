"""Minimal bigram language model for teaching next-token prediction."""


def count_bigrams(
    sentences: list[list[str]],
) -> dict[tuple[str, str], int]:
    """Count adjacent token pairs without crossing sentence boundaries."""
    counts: dict[tuple[str, str], int] = {}

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            pair = (sentence[i], sentence[i + 1])
            counts[pair] = counts.get(pair, 0) + 1

    return counts


def next_token_probabilities(
    current_token: str,
    bigram_counts: dict[tuple[str, str], int],
) -> dict[str, float]:
    """Calculate empirical next-token probabilities for one token."""
    next_token_counts = {
        next_token: count
        for (token, next_token), count in bigram_counts.items()
        if token == current_token
    }
    total = sum(next_token_counts.values())

    if total == 0:
        return {}

    return {
        next_token: count / total
        for next_token, count in next_token_counts.items()
    }


def predict_next_token(
    current_token: str,
    bigram_counts: dict[tuple[str, str], int],
) -> str | None:
    """Greedily select the most probable observed next token."""
    probabilities = next_token_probabilities(current_token, bigram_counts)

    if not probabilities:
        return None

    return max(probabilities, key=probabilities.get)


def generate_tokens(
    start_token: str,
    bigram_counts: dict[tuple[str, str], int],
    max_new_tokens: int,
) -> list[str]:
    """Generate tokens until the limit or an unobserved transition is reached.

    No observed transition means only that the corpus has no continuation,
    not that the generated sequence reached a true linguistic ending.
    """
    generated = [start_token]
    current_token = start_token

    for _ in range(max_new_tokens):
        next_token = predict_next_token(current_token, bigram_counts)

        if next_token is None:
            break

        generated.append(next_token)
        current_token = next_token

    return generated
