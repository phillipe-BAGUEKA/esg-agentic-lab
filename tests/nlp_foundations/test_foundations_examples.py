import pytest

from prototypes.nlp_foundations.bigram_language_model import (
    count_bigrams,
    generate_tokens,
    next_token_probabilities,
    predict_next_token,
)
from prototypes.nlp_foundations.tokenization_demo import (
    build_vocabulary,
    decode,
    detokenize,
    encode,
    tokenize_words,
)


def test_tokenize_words_lowercases_and_isolates_period() -> None:
    assert tokenize_words("BNP réduit ses émissions.") == [
        "bnp",
        "réduit",
        "ses",
        "émissions",
        ".",
    ]


def test_build_vocabulary_is_deterministic_and_reserves_unknown() -> None:
    tokens = ["ses", "bnp", "ses", ".", "<UNK>"]

    assert build_vocabulary(tokens) == {
        "<UNK>": 0,
        ".": 1,
        "bnp": 2,
        "ses": 3,
    }


def test_encode_known_tokens() -> None:
    vocabulary = {"<UNK>": 0, "bnp": 1, "réduit": 2}

    assert encode(["bnp", "réduit"], vocabulary) == [1, 2]


def test_encode_unknown_token_uses_unknown_id() -> None:
    vocabulary = {"<UNK>": 0, "bnp": 1}

    assert encode(["bnp", "augmente"], vocabulary) == [1, 0]


def test_decode_ids_including_unknown() -> None:
    vocabulary = {"<UNK>": 0, "bnp": 1}

    assert decode([1, 0], vocabulary) == ["bnp", "<UNK>"]


def test_detokenize_reattaches_period() -> None:
    tokens = ["bnp", "réduit", "ses", "émissions", "."]

    assert detokenize(tokens) == "bnp réduit ses émissions."


def test_count_bigrams() -> None:
    sentences = [
        ["bnp", "réduit", "ses", "émissions"],
        ["bnp", "réduit", "ses", "coûts"],
        ["sg", "augmente", "ses", "émissions"],
    ]

    assert count_bigrams(sentences) == {
        ("bnp", "réduit"): 2,
        ("réduit", "ses"): 2,
        ("ses", "émissions"): 2,
        ("ses", "coûts"): 1,
        ("sg", "augmente"): 1,
        ("augmente", "ses"): 1,
    }


def test_count_bigrams_does_not_cross_sentence_boundaries() -> None:
    counts = count_bigrams([["a", "b"], ["c", "d"]])

    assert counts == {("a", "b"): 1, ("c", "d"): 1}
    assert ("b", "c") not in counts


def test_next_token_probabilities() -> None:
    bigram_counts = {
        ("ses", "émissions"): 2,
        ("ses", "coûts"): 1,
    }

    assert next_token_probabilities("ses", bigram_counts) == pytest.approx(
        {"émissions": 2 / 3, "coûts": 1 / 3}
    )


def test_next_token_probabilities_without_transition() -> None:
    bigram_counts = {("bnp", "réduit"): 1}

    assert next_token_probabilities("émissions", bigram_counts) == {}


def test_predict_next_token_uses_greedy_decoding() -> None:
    bigram_counts = {
        ("bnp", "réduit"): 3,
        ("bnp", "augmente"): 1,
    }

    assert predict_next_token("bnp", bigram_counts) == "réduit"


def test_predict_next_token_without_candidate() -> None:
    bigram_counts = {("bnp", "réduit"): 1}

    assert predict_next_token("émissions", bigram_counts) is None


def test_generate_tokens_stops_without_transition() -> None:
    bigram_counts = {
        ("bnp", "réduit"): 3,
        ("bnp", "augmente"): 1,
        ("réduit", "ses"): 2,
        ("réduit", "les"): 1,
        ("ses", "émissions"): 2,
    }

    assert generate_tokens("bnp", bigram_counts, max_new_tokens=4) == [
        "bnp",
        "réduit",
        "ses",
        "émissions",
    ]


def test_generate_tokens_respects_max_new_tokens() -> None:
    bigram_counts = {
        ("bnp", "réduit"): 3,
        ("bnp", "augmente"): 1,
        ("réduit", "ses"): 2,
        ("réduit", "les"): 1,
        ("ses", "émissions"): 2,
    }

    assert generate_tokens("bnp", bigram_counts, max_new_tokens=2) == [
        "bnp",
        "réduit",
        "ses",
    ]
