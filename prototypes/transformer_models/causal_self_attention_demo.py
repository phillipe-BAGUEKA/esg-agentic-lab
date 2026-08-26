"""Demonstrate the intermediate tensors of causal self-attention."""

import torch

if __package__:
    from .manual_causal_self_attention import ManualCausalSelfAttention
else:
    from manual_causal_self_attention import ManualCausalSelfAttention


def rounded_matrix(tensor: torch.Tensor) -> list[list[float]]:
    """Convert a small matrix to readable rounded Python values."""
    return [
        [round(value, 4) for value in row]
        for row in tensor.detach().tolist()
    ]


def main() -> None:
    """Run a deterministic three-token causal-attention example."""
    torch.manual_seed(0)
    tokens = ["BNP", "réduit", "émissions"]
    hidden_states = torch.randn(1, 3, 8)
    attention = ManualCausalSelfAttention(d_model=8, num_heads=2)
    details = attention.inspect_attention(hidden_states)

    print("Tokens:", tokens)
    print("input:", tuple(hidden_states.shape))
    for name in (
        "query",
        "query_heads",
        "key_heads",
        "value_heads",
        "attention_scores",
        "masked_scores",
        "attention_weights",
        "head_outputs",
        "concatenated_output",
        "output",
    ):
        print(f"{name}:", tuple(details[name].shape))

    print("causal_mask:")
    print(details["causal_mask"].to(dtype=torch.int64))
    print("head 0 attention weights:")
    print(rounded_matrix(details["attention_weights"][0, 0]))

    future_positions = torch.triu(
        torch.ones(3, 3, dtype=torch.bool),
        diagonal=1,
    )
    future_weights = details["attention_weights"][..., future_positions]
    print(
        "future attention weights are exactly zero:",
        torch.equal(future_weights, torch.zeros_like(future_weights)),
    )

    modified_states = hidden_states.clone()
    modified_states[:, -1, :] += 10
    modified_output = attention(modified_states)
    print(
        "changing the last token preserves earlier outputs:",
        torch.allclose(
            details["output"][:, :-1, :],
            modified_output[:, :-1, :],
        ),
    )


if __name__ == "__main__":
    main()
