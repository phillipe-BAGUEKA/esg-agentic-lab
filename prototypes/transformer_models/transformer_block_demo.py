"""Demonstrate a manual pre-normalized causal Transformer block."""

import torch
from torch import nn

if __package__:
    from .manual_transformer_block import ManualTransformerBlock
else:
    from manual_transformer_block import ManualTransformerBlock


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in a module."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    """Run a deterministic causal Transformer-block example."""
    torch.manual_seed(0)
    batch_size = 2
    sequence_length = 5
    d_model = 8
    num_heads = 2
    d_ff = 32

    hidden_states = torch.randn(batch_size, sequence_length, d_model)
    block = ManualTransformerBlock(d_model, num_heads, d_ff)
    details = block.inspect_block(hidden_states)

    print("input:", tuple(hidden_states.shape))
    for name in (
        "attention_input",
        "attention_output",
        "after_attention",
        "feed_forward_input",
        "expanded",
        "activated",
        "feed_forward_output",
        "output",
    ):
        print(f"{name}:", tuple(details[name].shape))

    expected_shape = (batch_size, sequence_length, d_model)
    output_has_expected_shape = tuple(details["output"].shape) == expected_shape
    print("output has expected shape (2, 5, 8):", output_has_expected_shape)
    assert output_has_expected_shape

    modified_states = hidden_states.clone()
    modified_states[:, -1, :] += torch.tensor(
        [10.0, -9.0, 8.0, -7.0, 6.0, -5.0, 4.0, -3.0]
    )
    modified_output = block(modified_states)
    earlier_outputs_are_unchanged = torch.allclose(
        details["output"][:, :-1, :],
        modified_output[:, :-1, :],
    )
    print(
        "changing the last token preserves earlier outputs:",
        earlier_outputs_are_unchanged,
    )
    assert earlier_outputs_are_unchanged

    print("feed-forward parameters:", count_parameters(block.feed_forward))
    print("complete block parameters:", count_parameters(block))


if __name__ == "__main__":
    main()
