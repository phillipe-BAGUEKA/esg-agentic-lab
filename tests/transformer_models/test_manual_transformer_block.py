import pytest
import torch

from prototypes.transformer_models.manual_transformer_block import (
    ManualTransformerBlock,
    PositionWiseFeedForward,
)
from prototypes.transformer_models.transformer_block_demo import main


def test_feed_forward_validates_arguments_and_input_shape() -> None:
    with pytest.raises(ValueError, match="d_model must be greater"):
        PositionWiseFeedForward(d_model=0, d_ff=32)
    with pytest.raises(ValueError, match="d_ff must be greater"):
        PositionWiseFeedForward(d_model=8, d_ff=0)

    feed_forward = PositionWiseFeedForward(d_model=8, d_ff=32)
    with pytest.raises(ValueError, match="hidden_states must have shape"):
        feed_forward(torch.zeros(2, 8))
    with pytest.raises(ValueError, match="sequence_length"):
        feed_forward(torch.zeros(1, 0, 8))
    with pytest.raises(ValueError, match="d_model=8"):
        feed_forward(torch.zeros(1, 3, 7))


def test_transformer_block_validates_arguments_and_input_shape() -> None:
    with pytest.raises(ValueError, match="d_model must be greater"):
        ManualTransformerBlock(d_model=0, num_heads=1, d_ff=32)
    with pytest.raises(ValueError, match="num_heads must be greater"):
        ManualTransformerBlock(d_model=8, num_heads=0, d_ff=32)
    with pytest.raises(ValueError, match="divisible"):
        ManualTransformerBlock(d_model=8, num_heads=3, d_ff=32)
    with pytest.raises(ValueError, match="d_ff must be greater"):
        ManualTransformerBlock(d_model=8, num_heads=2, d_ff=0)

    block = ManualTransformerBlock(d_model=8, num_heads=2, d_ff=32)
    with pytest.raises(ValueError, match="hidden_states must have shape"):
        block(torch.zeros(2, 8))
    with pytest.raises(ValueError, match="sequence_length"):
        block(torch.zeros(1, 0, 8))
    with pytest.raises(ValueError, match="d_model=8"):
        block(torch.zeros(1, 3, 7))


def test_feed_forward_intermediate_shapes_and_exact_equation() -> None:
    torch.manual_seed(0)
    feed_forward = PositionWiseFeedForward(d_model=8, d_ff=32)
    hidden_states = torch.randn(2, 5, 8)

    details = feed_forward.inspect_forward(hidden_states)
    expected_expanded = feed_forward.input_projection(hidden_states)
    expected_activated = feed_forward.activation(expected_expanded)
    expected_output = feed_forward.output_projection(expected_activated)

    assert details["expanded"].shape == (2, 5, 32)
    assert details["activated"].shape == (2, 5, 32)
    assert details["output"].shape == (2, 5, 8)
    torch.testing.assert_close(details["expanded"], expected_expanded)
    torch.testing.assert_close(details["activated"], expected_activated)
    torch.testing.assert_close(details["output"], expected_output)
    torch.testing.assert_close(feed_forward(hidden_states), expected_output)


def test_block_intermediate_shapes_and_residual_equations() -> None:
    torch.manual_seed(0)
    block = ManualTransformerBlock(d_model=8, num_heads=2, d_ff=32)
    hidden_states = torch.randn(2, 5, 8)

    details = block.inspect_block(hidden_states)

    for name in (
        "attention_input",
        "attention_output",
        "after_attention",
        "feed_forward_input",
        "feed_forward_output",
        "output",
    ):
        assert details[name].shape == (2, 5, 8)
    assert details["expanded"].shape == (2, 5, 32)
    assert details["activated"].shape == (2, 5, 32)

    torch.testing.assert_close(
        details["attention_input"],
        block.attention_norm(hidden_states),
    )
    torch.testing.assert_close(
        details["attention_output"],
        block.attention(details["attention_input"]),
    )
    torch.testing.assert_close(
        details["after_attention"],
        hidden_states + details["attention_output"],
    )
    torch.testing.assert_close(
        details["feed_forward_input"],
        block.feed_forward_norm(details["after_attention"]),
    )
    torch.testing.assert_close(
        details["feed_forward_output"],
        block.feed_forward(details["feed_forward_input"]),
    )
    torch.testing.assert_close(
        details["output"],
        details["after_attention"] + details["feed_forward_output"],
    )
    torch.testing.assert_close(block(hidden_states), details["output"])


def test_changing_future_token_preserves_previous_block_outputs() -> None:
    torch.manual_seed(0)
    block = ManualTransformerBlock(d_model=8, num_heads=2, d_ff=32)
    hidden_states = torch.randn(2, 5, 8)
    modified_states = hidden_states.clone()
    modified_states[:, -1, :] += torch.tensor(
        [10.0, -9.0, 8.0, -7.0, 6.0, -5.0, 4.0, -3.0]
    )

    original_output = block(hidden_states)
    modified_output = block(modified_states)

    torch.testing.assert_close(
        original_output[:, :-1, :],
        modified_output[:, :-1, :],
    )


def test_changing_past_token_can_change_later_block_outputs() -> None:
    torch.manual_seed(0)
    block = ManualTransformerBlock(d_model=8, num_heads=2, d_ff=32)
    hidden_states = torch.randn(1, 5, 8)
    modified_states = hidden_states.clone()
    modified_states[:, 0, :] += torch.tensor(
        [10.0, -9.0, 8.0, -7.0, 6.0, -5.0, 4.0, -3.0]
    )

    original_output = block(hidden_states)
    modified_output = block(modified_states)

    assert not torch.allclose(
        original_output[:, -1, :],
        modified_output[:, -1, :],
    )


def test_zero_sublayer_outputs_make_block_an_exact_identity() -> None:
    torch.manual_seed(0)
    block = ManualTransformerBlock(d_model=8, num_heads=2, d_ff=32)
    with torch.no_grad():
        block.attention.output_projection.weight.zero_()
        block.attention.output_projection.bias.zero_()
        block.feed_forward.output_projection.weight.zero_()
        block.feed_forward.output_projection.bias.zero_()
    hidden_states = torch.randn(2, 5, 8)

    output = block(hidden_states)

    assert torch.equal(output, hidden_states)


def test_gradients_reach_input_and_every_trainable_parameter() -> None:
    torch.manual_seed(0)
    block = ManualTransformerBlock(d_model=8, num_heads=2, d_ff=32)
    hidden_states = torch.randn(2, 5, 8, requires_grad=True)

    block(hidden_states).square().mean().backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in block.parameters())


def test_parameter_counts() -> None:
    block = ManualTransformerBlock(d_model=8, num_heads=2, d_ff=32)

    attention_parameters = sum(
        parameter.numel() for parameter in block.attention.parameters()
    )
    layer_norm_parameters = sum(
        parameter.numel()
        for layer_norm in (block.attention_norm, block.feed_forward_norm)
        for parameter in layer_norm.parameters()
    )
    feed_forward_parameters = sum(
        parameter.numel() for parameter in block.feed_forward.parameters()
    )
    block_parameters = sum(
        parameter.numel() for parameter in block.parameters()
    )

    assert attention_parameters == 288
    assert layer_norm_parameters == 32
    assert feed_forward_parameters == 552
    assert block_parameters == 872


def test_demo_executes_and_reports_shapes_counts_and_causality(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()

    output = capsys.readouterr().out
    assert "output has expected shape (2, 5, 8): True" in output
    assert "changing the last token preserves earlier outputs: True" in output
    assert "feed-forward parameters: 552" in output
    assert "complete block parameters: 872" in output
