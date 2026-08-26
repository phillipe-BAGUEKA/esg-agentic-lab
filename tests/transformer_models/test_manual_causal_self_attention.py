import pytest
import torch

from prototypes.transformer_models.causal_self_attention_demo import main
from prototypes.transformer_models.manual_causal_self_attention import (
    ManualCausalSelfAttention,
)


def set_identity_projections(model: ManualCausalSelfAttention) -> None:
    """Make every projection an identity map for deterministic reasoning."""
    with torch.no_grad():
        for projection in (
            model.query_projection,
            model.key_projection,
            model.value_projection,
            model.output_projection,
        ):
            projection.weight.copy_(torch.eye(model.d_model))
            projection.bias.zero_()


def test_attention_validates_arguments_and_input_shape() -> None:
    with pytest.raises(ValueError, match="d_model must be greater"):
        ManualCausalSelfAttention(d_model=0, num_heads=1)
    with pytest.raises(ValueError, match="num_heads must be greater"):
        ManualCausalSelfAttention(d_model=8, num_heads=0)
    with pytest.raises(ValueError, match="divisible"):
        ManualCausalSelfAttention(d_model=8, num_heads=3)

    model = ManualCausalSelfAttention(d_model=8, num_heads=2)
    with pytest.raises(ValueError, match="hidden_states must have shape"):
        model(torch.zeros(3, 8))
    with pytest.raises(ValueError, match="sequence_length"):
        model(torch.zeros(1, 0, 8))
    with pytest.raises(ValueError, match="d_model=8"):
        model(torch.zeros(1, 3, 7))


def test_attention_intermediate_and_output_shapes() -> None:
    torch.manual_seed(0)
    model = ManualCausalSelfAttention(d_model=8, num_heads=2)
    hidden_states = torch.randn(2, 3, 8)

    details = model.inspect_attention(hidden_states)

    assert model.d_head == 4
    assert details["query"].shape == (2, 3, 8)
    assert details["query_heads"].shape == (2, 2, 3, 4)
    assert details["key_heads"].shape == (2, 2, 3, 4)
    assert details["value_heads"].shape == (2, 2, 3, 4)
    assert details["attention_scores"].shape == (2, 2, 3, 3)
    assert details["causal_mask"].shape == (3, 3)
    assert details["masked_scores"].shape == (2, 2, 3, 3)
    assert details["attention_weights"].shape == (2, 2, 3, 3)
    assert details["head_outputs"].shape == (2, 2, 3, 4)
    assert details["concatenated_output"].shape == (2, 3, 8)
    assert details["output"].shape == (2, 3, 8)
    assert model(hidden_states).shape == (2, 3, 8)


def test_attention_weights_sum_to_one_and_future_weights_are_zero() -> None:
    torch.manual_seed(0)
    model = ManualCausalSelfAttention(d_model=8, num_heads=2)
    details = model.inspect_attention(torch.randn(2, 4, 8))
    weights = details["attention_weights"]

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(2, 2, 4),
    )
    future_mask = torch.triu(
        torch.ones(4, 4, dtype=torch.bool),
        diagonal=1,
    )
    future_weights = weights[..., future_mask]
    assert torch.equal(future_weights, torch.zeros_like(future_weights))
    assert torch.isneginf(details["masked_scores"][..., future_mask]).all()


def test_changing_future_token_preserves_previous_outputs() -> None:
    torch.manual_seed(0)
    model = ManualCausalSelfAttention(d_model=8, num_heads=2)
    hidden_states = torch.randn(1, 3, 8)
    modified_states = hidden_states.clone()
    modified_states[:, -1, :] += 100

    original_output = model(hidden_states)
    modified_output = model(modified_states)

    torch.testing.assert_close(
        original_output[:, :-1, :],
        modified_output[:, :-1, :],
    )


def test_position_depends_on_available_past() -> None:
    model = ManualCausalSelfAttention(d_model=4, num_heads=2)
    set_identity_projections(model)
    hidden_states = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0]]]
    )
    modified_states = hidden_states.clone()
    modified_states[:, 0, :] = torch.tensor([[-2.0, 0.0, 0.0, 0.0]])

    original_output = model(hidden_states)
    modified_output = model(modified_states)

    assert not torch.allclose(
        original_output[:, -1, :],
        modified_output[:, -1, :],
    )


def test_gradients_reach_input_and_every_projection_parameter() -> None:
    torch.manual_seed(0)
    model = ManualCausalSelfAttention(d_model=8, num_heads=2)
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    model(hidden_states).square().sum().backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_attention_parameter_count() -> None:
    model = ManualCausalSelfAttention(d_model=8, num_heads=2)

    assert sum(parameter.numel() for parameter in model.parameters()) == 288


def test_demo_executes_and_reports_causality(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()

    output = capsys.readouterr().out
    assert "future attention weights are exactly zero: True" in output
    assert "changing the last token preserves earlier outputs: True" in output
