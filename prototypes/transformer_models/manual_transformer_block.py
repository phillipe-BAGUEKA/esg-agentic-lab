"""Manual pre-normalized causal Transformer block for learning."""

import torch
from torch import nn

if __package__:
    from .manual_causal_self_attention import ManualCausalSelfAttention
else:
    from manual_causal_self_attention import ManualCausalSelfAttention


class PositionWiseFeedForward(nn.Module):
    """Apply the same two-layer feed-forward network at every position."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be greater than zero")
        if d_ff <= 0:
            raise ValueError("d_ff must be greater than zero")

        self.d_model = d_model
        self.d_ff = d_ff
        self.input_projection = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(d_ff, d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return transformed states with shape ``(B, T, d_model)``."""
        return self.inspect_forward(hidden_states)["output"]

    def inspect_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the output and intermediate feed-forward tensors."""
        self._validate_input(hidden_states)
        expanded = self.input_projection(hidden_states)
        activated = self.activation(expanded)
        output = self.output_projection(activated)

        return {
            "expanded": expanded,
            "activated": activated,
            "output": output,
        }

    def _validate_input(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "(batch_size, sequence_length, d_model)"
            )
        if hidden_states.shape[1] == 0:
            raise ValueError("sequence_length must be greater than zero")
        if hidden_states.shape[2] != self.d_model:
            raise ValueError(
                f"hidden_states must have d_model={self.d_model}, "
                f"got {hidden_states.shape[2]}"
            )


class ManualTransformerBlock(nn.Module):
    """Combine causal attention and a feed-forward network with pre-norm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be greater than zero")
        if d_ff <= 0:
            raise ValueError("d_ff must be greater than zero")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = ManualCausalSelfAttention(d_model, num_heads)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the block output with shape ``(B, T, d_model)``."""
        return self.inspect_block(hidden_states)["output"]

    def inspect_block(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the output and intermediate pre-norm block tensors."""
        self._validate_input(hidden_states)

        attention_input = self.attention_norm(hidden_states)
        attention_output = self.attention(attention_input)
        after_attention = hidden_states + attention_output

        feed_forward_input = self.feed_forward_norm(after_attention)
        feed_forward_details = self.feed_forward.inspect_forward(
            feed_forward_input
        )
        feed_forward_output = feed_forward_details["output"]
        output = after_attention + feed_forward_output

        return {
            "attention_input": attention_input,
            "attention_output": attention_output,
            "after_attention": after_attention,
            "feed_forward_input": feed_forward_input,
            "expanded": feed_forward_details["expanded"],
            "activated": feed_forward_details["activated"],
            "feed_forward_output": feed_forward_output,
            "output": output,
        }

    def _validate_input(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "(batch_size, sequence_length, d_model)"
            )
        if hidden_states.shape[1] == 0:
            raise ValueError("sequence_length must be greater than zero")
        if hidden_states.shape[2] != self.d_model:
            raise ValueError(
                f"hidden_states must have d_model={self.d_model}, "
                f"got {hidden_states.shape[2]}"
            )
