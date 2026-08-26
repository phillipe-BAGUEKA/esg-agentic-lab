"""Manual multi-head causal self-attention for learning."""

from math import sqrt

import torch
from torch import nn


class ManualCausalSelfAttention(nn.Module):
    """Compute causal self-attention without high-level attention modules."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be greater than zero")
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return contextualized states with shape ``(B, T, d_model)``."""
        return self.inspect_attention(hidden_states)["output"]

    def inspect_attention(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the output and intermediate attention tensors."""
        self._validate_input(hidden_states)
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)

        query_heads = self._split_heads(query)
        key_heads = self._split_heads(key)
        value_heads = self._split_heads(value)

        attention_scores = (
            query_heads @ key_heads.transpose(-2, -1)
        ) / sqrt(self.d_head)
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            )
        )
        masked_scores = attention_scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )
        attention_weights = torch.softmax(masked_scores, dim=-1)
        head_outputs = attention_weights @ value_heads
        concatenated_output = (
            head_outputs.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )
        output = self.output_projection(concatenated_output)

        return {
            "query": query,
            "query_heads": query_heads,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "attention_scores": attention_scores,
            "causal_mask": causal_mask,
            "masked_scores": masked_scores,
            "attention_weights": attention_weights,
            "head_outputs": head_outputs,
            "concatenated_output": concatenated_output,
            "output": output,
        }

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return (
            tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.d_head,
            )
            .transpose(1, 2)
            .contiguous()
        )

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
