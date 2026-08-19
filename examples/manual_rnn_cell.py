"""Pedagogical implementation of a recurrent neural network cell."""

import torch
from torch import nn


class ManualRNNCell(nn.Module):
    """Compute one recurrent hidden state without using ``nn.RNNCell``."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # weight_xh, weight_hh, and bias_h correspond to W_x, W_h, and b_h.
        self.weight_xh = nn.Parameter(
            torch.randn(hidden_size, input_size) * 0.1
        )
        self.weight_hh = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.1
        )
        self.bias_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x_t: torch.Tensor,
        h_previous: torch.Tensor,
    ) -> torch.Tensor:
        """Return h_t for one position of a batch of sequences."""
        return torch.tanh(
            x_t @ self.weight_xh.T
            + h_previous @ self.weight_hh.T
            + self.bias_h
        )


def run_manual_rnn(
    cell: ManualRNNCell,
    sequence: torch.Tensor,
    h_0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unroll a manual RNN over a batch-first input sequence."""
    batch_size, sequence_length, _ = sequence.shape
    h_t = (
        sequence.new_zeros((batch_size, cell.hidden_size))
        if h_0 is None
        else h_0
    )
    hidden_states: list[torch.Tensor] = []

    for t in range(sequence_length):
        x_t = sequence[:, t, :]
        h_t = cell(x_t, h_t)
        hidden_states.append(h_t)

    output = torch.stack(hidden_states, dim=1)
    return output, h_t


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_cell = ManualRNNCell(input_size=3, hidden_size=4)
    demo_sequence = torch.randn(2, 5, 3)
    demo_output, demo_h_n = run_manual_rnn(demo_cell, demo_sequence)
    print("output shape:", tuple(demo_output.shape))
    print("h_n shape:", tuple(demo_h_n.shape))
