"""Manual GRU cell and single-layer sequence wrapper for learning."""

from math import sqrt

import torch
from torch import nn


class ManualGRUCell(nn.Module):
    """Compute one GRU step with PyTorch's ``r, z, n`` gate order."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be greater than zero")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Three stacked gates: reset, update, and new candidate.
        self.weight_ih = nn.Parameter(
            torch.empty(3 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(3 * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with the same bounds documented by PyTorch."""
        bound = 1 / sqrt(self.hidden_size)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.uniform_(-bound, bound)

    def forward(
        self,
        x_t: torch.Tensor,
        h_previous: torch.Tensor,
    ) -> torch.Tensor:
        """Return the next hidden state ``h_t``."""
        return self.inspect_step(x_t, h_previous)["h_t"]

    def inspect_step(
        self,
        x_t: torch.Tensor,
        h_previous: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return one GRU step, its gates, and both hidden contributions."""
        self._validate_step_inputs(x_t, h_previous)

        input_gates = x_t @ self.weight_ih.T + self.bias_ih
        hidden_gates = h_previous @ self.weight_hh.T + self.bias_hh
        input_r, input_z, input_n = input_gates.chunk(3, dim=1)
        hidden_r, hidden_z, hidden_n = hidden_gates.chunk(3, dim=1)

        r_t = torch.sigmoid(input_r + hidden_r)
        z_t = torch.sigmoid(input_z + hidden_z)
        n_t = torch.tanh(input_n + r_t * hidden_n)
        old_contribution = z_t * h_previous
        new_contribution = (1 - z_t) * n_t
        h_t = old_contribution + new_contribution

        return {
            "r_t": r_t,
            "z_t": z_t,
            "n_t": n_t,
            "old_contribution": old_contribution,
            "new_contribution": new_contribution,
            "h_t": h_t,
        }

    def _validate_step_inputs(
        self,
        x_t: torch.Tensor,
        h_previous: torch.Tensor,
    ) -> None:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape (batch_size, input_size)")
        if x_t.shape[1] != self.input_size:
            raise ValueError(
                f"x_t must have input_size={self.input_size}, "
                f"got {x_t.shape[1]}"
            )

        expected_shape = (x_t.shape[0], self.hidden_size)
        if tuple(h_previous.shape) != expected_shape:
            raise ValueError(
                f"h_previous must have shape {expected_shape}, "
                f"got {tuple(h_previous.shape)}"
            )


class ManualGRU(nn.Module):
    """Unroll one unidirectional GRU layer over batch-first sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("ManualGRU supports only batch_first=True")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = ManualGRUCell(input_size, hidden_size)

    def forward(
        self,
        sequence: torch.Tensor,
        h_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``output, h_n`` using PyTorch-compatible shapes."""
        self._validate_sequence(sequence)
        batch_size, sequence_length, _ = sequence.shape

        if h_0 is None:
            h_t = sequence.new_zeros((batch_size, self.hidden_size))
        else:
            expected_shape = (1, batch_size, self.hidden_size)
            if tuple(h_0.shape) != expected_shape:
                raise ValueError(
                    f"h_0 must have shape {expected_shape}, "
                    f"got {tuple(h_0.shape)}"
                )
            h_t = h_0[0]

        hidden_states: list[torch.Tensor] = []
        for t in range(sequence_length):
            x_t = sequence[:, t, :]
            h_t = self.cell(x_t, h_t)
            hidden_states.append(h_t)

        output = torch.stack(hidden_states, dim=1)
        return output, h_t.unsqueeze(0)

    def _validate_sequence(self, sequence: torch.Tensor) -> None:
        if sequence.ndim != 3:
            raise ValueError(
                "sequence must have shape "
                "(batch_size, sequence_length, input_size)"
            )
        if sequence.shape[1] == 0:
            raise ValueError("sequence_length must be greater than zero")
        if sequence.shape[2] != self.input_size:
            raise ValueError(
                f"sequence must have input_size={self.input_size}, "
                f"got {sequence.shape[2]}"
            )
