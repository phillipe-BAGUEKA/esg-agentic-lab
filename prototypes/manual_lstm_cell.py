"""Manual LSTM cell and single-layer sequence wrapper for learning."""

from math import sqrt

import torch
from torch import nn


class ManualLSTMCell(nn.Module):
    """Compute one LSTM step with PyTorch's ``i, f, g, o`` gate order."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be greater than zero")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Four stacked gates: input, forget, candidate, and output.
        self.weight_ih = nn.Parameter(
            torch.empty(4 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
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
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next hidden and cell states, ``(h_t, c_t)``."""
        step = self.inspect_step(x_t, state)
        return step["h_t"], step["c_t"]

    def inspect_step(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return one LSTM step together with its four gate activations."""
        self._validate_step_inputs(x_t, state)
        h_previous, c_previous = state

        gates = (
            x_t @ self.weight_ih.T
            + self.bias_ih
            + h_previous @ self.weight_hh.T
            + self.bias_hh
        )
        i_linear, f_linear, g_linear, o_linear = gates.chunk(4, dim=1)

        i_t = torch.sigmoid(i_linear)
        f_t = torch.sigmoid(f_linear)
        g_t = torch.tanh(g_linear)
        o_t = torch.sigmoid(o_linear)
        c_t = f_t * c_previous + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return {
            "i_t": i_t,
            "f_t": f_t,
            "g_t": g_t,
            "o_t": o_t,
            "c_t": c_t,
            "h_t": h_t,
        }

    def _validate_step_inputs(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape (batch_size, input_size)")
        if x_t.shape[1] != self.input_size:
            raise ValueError(
                f"x_t must have input_size={self.input_size}, "
                f"got {x_t.shape[1]}"
            )
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("state must be a tuple (h_previous, c_previous)")

        expected_shape = (x_t.shape[0], self.hidden_size)
        h_previous, c_previous = state
        if tuple(h_previous.shape) != expected_shape:
            raise ValueError(
                f"h_previous must have shape {expected_shape}, "
                f"got {tuple(h_previous.shape)}"
            )
        if tuple(c_previous.shape) != expected_shape:
            raise ValueError(
                f"c_previous must have shape {expected_shape}, "
                f"got {tuple(c_previous.shape)}"
            )


class ManualLSTM(nn.Module):
    """Unroll one unidirectional LSTM layer over batch-first sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("ManualLSTM supports only batch_first=True")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(
        self,
        sequence: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Return ``output, (h_n, c_n)`` using PyTorch-compatible shapes."""
        self._validate_sequence(sequence)
        batch_size, sequence_length, _ = sequence.shape

        if state is None:
            h_t = sequence.new_zeros((batch_size, self.hidden_size))
            c_t = sequence.new_zeros((batch_size, self.hidden_size))
        else:
            if not isinstance(state, tuple) or len(state) != 2:
                raise ValueError("state must be a tuple (h_0, c_0)")
            h_n, c_n = state
            expected_shape = (1, batch_size, self.hidden_size)
            if tuple(h_n.shape) != expected_shape:
                raise ValueError(
                    f"h_0 must have shape {expected_shape}, "
                    f"got {tuple(h_n.shape)}"
                )
            if tuple(c_n.shape) != expected_shape:
                raise ValueError(
                    f"c_0 must have shape {expected_shape}, "
                    f"got {tuple(c_n.shape)}"
                )
            h_t = h_n[0]
            c_t = c_n[0]

        hidden_states: list[torch.Tensor] = []
        for t in range(sequence_length):
            x_t = sequence[:, t, :]
            h_t, c_t = self.cell(x_t, (h_t, c_t))
            hidden_states.append(h_t)

        output = torch.stack(hidden_states, dim=1)
        return output, (h_t.unsqueeze(0), c_t.unsqueeze(0))

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
