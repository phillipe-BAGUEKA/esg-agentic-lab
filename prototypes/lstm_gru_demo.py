"""Reproducible demonstration of the manual LSTM and GRU prototypes."""

import torch
from torch import nn

if __package__:
    from .manual_gru_cell import ManualGRU, ManualGRUCell
    from .manual_lstm_cell import ManualLSTM, ManualLSTMCell
else:
    from manual_gru_cell import ManualGRU, ManualGRUCell
    from manual_lstm_cell import ManualLSTM, ManualLSTMCell


def parameter_count(module: nn.Module) -> int:
    """Count all trainable scalar parameters in a module."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def copy_lstm_parameters(
    manual: ManualLSTM,
    reference: nn.LSTM,
) -> None:
    """Copy the manual cell parameters into a one-layer PyTorch LSTM."""
    with torch.no_grad():
        reference.weight_ih_l0.copy_(manual.cell.weight_ih)
        reference.weight_hh_l0.copy_(manual.cell.weight_hh)
        reference.bias_ih_l0.copy_(manual.cell.bias_ih)
        reference.bias_hh_l0.copy_(manual.cell.bias_hh)


def copy_gru_parameters(
    manual: ManualGRU,
    reference: nn.GRU,
) -> None:
    """Copy the manual cell parameters into a one-layer PyTorch GRU."""
    with torch.no_grad():
        reference.weight_ih_l0.copy_(manual.cell.weight_ih)
        reference.weight_hh_l0.copy_(manual.cell.weight_hh)
        reference.bias_ih_l0.copy_(manual.cell.bias_ih)
        reference.bias_hh_l0.copy_(manual.cell.bias_hh)


def rounded_values(tensor: torch.Tensor) -> list[list[float]]:
    """Format a small batch of gate values without requiring NumPy."""
    return [
        [round(value, 4) for value in row]
        for row in tensor.detach().tolist()
    ]


def show_small_sequence_gates() -> None:
    """Inspect every recurrent step of a tiny deterministic sequence."""
    small_sequence = torch.tensor(
        [[[0.2, -0.1], [0.0, 0.3], [-0.2, 0.1]]]
    )
    lstm_cell = ManualLSTMCell(input_size=2, hidden_size=2)
    gru_cell = ManualGRUCell(input_size=2, hidden_size=2)
    h_lstm = small_sequence.new_zeros((1, 2))
    c_lstm = small_sequence.new_zeros((1, 2))
    h_gru = small_sequence.new_zeros((1, 2))

    print("\nSmall deterministic sequence, inspected token by token")
    print("Gate dimensions are numeric features, not literal concepts.")
    for t in range(small_sequence.shape[1]):
        x_t = small_sequence[:, t, :]
        lstm_step = lstm_cell.inspect_step(x_t, (h_lstm, c_lstm))
        gru_step = gru_cell.inspect_step(x_t, h_gru)
        h_lstm = lstm_step["h_t"]
        c_lstm = lstm_step["c_t"]
        h_gru = gru_step["h_t"]

        print(f"token position {t}")
        print(
            "  LSTM i/f/g/o:",
            rounded_values(lstm_step["i_t"]),
            rounded_values(lstm_step["f_t"]),
            rounded_values(lstm_step["g_t"]),
            rounded_values(lstm_step["o_t"]),
        )
        print(
            "  GRU r/z/n:",
            rounded_values(gru_step["r_t"]),
            rounded_values(gru_step["z_t"]),
            rounded_values(gru_step["n_t"]),
        )


def main() -> None:
    """Run shape, parity, gate, and gradient demonstrations."""
    torch.manual_seed(0)
    input_size = 4
    hidden_size = 50
    sequence = torch.randn(8, 10, input_size, requires_grad=True)

    manual_lstm = ManualLSTM(input_size, hidden_size)
    reference_lstm = nn.LSTM(
        input_size,
        hidden_size,
        batch_first=True,
    )
    copy_lstm_parameters(manual_lstm, reference_lstm)
    lstm_output, (lstm_h_n, lstm_c_n) = manual_lstm(sequence)
    reference_lstm_output, _ = reference_lstm(sequence)

    print("Manual LSTM and matching PyTorch reference")
    print("  input:", tuple(sequence.shape))
    print("  output:", tuple(lstm_output.shape))
    print("  h_n:", tuple(lstm_h_n.shape))
    print("  c_n:", tuple(lstm_c_n.shape))
    print("  weight_ih_l0:", tuple(reference_lstm.weight_ih_l0.shape))
    print("  weight_hh_l0:", tuple(reference_lstm.weight_hh_l0.shape))
    print("  bias_ih_l0:", tuple(reference_lstm.bias_ih_l0.shape))
    print("  bias_hh_l0:", tuple(reference_lstm.bias_hh_l0.shape))
    print("  parameter count:", parameter_count(manual_lstm))
    print(
        "  output[:, -1, :] == h_n[0]:",
        torch.allclose(lstm_output[:, -1, :], lstm_h_n[0]),
    )
    print(
        "  manual output == PyTorch output:",
        torch.allclose(
            lstm_output,
            reference_lstm_output,
            rtol=1e-5,
            atol=1e-6,
        ),
    )

    manual_gru = ManualGRU(input_size, hidden_size)
    reference_gru = nn.GRU(
        input_size,
        hidden_size,
        batch_first=True,
    )
    copy_gru_parameters(manual_gru, reference_gru)
    gru_output, gru_h_n = manual_gru(sequence)
    reference_gru_output, _ = reference_gru(sequence)

    print("\nManual GRU and matching PyTorch reference")
    print("  output:", tuple(gru_output.shape))
    print("  h_n:", tuple(gru_h_n.shape))
    print("  weight_ih_l0:", tuple(reference_gru.weight_ih_l0.shape))
    print("  weight_hh_l0:", tuple(reference_gru.weight_hh_l0.shape))
    print("  bias_ih_l0:", tuple(reference_gru.bias_ih_l0.shape))
    print("  bias_hh_l0:", tuple(reference_gru.bias_hh_l0.shape))
    print("  parameter count:", parameter_count(manual_gru))
    print(
        "  output[:, -1, :] == h_n[0]:",
        torch.allclose(gru_output[:, -1, :], gru_h_n[0]),
    )
    print(
        "  manual output == PyTorch output:",
        torch.allclose(
            gru_output,
            reference_gru_output,
            rtol=1e-5,
            atol=1e-6,
        ),
    )

    show_small_sequence_gates()

    loss = (
        lstm_output[:, -1, :].square().mean()
        + gru_output[:, -1, :].square().mean()
    )
    loss.backward()
    lstm_has_gradients = all(
        parameter.grad is not None for parameter in manual_lstm.parameters()
    )
    gru_has_gradients = all(
        parameter.grad is not None for parameter in manual_gru.parameters()
    )
    first_position_gradient = sequence.grad[:, 0, :].abs().sum().item()

    print("\nBackpropagation through time (BPTT)")
    print("  LSTM parameters have gradients:", lstm_has_gradients)
    print("  GRU parameters have gradients:", gru_has_gradients)
    print("  first input position gradient magnitude:", first_position_gradient)
    print(
        "  A final-position loss reached an earlier input through recurrent "
        "states. Gating can mitigate vanishing gradients; it does not "
        "eliminate vanishing or exploding gradients."
    )


if __name__ == "__main__":
    main()
