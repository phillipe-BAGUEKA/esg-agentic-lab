import pytest
import torch
from torch import nn

from prototypes.manual_lstm_cell import ManualLSTM, ManualLSTMCell


def test_lstm_parameter_shapes_and_count() -> None:
    cell = ManualLSTMCell(input_size=4, hidden_size=50)

    assert cell.weight_ih.shape == (200, 4)
    assert cell.weight_hh.shape == (200, 50)
    assert cell.bias_ih.shape == (200,)
    assert cell.bias_hh.shape == (200,)
    assert sum(parameter.numel() for parameter in cell.parameters()) == 11_200


def test_lstm_step_matches_equations_and_gate_ranges() -> None:
    torch.manual_seed(0)
    cell = ManualLSTMCell(input_size=2, hidden_size=3)
    x_t = torch.tensor([[0.2, -0.4], [0.1, 0.3]])
    h_previous = torch.tensor(
        [[0.5, -0.2, 0.1], [-0.1, 0.4, 0.2]]
    )
    c_previous = torch.tensor(
        [[0.3, 0.2, -0.5], [0.6, -0.2, 0.1]]
    )

    gates = (
        x_t @ cell.weight_ih.T
        + cell.bias_ih
        + h_previous @ cell.weight_hh.T
        + cell.bias_hh
    )
    i_linear, f_linear, g_linear, o_linear = gates.chunk(4, dim=1)
    expected_i = torch.sigmoid(i_linear)
    expected_f = torch.sigmoid(f_linear)
    expected_g = torch.tanh(g_linear)
    expected_o = torch.sigmoid(o_linear)
    expected_c = expected_f * c_previous + expected_i * expected_g
    expected_h = expected_o * torch.tanh(expected_c)

    step = cell.inspect_step(x_t, (h_previous, c_previous))

    torch.testing.assert_close(step["i_t"], expected_i)
    torch.testing.assert_close(step["f_t"], expected_f)
    torch.testing.assert_close(step["g_t"], expected_g)
    torch.testing.assert_close(step["o_t"], expected_o)
    torch.testing.assert_close(step["c_t"], expected_c)
    torch.testing.assert_close(step["h_t"], expected_h)
    assert step["h_t"].shape == (2, 3)
    assert step["c_t"].shape == (2, 3)
    assert torch.all((0 <= step["i_t"]) & (step["i_t"] <= 1))
    assert torch.all((0 <= step["f_t"]) & (step["f_t"] <= 1))
    assert torch.all((-1 <= step["g_t"]) & (step["g_t"] <= 1))
    assert torch.all((0 <= step["o_t"]) & (step["o_t"] <= 1))


def test_lstm_previous_cell_state_influences_next_cell_state() -> None:
    cell = ManualLSTMCell(input_size=2, hidden_size=2)
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.zero_()

    x_t = torch.zeros(1, 2)
    h_previous = torch.zeros(1, 2)
    c_previous = torch.tensor([[2.0, -2.0]])

    h_t, c_t = cell(x_t, (h_previous, c_previous))

    torch.testing.assert_close(c_t, 0.5 * c_previous)
    torch.testing.assert_close(h_t, 0.5 * torch.tanh(c_t))


def test_manual_lstm_unrolls_batch_first_sequence() -> None:
    torch.manual_seed(0)
    model = ManualLSTM(input_size=4, hidden_size=5).double()
    sequence = torch.randn(2, 6, 4, dtype=torch.float64)

    output, (h_n, c_n) = model(sequence)

    assert output.shape == (2, 6, 5)
    assert h_n.shape == (1, 2, 5)
    assert c_n.shape == (1, 2, 5)
    assert output.dtype == sequence.dtype
    torch.testing.assert_close(output[:, -1, :], h_n[0])


def test_lstm_backpropagates_to_early_input_and_all_parameters() -> None:
    torch.manual_seed(0)
    model = ManualLSTM(input_size=3, hidden_size=4)
    sequence = torch.randn(2, 5, 3, requires_grad=True)

    output, _ = model(sequence)
    output[:, -1, :].square().sum().backward()

    assert sequence.grad is not None
    assert sequence.grad[:, 0, :].abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_manual_lstm_cell_matches_pytorch_lstm_cell() -> None:
    torch.manual_seed(0)
    manual = ManualLSTMCell(input_size=3, hidden_size=4)
    reference = nn.LSTMCell(input_size=3, hidden_size=4)
    with torch.no_grad():
        reference.weight_ih.copy_(manual.weight_ih)
        reference.weight_hh.copy_(manual.weight_hh)
        reference.bias_ih.copy_(manual.bias_ih)
        reference.bias_hh.copy_(manual.bias_hh)

    x_t = torch.randn(2, 3)
    h_previous = torch.randn(2, 4)
    c_previous = torch.randn(2, 4)

    actual_h, actual_c = manual(x_t, (h_previous, c_previous))
    expected_h, expected_c = reference(x_t, (h_previous, c_previous))

    torch.testing.assert_close(actual_h, expected_h)
    torch.testing.assert_close(actual_c, expected_c)


def test_manual_lstm_matches_single_layer_pytorch_lstm() -> None:
    torch.manual_seed(0)
    manual = ManualLSTM(input_size=3, hidden_size=4)
    reference = nn.LSTM(
        input_size=3,
        hidden_size=4,
        batch_first=True,
    )
    with torch.no_grad():
        reference.weight_ih_l0.copy_(manual.cell.weight_ih)
        reference.weight_hh_l0.copy_(manual.cell.weight_hh)
        reference.bias_ih_l0.copy_(manual.cell.bias_ih)
        reference.bias_hh_l0.copy_(manual.cell.bias_hh)

    sequence = torch.randn(2, 5, 3)
    h_0 = torch.randn(1, 2, 4)
    c_0 = torch.randn(1, 2, 4)

    actual_output, (actual_h, actual_c) = manual(sequence, (h_0, c_0))
    expected_output, (expected_h, expected_c) = reference(
        sequence,
        (h_0, c_0),
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_h, expected_h)
    torch.testing.assert_close(actual_c, expected_c)


def test_lstm_rejects_invalid_shapes() -> None:
    cell = ManualLSTMCell(input_size=3, hidden_size=4)
    valid_x = torch.zeros(2, 3)
    valid_h = torch.zeros(2, 4)
    valid_c = torch.zeros(2, 4)

    with pytest.raises(ValueError, match="x_t must have shape"):
        cell(torch.zeros(2, 1, 3), (valid_h, valid_c))
    with pytest.raises(ValueError, match="input_size=3"):
        cell(torch.zeros(2, 2), (valid_h, valid_c))
    with pytest.raises(ValueError, match="h_previous must have shape"):
        cell(valid_x, (torch.zeros(1, 4), valid_c))
    with pytest.raises(ValueError, match="c_previous must have shape"):
        cell(valid_x, (valid_h, torch.zeros(2, 3)))

    model = ManualLSTM(input_size=3, hidden_size=4)
    with pytest.raises(ValueError, match="sequence must have shape"):
        model(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="input_size=3"):
        model(torch.zeros(2, 5, 2))
    with pytest.raises(ValueError, match="sequence_length"):
        model(torch.zeros(2, 0, 3))
    with pytest.raises(ValueError, match="h_0 must have shape"):
        model(
            torch.zeros(2, 5, 3),
            (torch.zeros(1, 1, 4), torch.zeros(1, 2, 4)),
        )
    with pytest.raises(ValueError, match="batch_first=True"):
        ManualLSTM(input_size=3, hidden_size=4, batch_first=False)
