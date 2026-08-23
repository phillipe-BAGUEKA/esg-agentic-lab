import pytest
import torch
from torch import nn

from prototypes.manual_gru_cell import ManualGRU, ManualGRUCell


def test_gru_parameter_shapes_and_count() -> None:
    cell = ManualGRUCell(input_size=4, hidden_size=50)

    assert cell.weight_ih.shape == (150, 4)
    assert cell.weight_hh.shape == (150, 50)
    assert cell.bias_ih.shape == (150,)
    assert cell.bias_hh.shape == (150,)
    assert sum(parameter.numel() for parameter in cell.parameters()) == 8_400


def test_gru_step_matches_equations_gate_order_and_contributions() -> None:
    torch.manual_seed(0)
    cell = ManualGRUCell(input_size=2, hidden_size=3)
    x_t = torch.tensor([[0.2, -0.4], [0.1, 0.3]])
    h_previous = torch.tensor(
        [[0.5, -0.2, 0.1], [-0.1, 0.4, 0.2]]
    )

    input_r, input_z, input_n = (
        x_t @ cell.weight_ih.T + cell.bias_ih
    ).chunk(3, dim=1)
    hidden_r, hidden_z, hidden_n = (
        h_previous @ cell.weight_hh.T + cell.bias_hh
    ).chunk(3, dim=1)
    expected_r = torch.sigmoid(input_r + hidden_r)
    expected_z = torch.sigmoid(input_z + hidden_z)
    expected_n = torch.tanh(input_n + expected_r * hidden_n)
    expected_old = expected_z * h_previous
    expected_new = (1 - expected_z) * expected_n
    expected_h = expected_old + expected_new

    step = cell.inspect_step(x_t, h_previous)

    torch.testing.assert_close(step["r_t"], expected_r)
    torch.testing.assert_close(step["z_t"], expected_z)
    torch.testing.assert_close(step["n_t"], expected_n)
    torch.testing.assert_close(step["old_contribution"], expected_old)
    torch.testing.assert_close(step["new_contribution"], expected_new)
    torch.testing.assert_close(step["h_t"], expected_h)
    assert step["h_t"].shape == (2, 3)
    assert torch.all((0 <= step["r_t"]) & (step["r_t"] <= 1))
    assert torch.all((0 <= step["z_t"]) & (step["z_t"] <= 1))
    assert torch.all((-1 <= step["n_t"]) & (step["n_t"] <= 1))


def test_gru_previous_hidden_state_influences_next_hidden_state() -> None:
    cell = ManualGRUCell(input_size=2, hidden_size=2)
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.zero_()

    x_t = torch.zeros(1, 2)
    h_previous = torch.tensor([[2.0, -2.0]])

    h_t = cell(x_t, h_previous)

    torch.testing.assert_close(h_t, 0.5 * h_previous)


def test_manual_gru_unrolls_batch_first_sequence() -> None:
    torch.manual_seed(0)
    model = ManualGRU(input_size=4, hidden_size=5).double()
    sequence = torch.randn(2, 6, 4, dtype=torch.float64)

    output, h_n = model(sequence)

    assert output.shape == (2, 6, 5)
    assert h_n.shape == (1, 2, 5)
    assert output.dtype == sequence.dtype
    torch.testing.assert_close(output[:, -1, :], h_n[0])


def test_gru_backpropagates_to_early_input_and_all_parameters() -> None:
    torch.manual_seed(0)
    model = ManualGRU(input_size=3, hidden_size=4)
    sequence = torch.randn(2, 5, 3, requires_grad=True)

    output, _ = model(sequence)
    output[:, -1, :].square().sum().backward()

    assert sequence.grad is not None
    assert sequence.grad[:, 0, :].abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_manual_gru_cell_matches_pytorch_gru_cell() -> None:
    torch.manual_seed(0)
    manual = ManualGRUCell(input_size=3, hidden_size=4)
    reference = nn.GRUCell(input_size=3, hidden_size=4)
    with torch.no_grad():
        reference.weight_ih.copy_(manual.weight_ih)
        reference.weight_hh.copy_(manual.weight_hh)
        reference.bias_ih.copy_(manual.bias_ih)
        reference.bias_hh.copy_(manual.bias_hh)

    x_t = torch.randn(2, 3)
    h_previous = torch.randn(2, 4)

    actual = manual(x_t, h_previous)
    expected = reference(x_t, h_previous)

    torch.testing.assert_close(actual, expected)


def test_manual_gru_matches_single_layer_pytorch_gru() -> None:
    torch.manual_seed(0)
    manual = ManualGRU(input_size=3, hidden_size=4)
    reference = nn.GRU(
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

    actual_output, actual_h = manual(sequence, h_0)
    expected_output, expected_h = reference(sequence, h_0)

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_h, expected_h)


def test_gru_rejects_invalid_shapes() -> None:
    cell = ManualGRUCell(input_size=3, hidden_size=4)
    valid_x = torch.zeros(2, 3)
    valid_h = torch.zeros(2, 4)

    with pytest.raises(ValueError, match="x_t must have shape"):
        cell(torch.zeros(2, 1, 3), valid_h)
    with pytest.raises(ValueError, match="input_size=3"):
        cell(torch.zeros(2, 2), valid_h)
    with pytest.raises(ValueError, match="h_previous must have shape"):
        cell(valid_x, torch.zeros(1, 4))
    with pytest.raises(ValueError, match="h_previous must have shape"):
        cell(valid_x, torch.zeros(2, 3))

    model = ManualGRU(input_size=3, hidden_size=4)
    with pytest.raises(ValueError, match="sequence must have shape"):
        model(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="input_size=3"):
        model(torch.zeros(2, 5, 2))
    with pytest.raises(ValueError, match="sequence_length"):
        model(torch.zeros(2, 0, 3))
    with pytest.raises(ValueError, match="h_0 must have shape"):
        model(torch.zeros(2, 5, 3), torch.zeros(1, 1, 4))
    with pytest.raises(ValueError, match="batch_first=True"):
        ManualGRU(input_size=3, hidden_size=4, batch_first=False)
