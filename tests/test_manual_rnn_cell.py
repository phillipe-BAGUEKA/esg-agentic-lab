import torch

from prototypes.manual_rnn_cell import ManualRNNCell, run_manual_rnn


def test_parameter_shapes_and_trainable_parameter_count() -> None:
    input_size = 3
    hidden_size = 4
    cell = ManualRNNCell(input_size=input_size, hidden_size=hidden_size)

    assert cell.weight_xh.shape == (hidden_size, input_size)
    assert cell.weight_hh.shape == (hidden_size, hidden_size)
    assert cell.bias_h.shape == (hidden_size,)

    parameter_count = sum(
        parameter.numel()
        for parameter in cell.parameters()
        if parameter.requires_grad
    )
    assert parameter_count == (
        hidden_size * input_size
        + hidden_size * hidden_size
        + hidden_size
    )


def test_cell_output_shape() -> None:
    cell = ManualRNNCell(input_size=3, hidden_size=4)
    x_t = torch.randn(2, 3)
    h_previous = torch.randn(2, 4)

    h_t = cell(x_t, h_previous)

    assert h_t.shape == (2, 4)


def test_cell_matches_manual_recurrent_equation() -> None:
    torch.manual_seed(0)
    cell = ManualRNNCell(input_size=2, hidden_size=3)
    x_t = torch.randn(2, 2)
    h_previous = torch.randn(2, 3)

    expected = torch.tanh(
        x_t @ cell.weight_xh.T
        + h_previous @ cell.weight_hh.T
        + cell.bias_h
    )

    assert torch.allclose(cell(x_t, h_previous), expected)


def test_sequence_output_and_final_hidden_shapes() -> None:
    cell = ManualRNNCell(input_size=3, hidden_size=4)
    sequence = torch.randn(2, 5, 3)

    output, h_n = run_manual_rnn(cell, sequence)

    assert output.shape == (2, 5, 4)
    assert h_n.shape == (2, 4)
    assert torch.equal(output[:, -1, :], h_n)


def test_previous_hidden_state_influences_output() -> None:
    cell = ManualRNNCell(input_size=2, hidden_size=2)
    with torch.no_grad():
        cell.weight_xh.zero_()
        cell.weight_hh.copy_(torch.eye(2))
        cell.bias_h.zero_()

    x_t = torch.zeros(1, 2)
    without_context = cell(x_t, torch.zeros(1, 2))
    with_context = cell(x_t, torch.ones(1, 2))

    assert not torch.equal(without_context, with_context)


def test_gradients_exist_after_unrolling_multiple_positions() -> None:
    torch.manual_seed(0)
    cell = ManualRNNCell(input_size=3, hidden_size=4)
    sequence = torch.randn(2, 5, 3)

    output, _ = run_manual_rnn(cell, sequence)
    loss = output.square().sum()
    loss.backward()

    assert all(parameter.grad is not None for parameter in cell.parameters())
