"""Unit tests for Micro-TTT Edge."""

import torch
from micro_ttt.model import MicroTTTCell, MicroTTTBlock, MicroTTTModel
from micro_ttt.tasks import generate_passkey_retrieval_batch


def test_micro_ttt_cell_step():
    cell = MicroTTTCell(d_model=32, d_k=16, d_v=16)
    x = torch.randn(2, 32)
    w = cell.init_state(2, torch.device("cpu"))
    y, w_next, telemetry = cell.step(x, w)
    assert y.shape == (2, 32)
    assert w_next.shape == (2, 16, 16)
    assert "error_norm" in telemetry


def test_micro_ttt_model_forward():
    model = MicroTTTModel(vocab_size=100, d_model=32, d_k=16, d_v=16, num_layers=1)
    input_ids = torch.randint(0, 100, (2, 64))
    logits = model(input_ids)
    assert logits.shape == (2, 64, 100)


def test_micro_ttt_backward():
    """Verify autograd backward pass works cleanly without in-place modification errors."""
    model = MicroTTTModel(vocab_size=100, d_model=32, d_k=16, d_v=16, num_layers=2)
    input_ids = torch.randint(0, 100, (2, 32))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_passkey_generation():
    batch = generate_passkey_retrieval_batch(seq_len=128, batch_size=2, vocab_size=100)
    assert batch["input_ids"].shape == (2, 127)
    assert batch["target_val"].shape == (2,)


def test_gated_cell_retains_memory_when_write_gate_closed():
    from micro_ttt.model import MicroTTTCell
    torch.manual_seed(0)
    cell = MicroTTTCell(d_model=16, d_k=8, d_v=8, gated=True)
    with torch.no_grad():
        cell.proj_gate.weight.zero_()
        cell.proj_gate.bias.copy_(torch.tensor([-30.0, 30.0]))  # write ~0, keep ~1
    w0 = torch.randn(2, 8, 8)
    _, w = cell(torch.randn(2, 4096, 16), w_init=w0)
    assert torch.allclose(w, w0, atol=1e-4)


def test_ungated_cell_forgets_over_long_context():
    from micro_ttt.model import MicroTTTCell
    torch.manual_seed(0)
    cell = MicroTTTCell(d_model=16, d_k=8, d_v=8)
    w0 = torch.randn(2, 8, 8)
    with torch.no_grad():
        _, w = cell(torch.zeros(2, 4096, 16), w_init=w0)
    assert w.norm() < 0.05 * w0.norm()  # fixed decay 0.999^4096 ~ 0.017


def test_gated_step_matches_forward():
    from micro_ttt.model import MicroTTTCell
    torch.manual_seed(0)
    cell = MicroTTTCell(d_model=16, d_k=8, d_v=8, gated=True)
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        _, w_seq = cell(x)
        w = cell.init_state(2, torch.device("cpu"))
        for t in range(5):
            _, w, _ = cell.step(x[:, t], w, compute_telemetry=False)
    assert torch.allclose(w, w_seq, atol=1e-5)
