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


def test_passkey_generation():
    batch = generate_passkey_retrieval_batch(seq_len=128, batch_size=2, vocab_size=100)
    assert batch["input_ids"].shape == (2, 127)
    assert batch["target_val"].shape == (2,)
