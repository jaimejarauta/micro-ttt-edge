"""Evaluation metrics and hardware efficiency benchmarks for Micro-TTT."""

import time
from typing import Dict
import torch


def compute_passkey_accuracy(logits: torch.Tensor, target_val: torch.Tensor) -> float:
    """Computes top-1 accuracy for the queried passkey token.
    
    Args:
        logits: Model predictions of shape (B, T, vocab_size)
        target_val: Ground truth target token (B,)
    """
    last_logits = logits[:, -1, :]  # (B, vocab_size)
    preds = torch.argmax(last_logits, dim=-1)
    correct = (preds == target_val).float().sum().item()
    return float(correct / len(target_val))


def benchmark_cpu_throughput(
    model: torch.nn.Module,
    seq_len: int = 4096,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Measures single-core/CPU streaming inference latency and memory footprint."""
    model.eval()
    x = torch.randint(10, 200, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        _ = model(x[:, :128])

    # Benchmarking
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    elapsed = time.perf_counter() - t0

    tokens = seq_len * batch_size
    tokens_per_sec = tokens / elapsed
    ms_per_token = (elapsed / tokens) * 1000.0

    # L2 Cache footprint: state size = num_layers * d_k * d_v * 4 bytes
    num_layers = len(model.layers) if hasattr(model, "layers") else 2
    state_bytes = num_layers * 32 * 32 * 4  # e.g., 2 * 1024 * 4 = 8 KB
    state_kb = state_bytes / 1024.0

    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token,
        "elapsed_sec": elapsed,
        "seq_len": seq_len,
        "state_kb": state_kb,
    }
