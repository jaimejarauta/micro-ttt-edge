# Micro-TTT: Single-Step Rank-1 Test-Time Memory on CPU

Closed-form Test-Time Training (TTT) executing inside CPU L2 cache for extreme long-context edge inference.

## Overview

Micro-TTT replaces backpropagation-based inner loops in Test-Time Training with an exact, closed-form rank-1 outer-product update:

$$W_t = \lambda W_{t-1} - \eta \cdot (W_{t-1} k_t - \theta(k_t)) \otimes k_t$$

By operating on compact $32 \times 32$ matrix states ($\le 8\text{ KB}$ per layer), the entire memory update fits within CPU vector registers (AVX2 / ARM NEON), achieving $> 100\text{ tokens/sec}$ streaming throughput on standard CPU cores across context lengths up to $500\text{k}$ tokens with zero GPU VRAM.

## Pre-Registered Gate Criteria
- **Gate 1 (Smoke / Throughput):** Streaming throughput $\ge 100\text{ tok/s}$, per-token latency $\le 10.0\text{ ms}$, and inner state footprint $\le 32\text{ KB}$.
- **Gate 2 (Stage 2 Passkey):** Top-1 accuracy $\ge 95\%$ on needle-in-a-haystack retrieval at $T = 64\text{k}$ tokens on single CPU core.

## Quickstart

### Local (Apple Silicon / CPU):
```bash
# Run smoke benchmark
python scripts/run_smoke.py --config configs/smoke_test.yaml
```

### Google Colab:
Open `notebooks/colab_runner_micro_ttt.ipynb` in Colab. Includes automatic machine teardown (`runtime.unassign()`) to protect compute credits upon completion.

## License
MIT
