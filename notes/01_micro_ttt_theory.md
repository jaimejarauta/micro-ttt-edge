# Note 01: Micro-TTT Mathematical Formulation & Cache Locality

**Author:** Jaime Jarauta  
**Status:** Active theory note  
**Target:** `src/micro_ttt/model.py`

---

## 1. Mathematical Derivation

Standard Test-Time Training (Sun et al., 2024) defines an inner-loop loss:
$$\mathcal{L}(W) = \frac{1}{2} \|f(x_t; W) - \theta(x_t)\|_2^2$$
and updates $W$ via gradient descent:
$$W_t = W_{t-1} - \eta \nabla_W \mathcal{L}(W_{t-1})$$

When $f(x_t; W) = W x_t$ (linear test-time memory) with normalized key $k_t = \frac{x_t}{\|x_t\|_2}$:
$$\nabla_W \mathcal{L} = (W_{t-1} k_t - \theta(k_t)) k_t^\top = e_t k_t^\top$$

The update becomes an exact rank-1 matrix outer product:
$$W_t = \lambda W_{t-1} - \eta \cdot e_t k_t^\top$$

## 2. Hardware Bounds on CPU L2 Cache

For dimension $d_k = d_v = 32$:
- Outer product requires: $32 \times 32 = 1,024$ fused multiply-accumulates (FMAs).
- Memory footprint per head: $32 \times 32 \times 4\text{ bytes} = 4,096\text{ bytes} = 4\text{ KB}$.
- For a 2-layer, 2-head edge model: $4 \times 4\text{ KB} = 16\text{ KB}$.
- Typical L2 cache per CPU core is $512\text{ KB} - 1\text{ MB}$.
- Thus, the entire state remains **100% resident in L2 cache**, achieving memory latency $< 5\text{ ns}$ per access.

## Run log 2026-10-26: first trained runs (passkey)

Train at T=256, evaluate held-out at T in {256, 1024, 4096}, 64 sequences each (chance 1/64). 1500 steps x 32,
d_model 64, 2 layers, seed 0, CPU. Rule: >= 80% at every length.

| Arm | T=256 | T=1024 | T=4096 |
|---|---|---|---|
| Fixed decay 0.999, fixed write rate | 100% | 81.2% | 9.4% |
| Data-dependent gates | 71.9% | 75.0% | 71.9% |

Both FAIL. The fixed-decay collapse at 4096 matches the retention bound (~0.997^n). The gated arm is flat in T,
which is the property the 32k claim needs, but is undertrained (final loss 1.03 vs 0.001). Next: longer gated run.
Manifest: results/passkey_train_manifest.json.

Gated arm, 4000 steps, lr 2e-3 (results/passkey_gated_4k_manifest.json):
T=256 78.1% | T=1024 82.8% | T=4096 68.8% | T=16384 75.0% (64 sequences each; +-~11 pts at 95%).
FAIL (>= 80% at every length). Accuracy is flat in T up to 64x the training length, so the gates remove the decay
limit; the remaining error is optimization (loss plateaus ~0.9, noisy). Next: train at T=1024 with a dense
multi-key objective (Colab run 5b) rather than one supervised token per sequence.
