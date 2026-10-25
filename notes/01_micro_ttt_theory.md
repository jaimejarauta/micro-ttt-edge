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
