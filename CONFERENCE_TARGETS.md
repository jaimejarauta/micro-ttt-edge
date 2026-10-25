# Micro-TTT Edge: Publication & Conference Strategy

**Author:** Jaime Jarauta  
**Primary Target:** **MLSys 2028** / **ICLR 2028** (Sub-quadratic sequence models & Edge AI Track)  
**Secondary Target:** **EMNLP 2027** / **IEEE Micro**  
**Strategic Workshop:** **ICML Hardware-Aware Efficient Training & Inference**

---

## 1. Target Venues & Submission Deadlines

| Venue | Track | Submission Window | Notification | Acceptance Rate |
| :--- | :--- | :---: | :---: | :---: |
| **MLSys 2028** | Systems for Machine Learning / Edge AI | Oct 2027 | Jan 2028 | ~22% |
| **ICLR 2028** | Efficient Architecture & Test-Time Compute | Sep 2027 | Jan 2028 | ~24% |
| **EMNLP 2027** | Efficient Methods for NLP | Jun 2027 | Aug 2027 | ~23% |
| **ICML Workshop** | Hardware-Aware Efficient Deep Learning | May 2027 | Jun 2027 | ~45% |

---

## 2. Reviewer Red Lines & Mandatory Epistemic Evidence

1. **Systems Skeptic:** Demands wall-clock proof on commodity hardware (ARM Cortex / Apple Silicon / Raspberry Pi). Must report exact FLOPs, memory bandwidth usage, and proof that inner state fits within L2 cache ($< 32\text{ KB}$).
2. **Associative Recall Hawkishness:** Demands passkey accuracy comparisons against Mamba-2, RWKV-6, and Full Multi-Head Attention at context lengths from $8\text{k}$ to $64\text{k}$ tokens.
3. **Quantization Stability:** Must demonstrate that INT8 simulated quantization of inner state matrix $W$ degrades retrieval accuracy by $< 2\%$.

---

## 3. Working Title & Hook

**Working Title:** *Micro-TTT: Single-Step Rank-1 Closed-Form Test-Time Memory in CPU L2 Cache*

**Abstract Hook:** While Test-Time Training (TTT) establishes neural network weights as recurrent hidden states, standard implementations rely on memory-bandwidth intensive autograd backward passes that preclude real-time execution on resource-constrained edge CPUs. We introduce Micro-TTT, reducing inner-loop test-time learning to an exact, closed-form rank-1 outer-product update on normalized key projections. Fitting entirely within 16 KB of L2 CPU cache, Micro-TTT sustains $> 180\text{ tokens/sec}$ streaming throughput on a single CPU core while preserving associative recall across long sequences.
