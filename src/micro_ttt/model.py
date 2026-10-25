"""Micro-TTT Architecture: Single-step rank-1 closed-form test-time training memory."""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroTTTCell(nn.Module):
    """Rank-1 closed-form outer-product Test-Time Training (TTT) cell.
    
    Maintains an inner weight matrix W in R^{d_k x d_v} updated at test-time
    via an exact closed-form gradient descent step on a self-supervised reconstruction task:
        L_test = 0.5 * ||W x_t - theta(x_t)||^2
        grad_W = (W x_t - theta(x_t)) x_t^T = e_t x_t^T
        W_t = lambda * W_{t-1} - eta * (e_t x_t^T)
    """

    def __init__(
        self,
        d_model: int,
        d_k: int = 32,
        d_v: int = 32,
        lr: float = 0.05,
        decay: float = 0.999,
        quantize_int8: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.lr = lr
        self.decay = decay
        self.quantize_int8 = quantize_int8

        # Projections to inner key/value/query subspaces
        self.proj_q = nn.Linear(d_model, d_k, bias=False)
        self.proj_k = nn.Linear(d_model, d_k, bias=False)
        self.proj_v = nn.Linear(d_model, d_v, bias=False)

        # Self-supervised target mapping theta(x_t)
        self.target_proj = nn.Linear(d_k, d_v, bias=False)

        # Output projection back to d_model
        self.out_proj = nn.Linear(d_v, d_model, bias=False)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize inner weight matrix W_0 in R^{B x d_k x d_v}."""
        return torch.zeros(batch_size, self.d_k, self.d_v, device=device, dtype=torch.float32)

    def step(
        self,
        x_t: torch.Tensor,
        w_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Single test-time training step.
        
        Args:
            x_t: Current token representation (B, d_model)
            w_prev: Previous inner memory state (B, d_k, d_v)
        Returns:
            y_t: Output token representation (B, d_model)
            w_next: Updated inner memory state (B, d_k, d_v)
            telemetry: Dictionary with error norm and update statistics
        """
        q_t = self.proj_q(x_t)  # (B, d_k)
        k_t = self.proj_k(x_t)  # (B, d_k)
        v_t = self.proj_v(x_t)  # (B, d_v)

        # Normalize key for numerical stability
        k_norm = F.normalize(k_t, p=2, dim=-1)

        # Inner prediction: pred = k_t @ W_{t-1}
        # (B, 1, d_k) @ (B, d_k, d_v) -> (B, 1, d_v) -> (B, d_v)
        pred = torch.bmm(k_norm.unsqueeze(1), w_prev).squeeze(1)

        # Target signal: combination of external value and self-supervised target
        target = v_t + self.target_proj(k_norm)
        error = pred - target  # e_t in R^{B x d_v}

        # Closed-form rank-1 update: delta_W = e_t^T k_norm
        # (B, d_k, 1) @ (B, 1, d_v) -> (B, d_k, d_v)
        delta_w = torch.bmm(k_norm.unsqueeze(2), error.unsqueeze(1))

        # Weight update with decay
        w_next = self.decay * w_prev - self.lr * delta_w

        # Optional INT8 simulated quantization for extreme cache compression
        if self.quantize_int8:
            scale = torch.max(torch.abs(w_next), dim=-1, keepdim=True)[0].clamp(min=1e-5) / 127.0
            w_next = torch.round(w_next / scale).clamp(-128, 127) * scale

        # Readout with query token: y_inner = q_t @ W_t
        q_norm = F.normalize(q_t, p=2, dim=-1)
        y_inner = torch.bmm(q_norm.unsqueeze(1), w_next).squeeze(1)  # (B, d_v)
        y_t = self.out_proj(y_inner)

        telemetry = {
            "error_norm": float(torch.norm(error, p=2, dim=-1).mean().item()),
            "w_norm": float(torch.norm(w_next, p=2, dim=(-2, -1)).mean().item()),
        }

        return y_t, w_next, telemetry

    def forward(
        self,
        x: torch.Tensor,
        w_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-level forward pass over T tokens.
        
        Args:
            x: Input sequence of shape (B, T, d_model)
            w_init: Optional initial state (B, d_k, d_v)
        Returns:
            outputs: Processed sequence (B, T, d_model)
            w_final: Final memory state (B, d_k, d_v)
        """
        b, t, d = x.shape
        device = x.device
        w_state = w_init if w_init is not None else self.init_state(b, device)

        y_steps = []
        for s in range(t):
            x_s = x[:, s, :]
            y_s, w_state, _ = self.step(x_s, w_state)
            y_steps.append(y_s)

        outputs = torch.stack(y_steps, dim=1)
        return outputs, w_state


class MicroTTTBlock(nn.Module):
    """Full Micro-TTT transformer-alternative layer with pre-norm and MLP."""

    def __init__(
        self,
        d_model: int = 64,
        d_k: int = 32,
        d_v: int = 32,
        d_ff: int = 128,
        lr: float = 0.05,
        decay: float = 0.999,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ttt = MicroTTTCell(d_model=d_model, d_k=d_k, d_v=d_v, lr=lr, decay=decay)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        w_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w_final = self.ttt(self.norm1(x), w_init=w_init)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, w_final


class MicroTTTModel(nn.Module):
    """Micro-TTT Language / Sequence Model for long-context recall."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        d_k: int = 32,
        d_v: int = 32,
        d_ff: int = 128,
        num_layers: int = 2,
        lr: float = 0.05,
        decay: float = 0.999,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MicroTTTBlock(d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff, lr=lr, decay=decay)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Maps (B, T) input token indices to logits (B, T, vocab_size)."""
        x = self.embed(input_ids)
        for layer in self.layers:
            x, _ = layer(x)
        return self.head(x)
