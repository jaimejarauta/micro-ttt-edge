#!/usr/bin/env python3
"""Train Micro-TTT on passkey retrieval at a short length, evaluate at longer lengths.
Arms: fixed scalar decay/write rate vs data-dependent gates, matched size, data and steps.
"""

import argparse
import json
import os
import subprocess
import time

import torch
import torch.nn.functional as F

from micro_ttt.model import MicroTTTModel
from micro_ttt.tasks import generate_passkey_retrieval_batch
from micro_ttt.metrics import compute_passkey_accuracy


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def evaluate(model, lengths, n, seed, device):
    model.eval()
    out = {}
    with torch.no_grad():
        for T in lengths:
            b = generate_passkey_retrieval_batch(seq_len=T, batch_size=n, vocab_size=256, seed=seed + T)
            out[T] = compute_passkey_accuracy(model(b["input_ids"].to(device)), b["target_val"].to(device))
    model.train()
    return out


def train_arm(gated: bool, args, device) -> dict:
    torch.manual_seed(args.seed)
    model = MicroTTTModel(vocab_size=256, d_model=args.d_model, num_layers=args.num_layers, gated=gated).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    for step in range(args.steps):
        b = generate_passkey_retrieval_batch(seq_len=args.train_len, batch_size=args.batch_size, vocab_size=256,
                                             seed=args.seed * 1_000_000 + step)
        logits = model(b["input_ids"].to(device))
        loss = F.cross_entropy(logits[:, -1], b["target_val"].to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"  [gated={gated}] step {step + 1}/{args.steps} loss {loss.item():.3f}")
    acc = evaluate(model, args.eval_lengths, args.eval_samples, seed=10**9, device=device)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"micro_ttt_gated{int(gated)}.pt"))
    return {"gated": gated, "accuracy": acc, "final_loss": loss.item(), "train_sec": time.time() - t0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-len", type=int, default=256)
    p.add_argument("--eval-lengths", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", type=str, nargs="+", default=["fixed", "gated"], choices=["fixed", "gated"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--output", type=str, default="results/passkey_train_manifest.json")
    args = p.parse_args()
    device = torch.device(args.device)

    # Gate (2026-10-26): an arm passes if held-out accuracy >= 0.80 at every eval length (chance = 1/64)
    arms = [train_arm(name == "gated", args, device) for name in args.arms]
    for a in arms:
        a["gate_passed"] = all(v >= 0.80 for v in a["accuracy"].values())
        accs = " | ".join(f"T={T}: {v * 100:.1f}%" for T, v in a["accuracy"].items())
        print(f"gated={a['gated']!s:<5} {accs} -> {'PASS' if a['gate_passed'] else 'FAIL'}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"git_commit": git_hash(), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "config": vars(args), "arms": arms}, f, indent=2)
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
