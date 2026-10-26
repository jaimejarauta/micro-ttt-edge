#!/usr/bin/env python3
"""Smoke test runner for Micro-TTT Edge."""

import argparse
import json
import os
import sys
import time
import yaml
import torch

from micro_ttt.model import MicroTTTModel
from micro_ttt.tasks import generate_passkey_retrieval_batch
from micro_ttt.metrics import compute_passkey_accuracy, benchmark_cpu_throughput


def main():
    parser = argparse.ArgumentParser(description="Micro-TTT Smoke Test")
    parser.add_argument("--config", type=str, default="configs/smoke_test.yaml")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="moonshot-micro-ttt-edge", help="W&B project")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run display name")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    use_wandb = args.wandb or os.environ.get("WANDB_MODE") in ["online", "offline"]
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            entity = os.environ.get("WANDB_ENTITY", "research-projekts")
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=entity,
                name=args.wandb_name or f"micro-ttt-smoke-{int(time.time())}",
                mode=args.wandb_mode if not args.wandb else "online",
                config={"seq_len": args.seq_len or cfg["task"]["seq_len"], "config": cfg},
            )
            print(f"[W&B] Initialized run '{wandb_run.name}' in project '{args.wandb_project}'")
        except Exception as e:
            print(f"[W&B Warning] Init skipped ({e}). Continuing locally.")
            use_wandb = False

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 65)
    print(f"MICRO-TTT: SINGLE-STEP RANK-1 TEST-TIME MEMORY SMOKE TEST")
    print(f"Device: {device}")
    print("=" * 65)

    seq_len = args.seq_len or cfg["task"]["seq_len"]
    batch_size = cfg["task"]["batch_size"]
    vocab_size = cfg["task"]["vocab_size"]

    print(f"\n[1/3] Instantiating MicroTTTModel (d_model={cfg['model']['d_model']}, d_k={cfg['model']['d_k']}, d_v={cfg['model']['d_v']})...")
    model = MicroTTTModel(
        vocab_size=vocab_size,
        d_model=cfg["model"]["d_model"],
        d_k=cfg["model"]["d_k"],
        d_v=cfg["model"]["d_v"],
        d_ff=cfg["model"]["d_ff"],
        num_layers=cfg["model"]["num_layers"],
        lr=cfg["model"]["lr"],
        decay=cfg["model"]["decay"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    state_kb = len(model.layers) * cfg["model"]["d_k"] * cfg["model"]["d_v"] * 4 / 1024.0
    print(f"Parameters: {param_count:,} | Inner State Cache Footprint: {state_kb:.1f} KB")

    print(f"\n[2/3] Generating synthetic passkey sequence (T={seq_len:,} tokens)...")
    batch = generate_passkey_retrieval_batch(
        seq_len=seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        seed=cfg["task"]["seed"],
    )
    input_ids = batch["input_ids"].to(device)
    target_val = batch["target_val"].to(device)

    print("\n[3/3] Running streaming inference benchmark...")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_ids)
    elapsed = time.perf_counter() - t0

    tokens = seq_len * batch_size
    tok_per_sec = tokens / elapsed
    ms_per_tok = (elapsed / tokens) * 1000.0

    target_tok_s = cfg["pre_registered_rules"]["throughput_threshold_tok_s"]
    max_state_kb = cfg["pre_registered_rules"]["max_state_kb"]
    gate_passed = tok_per_sec >= target_tok_s and state_kb <= max_state_kb

    print("\n" + "=" * 65)
    print("PRE-REGISTERED GATE ASSESSMENT:")
    print(f"  Streaming Throughput : {tok_per_sec:.1f} tok/s  (Target: >= {target_tok_s:.1f} tok/s)")
    print(f"  Per-Token Latency    : {ms_per_tok:.3f} ms/token (Target: <= {cfg['pre_registered_rules']['max_latency_ms_per_token']:.1f} ms)")
    print(f"  Inner State Footprint: {state_kb:.1f} KB       (Target: <= {max_state_kb:.1f} KB)")
    print(f"  Gate Status          : {'PASSED' if gate_passed else 'FAILED'}")
    print("=" * 65)

    os.makedirs("results", exist_ok=True)
    manifest = {
        "project": "micro-ttt-edge",
        "device": str(device),
        "seq_len": seq_len,
        "throughput_tok_s": tok_per_sec,
        "ms_per_token": ms_per_tok,
        "state_kb": state_kb,
        "gate_passed": gate_passed,
        "timestamp": time.time(),
    }
    with open("results/smoke_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved results to results/smoke_manifest.json")

    if use_wandb and wandb_run:
        wandb.summary["throughput_tok_s"] = tok_per_sec
        wandb.summary["ms_per_token"] = ms_per_tok
        wandb.summary["state_kb"] = state_kb
        wandb.summary["gate_passed"] = gate_passed
        wandb.summary["seq_len"] = seq_len
        wandb.log({
            "throughput_tok_s": tok_per_sec,
            "ms_per_token": ms_per_tok,
            "state_kb": state_kb,
            "gate_passed": int(gate_passed),
        })
        wandb.finish()
        print("[W&B] Run successfully finished and synced.")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
