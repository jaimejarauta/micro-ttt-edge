"""Synthetic Long-Context Benchmark Tasks for Micro-TTT."""

import random
from typing import Dict, List, Tuple
import numpy as np
import torch


def generate_passkey_retrieval_batch(
    seq_len: int = 4096,
    num_passkeys: int = 1,
    vocab_size: int = 256,
    batch_size: int = 4,
    seed: int = 2026,
) -> Dict[str, torch.Tensor]:
    """Generates synthetic passkey retrieval sequences.
    
    Structure:
        [Garbage tokens...] ... [Key_i, Value_i] ... [Garbage tokens...] ... [Query_i] -> Target: Value_i
    """
    rng = random.Random(seed)
    # Reserve special tokens
    KEY_START = 1
    KEY_END = max(2, vocab_size // 4)
    VAL_START = KEY_END + 1
    VAL_END = max(VAL_START + 1, vocab_size // 2)
    NOISE_START = VAL_END + 1
    NOISE_END = max(NOISE_START + 1, vocab_size - 1)

    input_ids_list = []
    target_ids_list = []
    query_pos_list = []

    for b in range(batch_size):
        # Fill sequence with random noise tokens
        seq = [rng.randint(NOISE_START, NOISE_END) for _ in range(seq_len)]
        
        # Insert passkeys at random positions in the first 75% of the sequence
        passkeys = {}
        insert_positions = sorted(rng.sample(range(50, int(seq_len * 0.75)), num_passkeys))
        
        for pos in insert_positions:
            k = rng.randint(KEY_START, KEY_END)
            v = rng.randint(VAL_START, VAL_END)
            passkeys[k] = v
            seq[pos] = k
            seq[pos + 1] = v

        # Place query at the end
        query_key = rng.choice(list(passkeys.keys()))
        target_val = passkeys[query_key]
        
        # Last token asks for query_key
        seq[-2] = query_key
        seq[-1] = target_val

        input_ids_list.append(seq[:-1])
        target_ids_list.append(target_val)
        query_pos_list.append(seq_len - 2)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "target_val": torch.tensor(target_ids_list, dtype=torch.long),
        "seq_len": seq_len,
    }


def generate_mqar_batch(
    seq_len: int = 2048,
    num_pairs: int = 16,
    vocab_size: int = 256,
    batch_size: int = 4,
    seed: int = 2026,
) -> Dict[str, torch.Tensor]:
    """Multi-Query Associative Recall benchmark sequence."""
    rng = random.Random(seed)
    input_ids_list = []
    labels_list = []

    for b in range(batch_size):
        seq = [rng.randint(100, vocab_size - 1) for _ in range(seq_len)]
        labels = [-100] * seq_len
        
        pairs = {}
        for p in range(num_pairs):
            k = rng.randint(10, 50)
            v = rng.randint(51, 99)
            pairs[k] = v
            pos = rng.randint(10, seq_len - 200)
            seq[pos] = k
            seq[pos + 1] = v

        # Insert multiple queries in the final 200 tokens
        query_keys = list(pairs.keys())
        rng.shuffle(query_keys)
        q_start = seq_len - len(query_keys) * 2
        for idx, qk in enumerate(query_keys):
            curr_pos = q_start + idx * 2
            seq[curr_pos] = qk
            seq[curr_pos + 1] = pairs[qk]
            labels[curr_pos + 1] = pairs[qk]

        input_ids_list.append(seq[:-1])
        labels_list.append(labels[1:])

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }
