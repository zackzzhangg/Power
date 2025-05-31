# -*- coding: utf-8 -*-
import numpy as np

def moving_block_bootstrap(X, y, block_len, n_samples, rng):
    """对 (X, y) 做 MBB，返回 n_samples 份 (Xb, yb)。"""
    T = len(X)
    k = int(np.ceil(T / block_len))
    blocks_X = np.array([X[i : i + block_len] for i in range(T - block_len + 1)])
    blocks_y = np.array([y[i : i + block_len] for i in range(T - block_len + 1)])

    samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(blocks_X), size=k)
        Xb = blocks_X[idx].reshape(-1, *X.shape[1:])[:T]
        yb = blocks_y[idx].reshape(-1)[:T]
        samples.append((Xb, yb))
    return samples
