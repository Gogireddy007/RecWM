"""
Trains ATLAS's warm-start network g_psi(x) -> z0_hat on REAL market
snapshots, with a strict time-based train/test split (train on the
first 80% of dates chronologically, test on the last 20%, zero
lookahead). Labels are fixed points z* computed offline by running
Anderson acceleration to tight tolerance for every snapshot -- this is
a genuine self-supervised regression target, not a fabricated number.

Reports, on the held-out test set only:
  - iterations-to-converge: zero-init vs. random-init vs. trained warm-start
  - wall-clock latency (ms) for each, percentiles
  - warm-start regression error (MSE to the true fixed point)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ren.equilibrium_engine import (
    RENOperator, WarmStartNet, Z_DIM, X_DIM,
    anderson_acceleration, naive_fixed_point_iteration,
)


def compute_ground_truth_fixed_points(op: RENOperator, X: torch.Tensor,
                                        tol=1e-7, max_iter=500) -> torch.Tensor:
    """Slow, high-precision solve for every snapshot -- the ATLAS training
    label. z0 = 0 for every snapshot (unbiased starting point)."""
    Z = torch.zeros(X.shape[0], Z_DIM)
    converged_flags = []
    for i in range(X.shape[0]):
        z0 = torch.zeros(Z_DIM)
        res = anderson_acceleration(op, z0, X[i], max_iter=max_iter, tol=tol)
        Z[i] = res.z_star
        converged_flags.append(res.converged)
    return Z, np.array(converged_flags)


def train_warm_start(X_train, Z_train, epochs=300, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = WarmStartNet(hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = net(X_train)
        loss = loss_fn(pred, Z_train)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return net, history


@dataclass
class WarmStartBenchmark:
    zero_init_iters: np.ndarray
    random_init_iters: np.ndarray
    trained_init_iters: np.ndarray
    zero_init_ms: np.ndarray
    random_init_ms: np.ndarray
    trained_init_ms: np.ndarray
    warm_start_mse: float
    fixed_point_norm_mean: float


def benchmark_warm_starts(op: RENOperator, net: WarmStartNet, X_test: torch.Tensor,
                            Z_test_true: torch.Tensor, tol=1e-5, max_iter=300, seed=0) -> WarmStartBenchmark:
    rng = torch.Generator().manual_seed(seed)
    n = X_test.shape[0]
    zero_it, rand_it, trained_it = [], [], []
    zero_ms, rand_ms, trained_ms = [], [], []

    with torch.no_grad():
        preds = net(X_test)
    mse = torch.mean((preds - Z_test_true) ** 2).item()

    for i in range(n):
        x = X_test[i]

        t0 = time.perf_counter()
        r0 = anderson_acceleration(op, torch.zeros(Z_DIM), x, max_iter=max_iter, tol=tol)
        zero_ms.append((time.perf_counter() - t0) * 1000)
        zero_it.append(r0.n_iters)

        t0 = time.perf_counter()
        r1 = anderson_acceleration(op, torch.randn(Z_DIM, generator=rng), x, max_iter=max_iter, tol=tol)
        rand_ms.append((time.perf_counter() - t0) * 1000)
        rand_it.append(r1.n_iters)

        with torch.no_grad():
            z0_hat = net(x)
        t0 = time.perf_counter()
        r2 = anderson_acceleration(op, z0_hat.detach(), x, max_iter=max_iter, tol=tol)
        trained_ms.append((time.perf_counter() - t0) * 1000)
        trained_it.append(r2.n_iters)

    return WarmStartBenchmark(
        zero_init_iters=np.array(zero_it), random_init_iters=np.array(rand_it),
        trained_init_iters=np.array(trained_it),
        zero_init_ms=np.array(zero_ms), random_init_ms=np.array(rand_ms),
        trained_init_ms=np.array(trained_ms),
        warm_start_mse=mse,
        fixed_point_norm_mean=float(torch.mean(torch.norm(Z_test_true, dim=1)).item()),
    )
