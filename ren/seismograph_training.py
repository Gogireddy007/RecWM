"""
The exact experiment proposed in review: train the operator so that
rho(D Phi) is DIRECTLY supervised toward real forward market stress,
then test on real held-out data whether the predicted rho -> 1 (near
crisis) relationship is recoverable, or whether the wrong-direction
result found on the untrained operator is a deeper structural property
that persists even under direct supervision.

Objective: at each real training-period date, compute the TRUE
spectral radius of D(Phi) at the real fixed point (via
torch.autograd.functional.jacobian(create_graph=True) -> eigvals --
small 32x32 problem, cheap enough to do exactly rather than
approximate), and regress it toward a real, continuous forward-stress
target: SPY's real forward 21-day realized volatility, z-scored
against its own trailing history, squashed into rho's achievable range
via a sigmoid.

This directly tests the fork identified in review:
  - if TEST-period rho now correlates POSITIVELY with real stress ->
    the Triple Correspondence needs learning, not just structure
  - if it still doesn't (or still goes the wrong way) -> the negative
    result is structural, not a training deficiency
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ren.equilibrium_engine import RENOperator, Z_DIM, anderson_acceleration


def compute_real_stress_targets(panel: pd.DataFrame, dates, vol_window=21) -> pd.Series:
    spy_close = panel[("SPY", "Close")]
    ret = spy_close.pct_change()
    fwd_vol = ret.rolling(vol_window).std().shift(-vol_window)
    z = (fwd_vol - fwd_vol.expanding().mean()) / (fwd_vol.expanding().std() + 1e-9)
    target_rho = 0.5 + 0.45 * torch.sigmoid(torch.tensor(z.reindex(dates).fillna(0.0).to_numpy(),
                                                            dtype=torch.float32))
    return pd.Series(target_rho.numpy(), index=dates)


def true_spectral_radius_differentiable(op: RENOperator, z_star: torch.Tensor, x: torch.Tensor):
    z = z_star.clone().detach().requires_grad_(True)

    def f(zz):
        return op(zz, x)

    J = torch.autograd.functional.jacobian(f, z, create_graph=True, vectorize=True)
    eigvals = torch.linalg.eigvals(J)
    rho = torch.max(torch.abs(eigvals))
    return rho


def train_seismograph_supervised(X_train: torch.Tensor, target_rho_train: torch.Tensor,
                                    epochs=10, samples_per_epoch=200, lr=1e-3,
                                    seed=42, damping=0.35, target_norm=0.6,
                                    solve_max_iter=100, solve_tol=1e-6, seed_data=0):
    torch.manual_seed(seed)
    op = RENOperator(seed=seed, damping=damping, target_norm=target_norm)
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    n = X_train.shape[0]
    rng = np.random.default_rng(seed_data)
    history = []
    for ep in range(epochs):
        idxs = rng.choice(n, size=min(samples_per_epoch, n), replace=False)
        epoch_losses = []
        for i in idxs:
            x = X_train[i]
            target = target_rho_train[i]
            with torch.no_grad():
                res = anderson_acceleration(op, torch.zeros(Z_DIM), x, max_iter=solve_max_iter, tol=solve_tol)
            rho = true_spectral_radius_differentiable(op, res.z_star, x)
            loss = (rho - target) ** 2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))
    return op, history
