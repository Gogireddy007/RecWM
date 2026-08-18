"""
Composite-loss IFT trainer, per the review's Week 1-2 plan: raw P&L is a
bad training objective (it collapsed to a flat book in v2/v3). This
replaces it with a windowed, sequential objective computed over
chronological real-data windows (Sharpe/drawdown/turnover are
sequential concepts and are meaningless on i.i.d.-shuffled days, which
is what the earlier P&L trainer used):

    loss = -sharpe_weight * batch_Sharpe(window)
           + drawdown_weight * |max_drawdown(window)|
           + turnover_weight * mean_turnover(window)
           + concentration_weight * mean_position_concentration(window)
           + l2_weight * mean_gross_exposure(window)

Gradients are the real IFT adjoint (ren/ift_training.py's method),
extended here to a whole window: each day's equilibrium is solved
walk-forward (previous day's z* warm-starts the next, exactly matching
how the live backtest itself walks forward), then ONE backward pass
over the window-level loss gives dL/dz*_t for every day t in the
window simultaneously (they're coupled through Sharpe/drawdown), and
each day's adjoint fixed point + parameter-gradient accumulation is
run exactly as in the single-day IFT method, summed across the window.

The SEISMOGRAPH auxiliary (ren/seismograph_training.py's rho-regression
loss) is NOT fused into this same backward pass -- it uses a different
differentiation path (Jacobian eigenvalues, not the P&L adjoint) and
fusing them cleanly is more machinery than this scope needs. Instead
it is alternated: every `seismograph_every` windows, a separate
seismograph-supervision step is taken on a few random days, sharing
the same operator and optimizer -- ordinary multi-task alternation.
"""
from __future__ import annotations

import numpy as np
import torch

from ren.equilibrium_engine import RENOperator, Z_DIM, N_ASSETS, anderson_acceleration
from ren.seismograph_training import compute_real_stress_targets, true_spectral_radius_differentiable


def window_composite_loss(z_leaves, fwd_window, sharpe_weight=1.0, drawdown_weight=0.15,
                            turnover_weight=0.08, concentration_weight=0.05, l2_weight=0.01):
    positions = torch.stack([zl[N_ASSETS:] for zl in z_leaves])  # (W, N_ASSETS)
    pnl = (positions * fwd_window).sum(dim=-1)  # (W,)
    sharpe = pnl.mean() / (pnl.std() + 1e-6)

    cum = torch.cumsum(pnl, dim=0)
    running_max = torch.cummax(cum, dim=0).values
    drawdown = (cum - running_max).min()  # <= 0, more negative = worse

    if positions.shape[0] > 1:
        turnover = (positions[1:] - positions[:-1]).abs().sum(dim=-1).mean()
    else:
        turnover = torch.tensor(0.0)

    gross = positions.abs().sum(dim=-1).mean()
    weights_abs = positions.abs()
    concentration = (weights_abs.max(dim=-1).values / (weights_abs.sum(dim=-1) + 1e-9)).mean()

    loss = (-sharpe_weight * sharpe
            + drawdown_weight * (-drawdown)
            + turnover_weight * turnover
            + concentration_weight * concentration
            + l2_weight * gross)

    metrics = dict(sharpe=sharpe.item(), drawdown=drawdown.item(), turnover=turnover.item(),
                    concentration=concentration.item(), gross=gross.item())
    return loss, metrics


def ift_windowed_step(op, opt, X_window, fwd_window, z0, loss_kwargs=None,
                        solve_max_iter=100, solve_tol=1e-6, adjoint_max_iter=40, adjoint_tol=1e-6,
                        grad_clip=1.0):
    loss_kwargs = loss_kwargs or {}
    W = X_window.shape[0]

    z = z0
    z_stars = []
    with torch.no_grad():
        for t in range(W):
            res = anderson_acceleration(op, z, X_window[t], max_iter=solve_max_iter, tol=solve_tol)
            z = res.z_star
            z_stars.append(z)

    z_leaves = [zt.clone().requires_grad_(True) for zt in z_stars]
    loss, metrics = window_composite_loss(z_leaves, fwd_window, **loss_kwargs)
    grad_zs = torch.autograd.grad(loss, z_leaves)

    opt.zero_grad()
    for t in range(W):
        z_for_vjp = z_stars[t].detach().requires_grad_(True)
        f = op(z_for_vjp, X_window[t])
        lam = grad_zs[t].clone()
        for _ in range(adjoint_max_iter):
            vjp_z, = torch.autograd.grad(f, z_for_vjp, grad_outputs=lam, retain_graph=True)
            lam_new = grad_zs[t] + vjp_z
            resid = torch.norm(lam_new - lam).item()
            lam = lam_new
            if resid < adjoint_tol:
                break
        torch.autograd.backward(f, grad_tensors=lam)

    torch.nn.utils.clip_grad_norm_(op.parameters(), grad_clip)
    opt.step()
    return dict(loss=loss.item(), z_final=z_stars[-1].detach(), **metrics)


def seismograph_aux_step(op, opt, X_train, target_rho_train, n_days=3, seed_data=None,
                            solve_max_iter=80, solve_tol=1e-6):
    n = X_train.shape[0]
    rng = np.random.default_rng(seed_data)
    idxs = rng.choice(n, size=n_days, replace=False)
    losses = []
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
        losses.append(loss.item())
    return float(np.mean(losses))


def train_composite(X_train, fwd_train, panel, dates_train, window_len=21, n_windows_per_epoch=40,
                      epochs=15, lr=5e-4, seed=42, damping=0.35, target_norm=0.6,
                      loss_kwargs=None, seismograph_every=5, seismograph_weight_days=3,
                      solve_max_iter=100, solve_tol=1e-6):
    torch.manual_seed(seed)
    op = RENOperator(seed=seed, damping=damping, target_norm=target_norm)
    opt = torch.optim.Adam(op.parameters(), lr=lr)

    target_rho_train = torch.tensor(
        compute_real_stress_targets(panel, dates_train).to_numpy(), dtype=torch.float32)

    n = X_train.shape[0]
    rng = np.random.default_rng(seed)
    history = []
    window_count = 0
    for ep in range(epochs):
        ep_losses, ep_sharpes = [], []
        for _ in range(n_windows_per_epoch):
            start = rng.integers(0, n - window_len)
            X_w = X_train[start:start + window_len]
            fwd_w = fwd_train[start:start + window_len]
            z0 = torch.zeros(Z_DIM)
            out = ift_windowed_step(op, opt, X_w, fwd_w, z0, loss_kwargs=loss_kwargs,
                                      solve_max_iter=solve_max_iter, solve_tol=solve_tol)
            ep_losses.append(out["loss"])
            ep_sharpes.append(out["sharpe"])
            window_count += 1
            if window_count % seismograph_every == 0:
                seismograph_aux_step(op, opt, X_train, target_rho_train,
                                       n_days=seismograph_weight_days, seed_data=window_count)
        history.append({"epoch": ep, "loss": float(np.mean(ep_losses)),
                          "window_sharpe": float(np.mean(ep_sharpes))})
    return op, history
