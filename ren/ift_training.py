"""
Real implicit-function-theorem (IFT) DEQ training -- replaces the
disclosed truncated-unroll approximation in training.py.

This is the actual Bai, Kolter & Koltun (2019) backward pass: given
z* = Phi(z*; x), for any downstream loss L(z*),

    dL/dtheta = lambda^T (dPhi/dtheta)|_{z*}
    where lambda solves  (I - J^T) lambda = dL/dz*,  J = dPhi/dz|_{z*}

Rather than forming J explicitly, lambda is found by the iterative
(Neumann-series) fixed point  lambda_{k+1} = dL/dz* + J^T lambda_k,
computed via vector-Jacobian products (cheap, autograd-native). This
iteration converges at the SAME rate as the forward solve, by the
same spectral-radius-<1 argument (rho(J) < 1) -- i.e. REN's own
contraction property is what makes this backward pass well-defined,
not an extra assumption.

Bug this replaces: the previous truncated-unroll trainer backpropagated
through only N_UNROLL explicit applications of Phi, which is a
disclosed approximation, and its finite-difference check on real data
showed a large discrepancy on at least one parameter. This module's
correctness is checked against finite differences on the FULL solve-
to-convergence loss (not an unrolled approximation of it) -- the
correct test.
"""
from __future__ import annotations

import numpy as np
import torch

from ren.equilibrium_engine import RENOperator, Z_DIM, N_ASSETS, anderson_acceleration


def solve_equilibrium(op: RENOperator, z0: torch.Tensor, x: torch.Tensor,
                        max_iter=200, tol=1e-6) -> torch.Tensor:
    with torch.no_grad():
        res = anderson_acceleration(op, z0, x, max_iter=max_iter, tol=tol)
    return res.z_star, res.converged, res.n_iters


def pnl_loss(z_final: torch.Tensor, fwd_return: torch.Tensor, l2_weight=0.01) -> torch.Tensor:
    position = z_final[..., N_ASSETS:]
    pnl = (position * fwd_return).sum(dim=-1)
    gross = position.abs().sum(dim=-1)
    return -pnl.mean() + l2_weight * gross.mean()


def ift_backward_and_step(op: RENOperator, opt: torch.optim.Optimizer,
                            x_batch: torch.Tensor, fwd_batch: torch.Tensor,
                            z0: torch.Tensor, l2_weight=0.01,
                            solve_max_iter=200, solve_tol=1e-6,
                            adjoint_max_iter=40, adjoint_tol=1e-6,
                            grad_clip=1.0):
    """One real IFT-gradient training step. Returns (loss, z_star,
    n_solve_iters, n_adjoint_iters, adjoint_residual)."""
    # 1. Forward: solve for the equilibrium, no grad tracked (cheap, exact).
    z_star, converged, n_solve_iters = solve_equilibrium(op, z0, x_batch,
                                                            max_iter=solve_max_iter, tol=solve_tol)

    # 2. dL/dz* via a tiny leaf-tensor graph.
    z_leaf = z_star.clone().requires_grad_(True)
    loss = pnl_loss(z_leaf, fwd_batch, l2_weight=l2_weight)
    grad_z, = torch.autograd.grad(loss, z_leaf)

    # 3. Adjoint fixed point: lambda_{k+1} = grad_z + J^T lambda_k, via VJPs
    #    of f = Phi(z, x) at z = z* (held fixed), w.r.t. z.
    z_for_vjp = z_star.detach().requires_grad_(True)
    f = op(z_for_vjp, x_batch)  # graph over op.parameters() AND z_for_vjp

    lam = grad_z.clone()
    n_adj = 0
    resid = float("inf")
    for k in range(adjoint_max_iter):
        vjp_z, = torch.autograd.grad(f, z_for_vjp, grad_outputs=lam, retain_graph=True)
        lam_new = grad_z + vjp_z
        resid = torch.norm(lam_new - lam).item()
        lam = lam_new
        n_adj = k + 1
        if resid < adjoint_tol:
            break

    # 4. dL/dtheta = lambda^T dPhi/dtheta|_{z*} -- backprop lam through f.
    opt.zero_grad()
    torch.autograd.backward(f, grad_tensors=lam)
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(op.parameters(), grad_clip)
    opt.step()

    return dict(loss=loss.item(), z_star=z_star.detach(), converged=converged,
                 n_solve_iters=n_solve_iters, n_adjoint_iters=n_adj, adjoint_residual=resid)


def train_ren_operator_ift(X_train, fwd_ret_train, epochs=30, batch_size=32, lr=3e-4,
                              seed=42, damping=0.35, target_norm=0.6, l2_weight=0.02,
                              solve_max_iter=150, solve_tol=1e-6):
    torch.manual_seed(seed)
    op = RENOperator(seed=seed, damping=damping, target_norm=target_norm)
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    n = X_train.shape[0]
    history = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, fb = X_train[idx], fwd_ret_train[idx]
            z0 = torch.zeros(xb.shape[0], Z_DIM)
            out = ift_backward_and_step(op, opt, xb, fb, z0, l2_weight=l2_weight,
                                          solve_max_iter=solve_max_iter, solve_tol=solve_tol)
            epoch_losses.append(out["loss"])
        history.append(float(np.mean(epoch_losses)))
    return op, history


def finite_difference_grad_check_ift(op: RENOperator, x: torch.Tensor, fwd: torch.Tensor,
                                        l2_weight=0.02, eps=1e-4, n_params_check=6,
                                        solve_max_iter=200, solve_tol=1e-7):
    """Correct test: compares the IFT gradient to finite differences of
    the loss AS A FUNCTION OF THE FULL SOLVE-TO-CONVERGENCE fixed point
    (not an unrolled approximation) -- perturb a parameter, RE-SOLVE the
    equilibrium from scratch, recompute the loss."""
    xb, fb = x.unsqueeze(0), fwd.unsqueeze(0)
    z0 = torch.zeros(1, Z_DIM)

    def loss_at_current_params():
        z_star, _, _ = solve_equilibrium(op, z0, xb, max_iter=solve_max_iter, tol=solve_tol)
        return pnl_loss(z_star, fb, l2_weight=l2_weight).item()

    # Analytic gradient via one IFT step (without taking the optimizer step)
    opt = torch.optim.SGD(op.parameters(), lr=0.0)  # lr=0 => step() is a no-op, just used to zero_grad cleanly
    out = ift_backward_and_step(op, opt, xb, fb, z0, l2_weight=l2_weight,
                                  solve_max_iter=solve_max_iter, solve_tol=solve_tol)
    analytic_grads = [p.grad.clone() for p in op.parameters()]

    rel_errors = []
    params = list(op.parameters())
    checked = 0
    for pi, param in enumerate(params):
        if checked >= n_params_check:
            break
        flat = param.data.view(-1)
        j = 0  # first element of each tensor
        orig = flat[j].item()

        flat[j] = orig + eps
        loss_p = loss_at_current_params()
        flat[j] = orig - eps
        loss_m = loss_at_current_params()
        flat[j] = orig

        numeric = (loss_p - loss_m) / (2 * eps)
        analytic = analytic_grads[pi].view(-1)[j].item()
        rel_err = abs(analytic - numeric) / (abs(numeric) + 1e-8)
        rel_errors.append(dict(param_index=pi, shape=tuple(param.shape),
                                 analytic=analytic, numeric=numeric, rel_err=rel_err))
        checked += 1
    return rel_errors
