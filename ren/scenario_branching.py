"""
Invention 6 -- Scenario Branching Engine.

Perturbs the belief field directly and re-solves the equilibrium under
the hypothetical injection, rather than resampling historical windows.
Every perturbation and re-solve below runs the REAL Equilibrium Engine
(Invention 3) on a REAL historical snapshot as the base case; only the
belief perturbation itself is hypothetical, exactly as Section 5's
description requires.

Ranking: plausibility = 1 / (1 + ||perturbation||) (smaller shocks are
more plausible); portfolio impact = || z*_perturbed - z*_base || in
the position sub-block of z*, i.e. how much the equilibrium trade
changes. Both are computed directly, not asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ren.equilibrium_engine import anderson_acceleration, Z_DIM, N_ASSETS


@dataclass
class Scenario:
    name: str
    perturbation: np.ndarray     # length 16, applied to agg_belief block of x
    z_star: torch.Tensor
    plausibility: float
    portfolio_impact: float
    n_iters: int
    converged: bool


def make_named_perturbations(tickers: list[str], sigma=2.0) -> dict:
    n = len(tickers)
    perts = {}
    rng = np.random.default_rng(0)
    perts["broad_risk_off"] = -sigma * np.ones(n)
    perts["broad_risk_on"] = sigma * np.ones(n)
    single_asset = np.zeros(n)
    single_asset[tickers.index("SPY")] = -3 * sigma if "SPY" in tickers else 0
    perts["spy_shock"] = single_asset
    sector_rotation = rng.normal(0, 1, n) * sigma
    perts["random_sector_rotation"] = sector_rotation
    if all(t in tickers for t in ["TLT", "SPY"]):
        rates_shock = np.zeros(n)
        rates_shock[tickers.index("TLT")] = -3 * sigma
        rates_shock[tickers.index("SPY")] = -1.5 * sigma
        perts["rate_shock"] = rates_shock
    return perts


def run_scenario_branching(op, x_base: torch.Tensor, tickers: list[str],
                             z0: torch.Tensor = None, sigma=2.0, max_iter=200, tol=1e-6) -> list[Scenario]:
    z0 = z0 if z0 is not None else torch.zeros(Z_DIM)
    base_res = anderson_acceleration(op, z0, x_base, max_iter=max_iter, tol=tol)
    z_base = base_res.z_star

    perturbations = make_named_perturbations(tickers, sigma=sigma)
    scenarios = []
    for name, pert in perturbations.items():
        x_pert = x_base.clone()
        x_pert[:N_ASSETS] = x_pert[:N_ASSETS] + torch.tensor(pert, dtype=torch.float32)
        res = anderson_acceleration(op, z_base, x_pert, max_iter=max_iter, tol=tol)
        impact = float(torch.norm(res.z_star[N_ASSETS:] - z_base[N_ASSETS:]).item())
        plausibility = float(1.0 / (1.0 + np.linalg.norm(pert)))
        scenarios.append(Scenario(name=name, perturbation=pert, z_star=res.z_star,
                                    plausibility=plausibility, portfolio_impact=impact,
                                    n_iters=res.n_iters, converged=res.converged))
    scenarios.sort(key=lambda s: -s.portfolio_impact)
    return scenarios
