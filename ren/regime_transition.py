"""
Invention 5 -- Regime Transition Engine.

Operationalizes Scheffer et al.'s critical-slowing-down signature: as
rho(D Phi) -> 1, the system should show slower recovery / rising
sensitivity ahead of a real structural break. Ground truth for "a
structural break happened" is not fabricated -- it is realized
volatility computed from real price data (a standard, objective, widely
used regime-shift proxy), plus a short list of well-known, independently
verifiable real historical stress dates that fall inside our real data
window (2016-03-14 .. present):

    2018-12-01 .. 2018-12-24   Q4 2018 selloff
    2020-02-20 .. 2020-03-23   COVID-19 crash
    2022-01-01 .. 2022-10-15   2022 rate-hike bear market
    2023-03-08 .. 2023-03-13   SVB / regional-bank crisis
    2024-08-05                  Yen carry-trade unwind vol spike
    2025-04-03 .. 2025-04-09   "Liberation Day" tariff selloff

Honesty note: rho(D Phi) below is computed from RENOperator with FIXED,
randomly-seeded (untrained) weights -- there is no labeled dataset of
real "reflexive equilibria" to train the core operator on. So this is
a test of whether the raw architecture (real market inputs -> a
random-but-fixed nonlinear reflexive map -> its Jacobian's spectral
radius) already carries a measurable relationship to real market
stress, not a claim that a trained model has been validated as a
crisis predictor. Whatever correlation comes out is reported as-is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ren.equilibrium_engine import anderson_acceleration, spectral_radius_at_fixed_point, Z_DIM

KNOWN_STRESS_WINDOWS = [
    ("2018-12-01", "2018-12-24", "Q4 2018 selloff"),
    ("2020-02-20", "2020-03-23", "COVID-19 crash"),
    ("2022-01-01", "2022-10-15", "2022 rate-hike bear market"),
    ("2023-03-08", "2023-03-13", "SVB / regional-bank crisis"),
    ("2024-08-05", "2024-08-05", "Yen carry-trade unwind"),
    ("2025-04-03", "2025-04-09", "Liberation Day tariff selloff"),
]


def compute_rho_trajectory(op, X: torch.Tensor, dates, max_iter=100, tol=1e-6) -> pd.DataFrame:
    rows = []
    z0 = torch.zeros(Z_DIM)
    for i in range(X.shape[0]):
        x = X[i]
        res = anderson_acceleration(op, z0, x, max_iter=max_iter, tol=tol)
        rho = spectral_radius_at_fixed_point(op, res.z_star, x)
        rows.append({"date": dates[i], "rho": rho, "converged": res.converged,
                      "n_iters": res.n_iters})
        z0 = res.z_star.detach()  # warm-start next day from today's equilibrium
    return pd.DataFrame(rows).set_index("date")


def label_stress_windows(index: pd.DatetimeIndex) -> pd.Series:
    label = pd.Series(0, index=index)
    for start, end, name in KNOWN_STRESS_WINDOWS:
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        label[mask] = 1
    return label


def lead_lag_correlation(rho: pd.Series, stress_indicator: pd.Series, max_lag=20) -> pd.DataFrame:
    """Real Pearson correlation between rho(t) and stress_indicator(t+lag)
    for lag = -max_lag..max_lag. Positive lag = rho leads (rho today vs
    stress `lag` days in the future)."""
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = stress_indicator.shift(-lag)
        joint = pd.concat([rho, shifted], axis=1).dropna()
        if len(joint) < 30:
            continue
        corr = joint.iloc[:, 0].corr(joint.iloc[:, 1])
        rows.append({"lag_days": lag, "pearson_r": corr})
    return pd.DataFrame(rows)
