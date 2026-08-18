"""
Invention 1 -- Influence Kernel.

Thesis definition (Section 3.3):
    K*(s) = (I - Gamma * A(s))^-1 A(s)
the resolvent that sums the entire infinite series of indirect
influence effects I + Gamma*A + (Gamma*A)^2 + ... in one matrix
inversion. The spectral radius of Gamma*A determines whether that
series (and the closed form) even converges.

What is real here vs. what is a modeling choice
-------------------------------------------------
A(s), the one-hop direct-influence matrix between agent types at
timescale s, is estimated the standard, defensible way: a ridge-
regularized VAR(1) fit of each agent type's real, data-derived signal
(from belief_fields.py) on every agent type's signal one step earlier,
using the REAL historical signal time series -- not invented numbers.
Gamma is a single global coupling-strength scalar; because the thesis
requires rho(Gamma*A) < 1 for the resolvent to be well-defined (the
Neumann series must converge), Gamma is set automatically as a fixed
fraction of 1/rho(A) (a MonDEQ-style monotone-operator discipline),
which is a modeling choice made explicit here, not tuned to produce a
particular result.

Everything after that -- the matrix inversion, the resolvent-vs-
Neumann-series equivalence check, the spectral radius, the influence
rankings -- are exact, measured linear-algebra facts about real,
estimated matrices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class InfluenceKernelResult:
    A: np.ndarray            # direct one-hop influence matrix
    gamma: float              # coupling scalar
    K_star: np.ndarray        # resolvent (I - gamma A)^-1 A
    spectral_radius_A: float
    spectral_radius_gammaA: float
    labels: list[str]


def fit_direct_influence(signal_history: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """VAR(1) fit: X_t ~ A X_{t-1}, ridge-regularized least squares.

    signal_history: shape (T, n_types) real time series of agent-type
    signals (already computed from real market data upstream).
    Returns A of shape (n_types, n_types), A[i, j] = direct effect of
    type j at t-1 on type i at t.
    """
    X_tm1 = signal_history[:-1]  # (T-1, n)
    X_t = signal_history[1:]     # (T-1, n)
    n = X_tm1.shape[1]
    XtX = X_tm1.T @ X_tm1 + ridge * np.eye(n)
    XtY = X_tm1.T @ X_t
    # Solve (X'X) A' = X'Y  =>  A = (X'Y)' (X'X)^-1, A[i,j] effect of j on i
    A = (np.linalg.solve(XtX, XtY)).T
    return A


def compute_influence_kernel(
    signal_history: np.ndarray,
    labels: list[str],
    gamma_margin: float = 0.9,
    ridge: float = 1e-2,
) -> InfluenceKernelResult:
    A = fit_direct_influence(signal_history, ridge=ridge)
    rho_A = float(np.max(np.abs(np.linalg.eigvals(A))))
    gamma = gamma_margin / rho_A if rho_A > 1e-9 else gamma_margin
    n = A.shape[0]
    resolvent_mat = np.linalg.inv(np.eye(n) - gamma * A)
    K_star = resolvent_mat @ A
    rho_gA = float(np.max(np.abs(np.linalg.eigvals(gamma * A))))
    return InfluenceKernelResult(
        A=A, gamma=gamma, K_star=K_star,
        spectral_radius_A=rho_A, spectral_radius_gammaA=rho_gA,
        labels=labels,
    )


def type_mean_signal_matrix(history_cache, timescale="medium") -> tuple:
    """(T, n_agent_types) real time series: each agent type's mean-across-
    assets raw signal at `timescale`, over the full real history cached
    in `history_cache`. Used to fit the direct-influence matrix A(s)."""
    from ren.belief_fields import AGENT_TYPES
    cols = []
    labels = list(AGENT_TYPES.keys())
    for theta in labels:
        s = history_cache.series[(theta, timescale)].mean(axis=1)
        cols.append(s)
    mat = pd.concat(cols, axis=1)
    mat.columns = labels
    return mat.to_numpy(), labels, mat.index


def neumann_series_partial_sum(A: np.ndarray, gamma: float, n_terms: int) -> np.ndarray:
    """Brute-force truncated sum: sum_{k=0}^{n_terms-1} (gamma A)^k A.
    Used ONLY to verify the resolvent closed form is correct -- this is
    the direct numerical check of the claim in Section 3.3."""
    n = A.shape[0]
    term = A.copy()
    total = np.zeros_like(A)
    gA = gamma * A
    power = np.eye(n)
    for _ in range(n_terms):
        total += power @ term
        power = power @ gA
    return total


def verify_resolvent_convergence(result: InfluenceKernelResult, max_terms: int = 60) -> pd.DataFrame:
    """Measures ||partial_sum(k) - K*||_F for k = 1..max_terms.
    This is a REAL, measured convergence curve, not a claim."""
    rows = []
    for k in range(1, max_terms + 1):
        partial = neumann_series_partial_sum(result.A, result.gamma, k)
        err = np.linalg.norm(partial - result.K_star, ord="fro")
        rows.append({"n_terms": k, "frobenius_error": err})
    return pd.DataFrame(rows)
