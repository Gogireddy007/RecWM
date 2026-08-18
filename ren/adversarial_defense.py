"""
Invention 9 -- Adversarial Defense.

Two sub-components, each tested separately with real, measured numbers.

1. Crowding radar: measures REAL pairwise correlation among the five
   agent-type signals (belief_fields.py) over the real historical
   panel, flags high-correlation ("crowded trade") days, and tests --
   honestly, on real data -- whether elevated crowding actually
   precedes worse forward risk-adjusted outcomes for a naive consensus
   strategy. This is a real, falsifiable empirical claim, checked
   directly.

2. Execution camouflage: a genuine simulation (no real market data
   exists for order-level execution of a non-existent fund, so this is
   explicitly a synthetic-schedule study, not a real-data backtest).
   Compares TWAP-style equal-slice scheduling against a randomized-
   size, randomized-time slicing scheme, and measures each schedule's
   *detectability* via the strength of the dominant periodogram peak
   in the child-order-size time series -- a standard, real signal-
   processing technique for spotting algorithmic execution signatures.
   Lower peak power = harder to detect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ren.signals_history import HistoryCache, AGENT_SERIES_FNS


def crowding_series(hist: HistoryCache, timescale="medium", window=21) -> pd.Series:
    """Mean pairwise correlation across the 5 agent types' cross-asset
    mean signal, computed on a rolling real window."""
    mats = []
    for theta in AGENT_SERIES_FNS:
        mats.append(hist.series[(theta, timescale)].mean(axis=1))
    df = pd.concat(mats, axis=1)
    df.columns = list(AGENT_SERIES_FNS.keys())

    n_types = len(AGENT_SERIES_FNS)
    mask = ~np.eye(n_types, dtype=bool)
    roll_corr = df.rolling(window).corr()  # MultiIndex (date, type) x type

    def mean_offdiag(date):
        c = roll_corr.loc[date].to_numpy()
        vals = c[mask]
        return np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan

    result = pd.Series({d: mean_offdiag(d) for d in df.index}, name="crowding")
    result.index.name = df.index.name
    return result


@dataclass
class CrowdingBacktestResult:
    crowding: pd.Series
    forward_return: pd.Series
    quantile_returns: pd.DataFrame
    spearman_corr: float
    spearman_pvalue: float


def crowding_predicts_bad_outcomes(hist: HistoryCache, panel: pd.DataFrame, tickers: list[str],
                                     timescale="medium", window=21, horizon=21) -> CrowdingBacktestResult:
    crowd = crowding_series(hist, timescale, window)
    close = panel[[(t, "Close") for t in tickers]]; close.columns = tickers
    basket_fwd_ret = close.mean(axis=1).pct_change(horizon).shift(-horizon)
    basket_fwd_vol = close.mean(axis=1).pct_change().rolling(horizon).std().shift(-horizon) * np.sqrt(252)

    joint = pd.concat([crowd.rename("crowding"), basket_fwd_ret.rename("fwd_ret"),
                        basket_fwd_vol.rename("fwd_vol")], axis=1).dropna()
    joint["fwd_sharpe_like"] = joint["fwd_ret"] / (joint["fwd_vol"] + 1e-9)

    joint["quantile"] = pd.qcut(joint["crowding"], 5, labels=False, duplicates="drop")
    qret = joint.groupby("quantile")[["fwd_ret", "fwd_vol", "fwd_sharpe_like"]].mean()

    from scipy import stats
    rho, pval = stats.spearmanr(joint["crowding"], joint["fwd_sharpe_like"])

    return CrowdingBacktestResult(crowding=crowd, forward_return=basket_fwd_ret,
                                    quantile_returns=qret, spearman_corr=float(rho),
                                    spearman_pvalue=float(pval))


# ---------------------------------------------------------------------------
# Execution camouflage: synthetic schedule detectability study
# ---------------------------------------------------------------------------

def twap_schedule(total_qty: float, n_slices: int, n_bins: int = 480) -> np.ndarray:
    """TWAP: equal-size clips at perfectly regular time bins -- a clean
    impulse train, over `n_bins` fine time steps spanning the horizon."""
    x = np.zeros(n_bins)
    positions = np.linspace(0, n_bins, n_slices, endpoint=False).astype(int)
    x[positions] = total_qty / n_slices
    return x


def randomized_schedule(total_qty: float, n_slices: int, seed=0, dispersion=0.6,
                          n_bins: int = 480, time_jitter_frac: float = 0.5) -> np.ndarray:
    """Randomized-size, randomized-time clips: sizes drawn from a
    Dirichlet (so they still sum to total_qty), positions jittered
    around the TWAP grid points by up to `time_jitter_frac` of the
    inter-clip spacing."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_bins)
    base_positions = np.linspace(0, n_bins, n_slices, endpoint=False)
    spacing = n_bins / n_slices
    jitter = rng.uniform(-time_jitter_frac, time_jitter_frac, n_slices) * spacing
    positions = np.clip((base_positions + jitter).astype(int), 0, n_bins - 1)
    weights = rng.dirichlet(np.full(n_slices, 1.0 / dispersion))
    for p, w in zip(positions, weights):
        x[p] += w * total_qty
    return x


def detectability_score(schedule: np.ndarray) -> float:
    """Strength of the dominant periodogram peak, normalized by total
    power -- a real, standard measure of how 'clock-like' (and hence
    detectable) a trade schedule's execution pattern is, over the fine-
    grained impulse-train representation (captures both timing and
    size regularity, not size variance alone)."""
    x = schedule - schedule.mean()
    if np.allclose(x, 0):
        return 0.0
    freqs, power = _periodogram(x)
    total_power = power.sum() + 1e-12
    return float(power.max() / total_power)


def _periodogram(x: np.ndarray):
    n = len(x)
    fft = np.fft.rfft(x)
    power = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n)
    return freqs[1:], power[1:]  # drop DC term


def run_camouflage_study(total_qty=1.0, n_slices=48, n_trials=200, dispersion=0.6) -> pd.DataFrame:
    twap = twap_schedule(total_qty, n_slices)
    twap_score = detectability_score(twap)

    rand_scores = [detectability_score(randomized_schedule(total_qty, n_slices, seed=i, dispersion=dispersion))
                   for i in range(n_trials)]

    return pd.DataFrame({
        "schedule": ["TWAP"] + [f"randomized_{i}" for i in range(n_trials)],
        "detectability": [twap_score] + rand_scores,
    })
