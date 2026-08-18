"""
Pressure tests: is the headline Sharpe number for each model a robust
estimate, or a fragile point estimate that a different resample / a
different market regime would overturn?

1. Block bootstrap confidence intervals on annualized Sharpe -- resamples
   contiguous BLOCKS of daily P&L (not i.i.d. days, which would destroy
   the real autocorrelation structure in daily returns) with replacement,
   recomputes Sharpe each time, and reports the empirical 90% interval.
   Standard technique (Politis & Romano, 1994, stationary/moving block
   bootstrap) for exactly this kind of financial time-series inference
   problem.

2. Regime-conditional breakdown -- splits the real test period into
   terciles by real, contemporaneous market volatility (SPY realized
   vol) and reports each model's Sharpe within each regime, to check
   whether a model's edge (or lack of one) is uniform across real market
   conditions or concentrated in one regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_sharpe(daily_pnl: np.ndarray, n_boot=2000, block_size=21, seed=0) -> dict:
    daily_pnl = np.asarray(daily_pnl)
    n = len(daily_pnl)
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)

    boot_sharpes = np.zeros(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([daily_pnl[s:s + block_size] for s in starts])[:n]
        vol = sample.std()
        boot_sharpes[b] = (sample.mean() * 252) / (vol * np.sqrt(252)) if vol > 1e-12 else 0.0

    return {
        "point_estimate": float((daily_pnl.mean() * 252) / (daily_pnl.std() * np.sqrt(252) + 1e-12)),
        "bootstrap_mean": float(boot_sharpes.mean()),
        "ci_5": float(np.percentile(boot_sharpes, 5)),
        "ci_95": float(np.percentile(boot_sharpes, 95)),
        "prob_sharpe_gt_0": float((boot_sharpes > 0).mean()),
        "n_boot": n_boot, "block_size": block_size,
    }


def regime_breakdown(daily_pnl: np.ndarray, dates, panel: pd.DataFrame, n_regimes=3,
                        vol_window=21) -> pd.DataFrame:
    spy_close = panel[("SPY", "Close")]
    spy_vol = spy_close.pct_change().rolling(vol_window).std().reindex(dates)

    df = pd.DataFrame({"pnl": np.asarray(daily_pnl), "vol": spy_vol.to_numpy()}, index=dates)
    df = df.dropna()
    df["regime"] = pd.qcut(df["vol"], n_regimes, labels=["calm", "normal", "turbulent"])

    rows = []
    for regime in ["calm", "normal", "turbulent"]:
        sub = df.loc[df["regime"] == regime, "pnl"]
        vol = sub.std()
        sharpe = (sub.mean() * 252) / (vol * np.sqrt(252)) if vol > 1e-12 else 0.0
        rows.append({"regime": regime, "n_days": len(sub), "ann_return": float(sub.mean() * 252),
                      "sharpe": float(sharpe)})
    return pd.DataFrame(rows)


def calendar_subperiod_breakdown(daily_pnl: np.ndarray, dates) -> pd.DataFrame:
    df = pd.DataFrame({"pnl": np.asarray(daily_pnl)}, index=pd.DatetimeIndex(dates))
    df["period"] = df.index.year.astype(str) + ("H1", "H2")[0]  # placeholder, replaced below
    df["period"] = [f"{d.year}H{1 if d.month <= 6 else 2}" for d in df.index]
    rows = []
    for period, sub in df.groupby("period"):
        vol = sub["pnl"].std()
        sharpe = (sub["pnl"].mean() * 252) / (vol * np.sqrt(252)) if vol > 1e-12 else 0.0
        rows.append({"period": period, "n_days": len(sub), "ann_return": float(sub["pnl"].mean() * 252),
                      "sharpe": float(sharpe)})
    return pd.DataFrame(rows)
