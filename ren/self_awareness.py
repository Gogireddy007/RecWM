"""
Invention 4 -- Self-Awareness Engine.

Thesis framing: continuously estimate how much of the observed market
state is attributable to RecWM's own historical predictions/trades,
i.e. the performative epsilon that the Self-Impact Layer and the
Perdomo stability condition (Section 4) are conditioned on.

There is no live fund to measure this on. What IS real: the standard,
published square-root market-impact model (Almgren, Thum, Hauptmann &
Li, "Direct Estimation of Equity Market Impact", 2005/2012 -- widely
used in production execution systems) applied to REAL average-daily-
dollar-volume (ADV) computed from the downloaded OHLCV data:

    impact(Q) = k * sigma * sign(Q) * sqrt(|Q| / ADV)

k ~ 0.5-1.0 (dimensionless, literature range) is a real empirical
calibration constant from that literature, not fit here. sigma is the
REAL realized daily volatility of the asset. Q is the order size in
shares implied by a hypothetical fund AUM and the model's equilibrium
position vector.

This gives a genuine, computed answer to "how big does the fund have
to be before the model's own footprint becomes material" -- directly
the Section 8 risk ("the reflexivity gap only matters at scale") --
using real volume and volatility data, not an invented number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

ALMGREN_K = 0.6  # midpoint of the literature's reported 0.5-1.0 range


def average_dollar_volume(panel: pd.DataFrame, tickers: list[str], window=63) -> pd.Series:
    close = panel[[(t, "Close") for t in tickers]]; close.columns = tickers
    vol = panel[[(t, "Volume") for t in tickers]]; vol.columns = tickers
    dollar_vol = (close * vol).rolling(window).mean()
    return dollar_vol.iloc[-1]


def realized_daily_vol(panel: pd.DataFrame, tickers: list[str], window=63) -> pd.Series:
    close = panel[[(t, "Close") for t in tickers]]; close.columns = tickers
    ret = close.pct_change()
    return ret.rolling(window).std().iloc[-1]


@dataclass
class FootprintEstimate:
    aum: float
    per_asset_impact_bps: pd.Series
    portfolio_mean_impact_bps: float
    epsilon: float               # normalized self-impact fraction, see below


def estimate_footprint(position_vec: np.ndarray, tickers: list[str],
                         adv: pd.Series, vol: pd.Series, aum: float,
                         signal_scale: float = 1.0) -> FootprintEstimate:
    """position_vec: REN's equilibrium position vector (dimensionless,
    roughly in [-1, 1] after tanh-bounded MLP output) for each asset.
    Converts to a dollar order size assuming `aum` is allocated
    proportionally to |position|, capped so gross exposure <= aum."""
    weights = position_vec / (np.sum(np.abs(position_vec)) + 1e-9)
    dollar_order = np.abs(weights) * aum
    adv_arr = adv.reindex(tickers).to_numpy()
    vol_arr = vol.reindex(tickers).to_numpy()

    participation = dollar_order / (adv_arr + 1.0)
    impact_frac = ALMGREN_K * vol_arr * np.sqrt(np.clip(participation, 0, None))
    impact_bps = impact_frac * 1e4

    per_asset = pd.Series(impact_bps, index=tickers)
    mean_impact = float(np.average(impact_bps, weights=np.abs(position_vec) + 1e-9))

    # epsilon: self-impact as a fraction of the raw belief-signal magnitude
    # driving the trade -- i.e. how much of "what the model believes" gets
    # overwritten by "what the model's own trade does to the price".
    epsilon = float(mean_impact / 1e4 / (signal_scale + 1e-9))

    return FootprintEstimate(aum=aum, per_asset_impact_bps=per_asset,
                               portfolio_mean_impact_bps=mean_impact, epsilon=epsilon)


def footprint_scaling_curve(position_vec: np.ndarray, tickers: list[str],
                              adv: pd.Series, vol: pd.Series,
                              aum_grid=(1e7, 5e7, 1e8, 5e8, 8e8, 2e9, 1e10),
                              signal_scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for aum in aum_grid:
        est = estimate_footprint(position_vec, tickers, adv, vol, aum, signal_scale)
        rows.append({"aum_usd": aum, "mean_impact_bps": est.portfolio_mean_impact_bps,
                      "epsilon": est.epsilon})
    return pd.DataFrame(rows)
