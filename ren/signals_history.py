"""
Vectorized, full-history versions of the agent-type signal functions in
belief_fields.py. belief_fields.py deliberately keeps each signal
function "just compute today's value" for clarity/live-deployment
semantics; this module computes the *entire* real historical time
series in one pass (pandas is already vectorized over the rolling
windows), so building thousands of daily snapshots costs O(T) total
instead of O(T^2) from naively re-slicing the panel per day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ren.belief_fields import TIMESCALES


def _cols(panel, tickers, field):
    df = panel[[(t, field) for t in tickers]]
    df.columns = tickers
    return df


def momentum_series(panel, tickers, window):
    close = _cols(panel, tickers, "Close")
    ret = close.pct_change(window)
    return (ret - ret.rolling(252, min_periods=window).mean()) / (
        ret.rolling(252, min_periods=window).std() + 1e-9)


def mean_reversion_series(panel, tickers, window):
    close = _cols(panel, tickers, "Close")
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return -(close - ma) / (sd + 1e-9)


def volatility_targeting_series(panel, tickers, window):
    close = _cols(panel, tickers, "Close")
    ret = close.pct_change()
    vol = ret.rolling(window).std()
    dvol = vol.diff(window)
    return -dvol / (vol.rolling(252, min_periods=window).std() + 1e-9)


def carry_macro_series(panel, tickers, window):
    close = _cols(panel, tickers, "Close")
    ret = close.pct_change(window)
    basket_ret = ret.mean(axis=1)
    rel = ret.sub(basket_ret, axis=0)
    return rel / (rel.rolling(252, min_periods=window).std() + 1e-9)


def liquidity_provision_series(panel, tickers, window):
    high = _cols(panel, tickers, "High")
    low = _cols(panel, tickers, "Low")
    close = _cols(panel, tickers, "Close")
    rng = (high - low) / (close + 1e-9)
    rng_ma = rng.rolling(window).mean()
    return -(rng_ma - rng_ma.rolling(252, min_periods=window).mean()) / (
        rng_ma.rolling(252, min_periods=window).std() + 1e-9)


AGENT_SERIES_FNS = {
    "momentum": momentum_series,
    "mean_reversion": mean_reversion_series,
    "volatility_targeting": volatility_targeting_series,
    "carry_macro": carry_macro_series,
    "liquidity_provision": liquidity_provision_series,
}


def forward_return_series(panel, tickers, horizon):
    """Real, realized forward return per asset -- used ONLY as a label
    for backtests, never as a model input (no lookahead)."""
    close = _cols(panel, tickers, "Close")
    return close.pct_change(horizon).shift(-horizon)


def realized_vol_series(panel, tickers, window=21):
    close = _cols(panel, tickers, "Close")
    ret = close.pct_change()
    return ret.rolling(window).std() * np.sqrt(252)


class HistoryCache:
    """Precomputes every agent-type signal at every timescale, once, over
    the full real panel. All downstream snapshot construction is then
    O(1) per day (a dict/row lookup)."""

    def __init__(self, panel: pd.DataFrame, tickers: list[str]):
        self.panel = panel
        self.tickers = tickers
        self.index = panel.index
        self.series = {}
        for theta, fn in AGENT_SERIES_FNS.items():
            for s_name, window in TIMESCALES.items():
                self.series[(theta, s_name)] = fn(panel, tickers, window).fillna(0.0).replace(
                    [np.inf, -np.inf], 0.0)
        self.fwd_ret_1d = forward_return_series(panel, tickers, 1).fillna(0.0)
        self.fwd_ret_5d = forward_return_series(panel, tickers, 5).fillna(0.0)
        self.rvol_21d = realized_vol_series(panel, tickers, 21).bfill().fillna(0.0)

    def belief_matrix(self, date, timescale="medium") -> np.ndarray:
        """(5, n_assets) raw signal matrix at `date`."""
        rows = [self.series[(theta, timescale)].loc[date].to_numpy()
                for theta in AGENT_SERIES_FNS]
        return np.vstack(rows)

    def constraint_vector(self, date) -> np.ndarray:
        """Real, time-varying risk-limit proxy: rolling realized vol per
        asset (z-scored against its own trailing history so it is on a
        comparable scale to the belief signals)."""
        v = self.rvol_21d.loc[date].to_numpy()
        hist = self.rvol_21d.loc[:date]
        mu, sd = hist.mean().to_numpy(), hist.std().to_numpy() + 1e-9
        return (v - mu) / sd

    def valid_dates(self, min_history=280):
        return self.index[min_history:]
