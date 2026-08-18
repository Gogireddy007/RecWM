"""
Standard, real, independently-implemented baseline market models, run on
the EXACT same real data, test period, and transaction-cost assumption
(5bps of turnover) as every REN backtest in this project -- so the
comparison is fair rather than REN being measured under one protocol and
baselines under another.

Six baselines, spanning both "classical quant" and "standard ML" so the
comparison set matches what the thesis itself frames as REN's
alternative: static allocation (60/40), risk-based allocation (inverse-
vol), two classical factor strategies (time-series momentum, short-term
mean-reversion), and two standard supervised-ML models (ridge
regression, random forest) trained on the identical feature set REN
uses (the same real 32-dim belief+constraint snapshot), so the
comparison also directly tests the thesis's own claim that "the
standard ML move" (fit P(x_t+1|x_1:t) as an exogenous process) is what
REN is supposed to beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from ren.equilibrium_engine import N_ASSETS
from ren.live_system import _metrics, BacktestResult


def backtest_static_weights(weights: np.ndarray, fwd_ret_test: torch.Tensor, dates, name: str,
                              tc_bps=5.0) -> BacktestResult:
    n = fwd_ret_test.shape[0]
    pnl_net, turns, positions = [], [], []
    prev = np.zeros(N_ASSETS)
    for i in range(n):
        pos = weights.copy()
        fwd = fwd_ret_test[i].numpy()
        turnover = float(np.abs(pos - prev).sum())
        cost = turnover * tc_bps / 1e4
        pnl_net.append(float(np.dot(pos, fwd)) - cost)
        turns.append(turnover)
        positions.append(pos.copy())
        prev = pos
    return _metrics(name, dates, pnl_net, turns, positions)


def backtest_60_40(tickers: list[str], fwd_ret_test, dates, tc_bps=5.0) -> BacktestResult:
    w = np.zeros(len(tickers))
    if "SPY" in tickers:
        w[tickers.index("SPY")] = 0.6
    if "TLT" in tickers:
        w[tickers.index("TLT")] = 0.4
    return backtest_static_weights(w, fwd_ret_test, dates, "60/40 (SPY/TLT)", tc_bps)


def backtest_inverse_vol(panel: pd.DataFrame, tickers: list[str], dates_test, fwd_ret_test,
                            tc_bps=5.0, window=21) -> BacktestResult:
    close = panel[[(t, "Close") for t in tickers]]; close.columns = tickers
    vol = close.pct_change().rolling(window).std()
    n = fwd_ret_test.shape[0]
    pnl_net, turns, positions = [], [], []
    prev = np.zeros(N_ASSETS)
    for i, d in enumerate(dates_test):
        v = vol.loc[d].to_numpy()
        inv = 1.0 / (v + 1e-6)
        pos = inv / inv.sum()
        fwd = fwd_ret_test[i].numpy()
        turnover = float(np.abs(pos - prev).sum())
        cost = turnover * tc_bps / 1e4
        pnl_net.append(float(np.dot(pos, fwd)) - cost)
        turns.append(turnover)
        positions.append(pos.copy())
        prev = pos
    return _metrics("Risk parity (inverse-vol)", dates_test, pnl_net, turns, positions)


def backtest_ts_momentum(panel: pd.DataFrame, tickers: list[str], dates_test, fwd_ret_test,
                            tc_bps=5.0, lookback=252, vol_window=21, gross_cap=1.0) -> BacktestResult:
    """Classic time-series momentum (Moskowitz, Ooi & Pedersen 2012):
    position ~ sign(12m trailing return) / trailing vol, per asset."""
    close = panel[[(t, "Close") for t in tickers]]; close.columns = tickers
    ret12m = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_window).std()
    n = fwd_ret_test.shape[0]
    pnl_net, turns, positions = [], [], []
    prev = np.zeros(N_ASSETS)
    for i, d in enumerate(dates_test):
        sig = np.sign(ret12m.loc[d].to_numpy())
        v = vol.loc[d].to_numpy()
        raw = sig / (v + 1e-6)
        gross = np.abs(raw).sum()
        pos = raw / gross * gross_cap if gross > 1e-9 else raw
        fwd = fwd_ret_test[i].numpy()
        turnover = float(np.abs(pos - prev).sum())
        cost = turnover * tc_bps / 1e4
        pnl_net.append(float(np.dot(pos, fwd)) - cost)
        turns.append(turnover)
        positions.append(pos.copy())
        prev = pos
    return _metrics("Time-series momentum (12m)", dates_test, pnl_net, turns, positions)


def backtest_mean_reversion(panel: pd.DataFrame, tickers: list[str], dates_test, fwd_ret_test,
                               tc_bps=5.0, lookback=5, gross_cap=1.0) -> BacktestResult:
    close = panel[[(t, "Close") for t in tickers]]; close.columns = tickers
    ret = close.pct_change(lookback)
    z = (ret - ret.rolling(252).mean()) / (ret.rolling(252).std() + 1e-9)
    n = fwd_ret_test.shape[0]
    pnl_net, turns, positions = [], [], []
    prev = np.zeros(N_ASSETS)
    for i, d in enumerate(dates_test):
        raw = -z.loc[d].to_numpy()
        raw = np.nan_to_num(raw)
        gross = np.abs(raw).sum()
        pos = raw / gross * gross_cap if gross > 1e-9 else raw
        fwd = fwd_ret_test[i].numpy()
        turnover = float(np.abs(pos - prev).sum())
        cost = turnover * tc_bps / 1e4
        pnl_net.append(float(np.dot(pos, fwd)) - cost)
        turns.append(turnover)
        positions.append(pos.copy())
        prev = pos
    return _metrics("Short-term mean-reversion (5d)", dates_test, pnl_net, turns, positions)


def _ml_backtest(model_factory, name, X_train, fwd_train, X_test, fwd_test, dates_test,
                   tc_bps=5.0, gross_cap=1.0):
    """Fits ONE model per asset (16 independent regressions), trained
    once on TRAIN, evaluated on TEST -- same protocol as REN (fit once,
    no daily refit / no lookahead), for a fair comparison."""
    Xtr = X_train.numpy()
    Ytr = fwd_train.numpy()
    Xte = X_test.numpy()

    preds = np.zeros((Xte.shape[0], N_ASSETS))
    for a in range(N_ASSETS):
        model = model_factory()
        model.fit(Xtr, Ytr[:, a])
        preds[:, a] = model.predict(Xte)

    n = Xte.shape[0]
    pnl_net, turns, positions = [], [], []
    prev = np.zeros(N_ASSETS)
    for i in range(n):
        raw = preds[i]
        gross = np.abs(raw).sum()
        pos = raw / gross * gross_cap if gross > 1e-9 else raw
        fwd = fwd_test[i].numpy()
        turnover = float(np.abs(pos - prev).sum())
        cost = turnover * tc_bps / 1e4
        pnl_net.append(float(np.dot(pos, fwd)) - cost)
        turns.append(turnover)
        positions.append(pos.copy())
        prev = pos
    return _metrics(name, dates_test, pnl_net, turns, positions)


def backtest_ridge(X_train, fwd_train, X_test, fwd_test, dates_test, tc_bps=5.0, alpha=10.0):
    return _ml_backtest(lambda: Ridge(alpha=alpha), "Ridge regression (ML baseline)",
                          X_train, fwd_train, X_test, fwd_test, dates_test, tc_bps)


def backtest_random_forest(X_train, fwd_train, X_test, fwd_test, dates_test, tc_bps=5.0,
                              n_estimators=200, max_depth=4, seed=0):
    return _ml_backtest(
        lambda: RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                        random_state=seed, n_jobs=-1),
        "Random forest (ML baseline)", X_train, fwd_train, X_test, fwd_test, dates_test, tc_bps)
