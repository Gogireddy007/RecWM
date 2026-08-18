"""
Invention 10 -- Live Architecture & Fusion.

Wires every other invention into one pipeline and runs a real, walk-
forward backtest on held-out real market data:

  real OHLCV -> Belief Fields -> Influence Kernel consensus ->
  Equilibrium Engine (position_vec = equilibrium trade) ->
  realized real forward return -> real P&L, with real transaction
  costs, benchmarked against real, simple baselines.

No step in this chain uses fabricated data or a fabricated result;
every number in the returned BacktestResult is computed directly from
real prices and the actual model output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ren.equilibrium_engine import anderson_acceleration, Z_DIM, N_ASSETS


@dataclass
class BacktestResult:
    name: str
    dates: pd.DatetimeIndex
    daily_pnl: np.ndarray
    daily_pnl_net: np.ndarray
    turnover: np.ndarray
    positions: np.ndarray
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    avg_daily_turnover: float


def _metrics(name, dates, pnl_net, turnover, positions) -> BacktestResult:
    pnl_net = np.asarray(pnl_net)
    ann_return = float(pnl_net.mean() * 252)
    ann_vol = float(pnl_net.std() * np.sqrt(252))
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-9 else 0.0
    cum = np.cumsum(pnl_net)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_dd = float(drawdown.min())
    hit_rate = float((pnl_net > 0).mean())
    return BacktestResult(name=name, dates=dates, daily_pnl=pnl_net, daily_pnl_net=pnl_net,
                            turnover=np.asarray(turnover), positions=np.asarray(positions),
                            ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe,
                            max_drawdown=max_dd, hit_rate=hit_rate,
                            avg_daily_turnover=float(np.mean(turnover)))


def backtest_ren(op, X_test: torch.Tensor, fwd_ret_test: torch.Tensor, dates,
                   tc_bps=5.0, max_iter=100, tol=1e-5, gross_cap=1.0) -> BacktestResult:
    n = X_test.shape[0]
    z = torch.zeros(Z_DIM)
    pnl_gross, pnl_net, turns, positions = [], [], [], []
    prev_pos = np.zeros(N_ASSETS)
    with torch.no_grad():
        for i in range(n):
            res = anderson_acceleration(op, z, X_test[i], max_iter=max_iter, tol=tol)
            z = res.z_star
            pos = z[N_ASSETS:].numpy()
            gross = np.abs(pos).sum()
            if gross > gross_cap:
                pos = pos / gross * gross_cap
            fwd = fwd_ret_test[i].numpy()
            gross_pnl = float(np.dot(pos, fwd))
            turnover = float(np.abs(pos - prev_pos).sum())
            cost = turnover * tc_bps / 1e4
            pnl_gross.append(gross_pnl)
            pnl_net.append(gross_pnl - cost)
            turns.append(turnover)
            positions.append(pos.copy())
            prev_pos = pos
    return _metrics("REN equilibrium", dates, pnl_net, turns, positions)


def backtest_momentum_baseline(X_test: torch.Tensor, fwd_ret_test: torch.Tensor, dates,
                                  tc_bps=5.0, gross_cap=1.0) -> BacktestResult:
    """Simple real baseline: position proportional to the raw aggregate
    belief vector (dominated by momentum/mean-reversion/etc signals),
    with NO equilibrium computation -- i.e. what a standard, non-
    reflexive ML pipeline would do with the same real inputs."""
    agg_belief = X_test[:, :N_ASSETS].numpy()
    pnl_net, turns, positions = [], [], []
    prev_pos = np.zeros(N_ASSETS)
    for i in range(agg_belief.shape[0]):
        raw = agg_belief[i]
        gross = np.abs(raw).sum()
        pos = raw / gross * gross_cap if gross > 1e-9 else raw
        fwd = fwd_ret_test[i].numpy()
        gross_pnl = float(np.dot(pos, fwd))
        turnover = float(np.abs(pos - prev_pos).sum())
        cost = turnover * tc_bps / 1e4
        pnl_net.append(gross_pnl - cost)
        turns.append(turnover)
        positions.append(pos.copy())
        prev_pos = pos
    return _metrics("Naive belief baseline (no equilibrium)", dates, pnl_net, turns, positions)


def backtest_equal_weight(fwd_ret_test: torch.Tensor, dates) -> BacktestResult:
    n_assets = fwd_ret_test.shape[1]
    pos = np.full(n_assets, 1.0 / n_assets)
    pnl = [float(np.dot(pos, fwd_ret_test[i].numpy())) for i in range(fwd_ret_test.shape[0])]
    turns = [0.0] * fwd_ret_test.shape[0]
    positions = [pos] * fwd_ret_test.shape[0]
    return _metrics("Equal-weight buy & hold", dates, pnl, turns, positions)


def backtest_random(X_test: torch.Tensor, fwd_ret_test: torch.Tensor, dates,
                       tc_bps=5.0, gross_cap=1.0, seed=0) -> BacktestResult:
    rng = np.random.default_rng(seed)
    n, n_assets = fwd_ret_test.shape
    pnl_net, turns, positions = [], [], []
    prev_pos = np.zeros(n_assets)
    for i in range(n):
        raw = rng.normal(0, 1, n_assets)
        pos = raw / np.abs(raw).sum() * gross_cap
        fwd = fwd_ret_test[i].numpy()
        gross_pnl = float(np.dot(pos, fwd))
        turnover = float(np.abs(pos - prev_pos).sum())
        cost = turnover * tc_bps / 1e4
        pnl_net.append(gross_pnl - cost)
        turns.append(turnover)
        positions.append(pos.copy())
        prev_pos = pos
    return _metrics("Random baseline (null)", dates, pnl_net, turns, positions)
