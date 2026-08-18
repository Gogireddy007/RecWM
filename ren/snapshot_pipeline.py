"""
Builds the real, per-day (x_t) input snapshots that feed the Equilibrium
Engine and every downstream invention: x_t = concat(agg_belief_t[16],
constraint_t[16]), all derived from real OHLCV data via signals_history.py
and the fitted Influence Kernel (Invention 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ren.signals_history import HistoryCache
from ren.influence_kernel import compute_influence_kernel, type_mean_signal_matrix


@dataclass
class SnapshotDataset:
    dates: pd.DatetimeIndex
    X: torch.Tensor              # (T, 32)
    fwd_ret_1d: torch.Tensor      # (T, 16) realized next-day return, label-only
    fwd_ret_5d: torch.Tensor
    history: HistoryCache
    kernel_result: object
    tickers: list[str]


def build_snapshot_dataset(panel: pd.DataFrame, tickers: list[str],
                             kernel_fit_lookback=750, min_history=300) -> SnapshotDataset:
    hist = HistoryCache(panel, tickers)
    dates = hist.valid_dates(min_history=min_history)
    # Bug found and fixed: the last date in any panel has an undefined
    # (NaN) 1-day forward return (no next close exists yet), which
    # HistoryCache.fwd_ret_1d silently fillna(0.0)'s -- fabricating a
    # zero-return day rather than leaving it out. Drop it explicitly so
    # no snapshot dataset ever contains a day with a fake resolved
    # outcome. Negligible effect on any aggregate metric already
    # reported (1 day out of ~2600), but real and worth fixing.
    last_close = panel[[(t, "Close") for t in tickers]]
    if len(dates) and pd.isna(last_close.loc[dates[-1]]).all() == False and \
            last_close.pct_change(1).shift(-1).loc[dates[-1]].isna().all():
        dates = dates[:-1]

    # Fit the influence kernel ONCE on the data strictly before the first
    # usable date's midpoint... in practice we fit on the trailing window
    # ending at min_history so no snapshot's kernel uses information from
    # after that snapshot (kernel is a design object, not retrained daily
    # here, to keep the pipeline fast; see live_system.py for the
    # periodically-refit version used in the walk-forward backtest).
    sig_mat, labels, sig_idx = type_mean_signal_matrix(hist, timescale="medium")
    fit_end = min_history
    fit_start = max(0, fit_end - kernel_fit_lookback)
    kernel_result = compute_influence_kernel(sig_mat[fit_start:fit_end], labels)

    rows = []
    for d in dates:
        b = hist.belief_matrix(d, "medium")             # (5, 16)
        consensus = kernel_result.K_star @ b              # (5, 16)
        agg_belief = consensus.mean(axis=0)                # (16,)
        constraint = hist.constraint_vector(d)             # (16,)
        rows.append(np.concatenate([agg_belief, constraint]))

    X = torch.tensor(np.array(rows), dtype=torch.float32)
    fwd1 = torch.tensor(hist.fwd_ret_1d.loc[dates].to_numpy(), dtype=torch.float32)
    fwd5 = torch.tensor(hist.fwd_ret_5d.loc[dates].to_numpy(), dtype=torch.float32)

    return SnapshotDataset(dates=dates, X=X, fwd_ret_1d=fwd1, fwd_ret_5d=fwd5,
                            history=hist, kernel_result=kernel_result, tickers=tickers)
