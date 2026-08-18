"""
Genuine walk-forward test on real data that did not exist when any model
in this project was built: re-downloaded market data adds exactly one
new real trading day (2026-08-12) beyond the original snapshot
(2026-08-11) all backtests in this project were run against.

Honesty note: n=1 new day has essentially zero statistical power on its
own. This is not a replacement for the 524-day held-out backtest
elsewhere in this project -- it is the single most out-of-sample data
point available, literally nonexistent at the time every model here was
trained, and is reported as exactly that: one real data point, not a
verdict.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.equilibrium_engine import RENOperator, N_ASSETS
from ren.live_system import backtest_ren, backtest_equal_weight

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    old_panel = pd.read_parquet(DATA / "market_panel.parquet")
    fresh_panel = pd.read_parquet(DATA / "market_panel_fresh.parquet")
    tickers = sorted({c[0] for c in fresh_panel.columns})

    new_dates = fresh_panel.index.difference(old_panel.index)
    print(f"Genuinely new real trading days since the original snapshot: {list(new_dates.date)}")
    if len(new_dates) == 0:
        print("No new trading day yet (markets haven't produced one since the original download). Stopping.")
        return

    # BUG CAUGHT AND FIXED: forward_return_series shifts by -1 and the
    # LAST day in any panel therefore has an undefined (NaN) forward
    # return, which HistoryCache silently fillna(0.0)'s -- meaning the
    # newest day (2026-08-12) has NO real resolved outcome yet (we'd
    # need 2026-08-13's close). The genuinely new, REAL, resolved data
    # point available right now is the PREVIOUS last day (2026-08-11),
    # whose forward return only just became knowable with 2026-08-12's
    # close, and which was ITSELF a fabricated-zero day in every
    # backtest run earlier in this project (it was the last day of the
    # old panel too). We evaluate that date instead, and print both for
    # transparency.
    resolved_new_dates = [d for d in new_dates if d != fresh_panel.index.max()]
    prior_last_date = old_panel.index.max()
    print(f"NOTE: {fresh_panel.index.max().date()} has no resolved forward return yet (needs tomorrow's close). "
          f"The real, newly-resolved out-of-sample day is {prior_last_date.date()} "
          f"(fabricated as a zero-return day in every prior backtest in this project; now real).")
    target_dates = set(resolved_new_dates) | {prior_last_date}

    ds = build_snapshot_dataset(fresh_panel, tickers)
    n = ds.X.shape[0]
    # keep the SAME test-period start date used throughout this project,
    # now extended through the new real day(s), so the walk-forward state
    # (z carried day to day) is consistent with every other backtest here
    old_ds = build_snapshot_dataset(old_panel, tickers)
    split = int(old_ds.X.shape[0] * 0.8)
    test_start_date = old_ds.dates[split]

    test_mask = ds.dates >= test_start_date
    X_test = ds.X[test_mask.to_numpy() if hasattr(test_mask, "to_numpy") else test_mask]
    fwd_test = ds.fwd_ret_1d[test_mask.to_numpy() if hasattr(test_mask, "to_numpy") else test_mask]
    dates_test = ds.dates[test_mask]
    print(f"Extended TEST period: {dates_test[0].date()} .. {dates_test[-1].date()} ({len(dates_test)} days)")

    new_day_idx = [i for i, d in enumerate(dates_test) if d in target_dates]
    print(f"Index/indices of the new day(s) within the extended test walk: {new_day_idx}")

    checkpoints = {
        "untrained_seed42": None,  # constructed fresh, no checkpoint
        "ift_trained": RESULTS / "ren_ift_trained_operator.pt",
        "seismograph_trained": RESULTS / "ren_seismograph_trained_operator.pt",
        "composite_seed0": RESULTS / "ren_composite_trained_seed0.pt",
        "composite_seed1": RESULTS / "ren_composite_trained_seed1.pt",
        "composite_seed2": RESULTS / "ren_composite_trained_seed2.pt",
        "composite_seed3": RESULTS / "ren_composite_trained_seed3.pt",
        "composite_seed4": RESULTS / "ren_composite_trained_seed4.pt",
    }

    eqw = backtest_equal_weight(fwd_test, dates_test)
    report = {"new_real_days": [str(d.date()) for d in new_dates],
               "extended_test_range": [str(dates_test[0].date()), str(dates_test[-1].date())],
               "equal_weight_new_day_pnl": float(eqw.daily_pnl[new_day_idx[-1]]) if new_day_idx else None,
               "models": {}}

    for name, ckpt_path in checkpoints.items():
        op = RENOperator(seed=42, damping=0.35, target_norm=0.6)
        if ckpt_path is not None:
            op.load_state_dict(torch.load(ckpt_path))
        op.eval()

        res = backtest_ren(op, X_test, fwd_test, dates_test)
        new_day_pnl = float(res.daily_pnl[new_day_idx[-1]]) if new_day_idx else None
        new_day_position = res.positions[new_day_idx[-1]] if new_day_idx else None

        print(f"\n[{name}] full extended-test Sharpe={res.sharpe:.3f} ann_ret={res.ann_return:.2%} | "
              f"NEW DAY ({dates_test[new_day_idx[-1]].date() if new_day_idx else 'n/a'}) pnl={new_day_pnl}")

        report["models"][name] = {
            "full_extended_test_sharpe": res.sharpe,
            "full_extended_test_ann_return": res.ann_return,
            "new_day_pnl": new_day_pnl,
            "new_day_gross_exposure": float(np.abs(new_day_position).sum()) if new_day_position is not None else None,
        }

    print(f"\n[equal_weight buy&hold] NEW DAY pnl={report['equal_weight_new_day_pnl']}")

    with open(RESULTS / "forward_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved:", RESULTS / "forward_test_report.json")


if __name__ == "__main__":
    main()
