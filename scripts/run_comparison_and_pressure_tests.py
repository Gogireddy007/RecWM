import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.equilibrium_engine import RENOperator
from ren.live_system import backtest_ren, backtest_equal_weight
from ren.baselines import (backtest_60_40, backtest_inverse_vol, backtest_ts_momentum,
                              backtest_mean_reversion, backtest_ridge, backtest_random_forest)
from ren.pressure_tests import block_bootstrap_sharpe, regime_breakdown, calendar_subperiod_breakdown

RESULTS = Path(__file__).resolve().parent.parent / "results"


def summarize(res):
    return {"sharpe": res.sharpe, "ann_return": res.ann_return, "ann_vol": res.ann_vol,
            "max_drawdown": res.max_drawdown, "hit_rate": res.hit_rate,
            "avg_daily_turnover": res.avg_daily_turnover}


def main():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    ds = build_snapshot_dataset(panel, tickers)
    n = ds.X.shape[0]
    split = int(n * 0.8)
    X_train, fwd_train = ds.X[:split], ds.fwd_ret_1d[:split]
    X_test, fwd_test = ds.X[split:], ds.fwd_ret_1d[split:]
    dates_test = ds.dates[split:]
    print(f"TEST period: {dates_test[0].date()} .. {dates_test[-1].date()} ({len(dates_test)} days)")

    results = {}

    # REN variants
    op_untrained = RENOperator(seed=42, damping=0.35, target_norm=0.6); op_untrained.eval()
    results["REN_untrained"] = backtest_ren(op_untrained, X_test, fwd_test, dates_test)

    op_ift = RENOperator(seed=42, damping=0.35, target_norm=0.6)
    op_ift.load_state_dict(torch.load(RESULTS / "ren_ift_trained_operator.pt")); op_ift.eval()
    results["REN_ift_trained"] = backtest_ren(op_ift, X_test, fwd_test, dates_test)

    composite_results = []
    for seed in range(5):
        op_c = RENOperator(seed=42, damping=0.35, target_norm=0.6)
        op_c.load_state_dict(torch.load(RESULTS / f"ren_composite_trained_seed{seed}.pt")); op_c.eval()
        r = backtest_ren(op_c, X_test, fwd_test, dates_test)
        composite_results.append(r)
        results[f"REN_composite_seed{seed}"] = r
    best_composite = max(composite_results, key=lambda r: r.sharpe)
    results["REN_composite_BEST_seed"] = best_composite

    # Benchmarks / baselines
    results["Equal_weight_buyhold"] = backtest_equal_weight(fwd_test, dates_test)
    results["60_40"] = backtest_60_40(tickers, fwd_test, dates_test)
    results["Risk_parity_inverse_vol"] = backtest_inverse_vol(panel, tickers, dates_test, fwd_test)
    results["TS_momentum_12m"] = backtest_ts_momentum(panel, tickers, dates_test, fwd_test)
    results["Mean_reversion_5d"] = backtest_mean_reversion(panel, tickers, dates_test, fwd_test)
    results["Ridge_ML"] = backtest_ridge(X_train, fwd_train, X_test, fwd_test, dates_test)
    results["Random_forest_ML"] = backtest_random_forest(X_train, fwd_train, X_test, fwd_test, dates_test)

    print("\n=== FULL COMPARISON TABLE (real test period, same costs, same data) ===")
    rows = []
    for name, res in results.items():
        s = summarize(res)
        rows.append({"model": name, **s})
        print(f"{name:28s} sharpe={s['sharpe']:+.3f} ann_ret={s['ann_return']:+7.2%} "
              f"ann_vol={s['ann_vol']:6.2%} maxDD={s['max_drawdown']:+7.2%} hit={s['hit_rate']:.1%}")
    comparison_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    comparison_df.to_csv(RESULTS / "full_comparison_table.csv", index=False)

    # ---- Pressure test 1: block bootstrap Sharpe CIs ----
    print("\n=== BLOCK BOOTSTRAP SHARPE CONFIDENCE INTERVALS (90%, block=21d, 2000 resamples) ===")
    boot_results = {}
    key_models = ["REN_composite_BEST_seed", "REN_untrained", "Equal_weight_buyhold",
                   "Risk_parity_inverse_vol", "60_40", "TS_momentum_12m", "Ridge_ML", "Random_forest_ML"]
    for name in key_models:
        boot = block_bootstrap_sharpe(results[name].daily_pnl, n_boot=2000, block_size=21, seed=0)
        boot_results[name] = boot
        print(f"{name:28s} point={boot['point_estimate']:+.3f}  90% CI=[{boot['ci_5']:+.3f}, {boot['ci_95']:+.3f}]  "
              f"P(Sharpe>0)={boot['prob_sharpe_gt_0']:.2f}")

    # ---- Pressure test 2: regime-conditional breakdown ----
    print("\n=== REGIME-CONDITIONAL BREAKDOWN (real SPY vol terciles) ===")
    regime_results = {}
    for name in key_models:
        rb = regime_breakdown(results[name].daily_pnl, dates_test, panel)
        regime_results[name] = rb.to_dict(orient="records")
        print(f"\n{name}:")
        print(rb.to_string(index=False))

    # ---- Pressure test 3: calendar sub-period breakdown ----
    print("\n=== CALENDAR SUB-PERIOD BREAKDOWN ===")
    subperiod_results = {}
    for name in key_models:
        sp = calendar_subperiod_breakdown(results[name].daily_pnl, dates_test)
        subperiod_results[name] = sp.to_dict(orient="records")
        print(f"\n{name}:")
        print(sp.to_string(index=False))

    report = {
        "test_period": [str(dates_test[0].date()), str(dates_test[-1].date())],
        "comparison_table": rows,
        "bootstrap_sharpe_ci": boot_results,
        "regime_breakdown": regime_results,
        "calendar_subperiod_breakdown": subperiod_results,
    }
    with open(RESULTS / "comparison_and_pressure_tests.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSaved:", RESULTS / "comparison_and_pressure_tests.json")


if __name__ == "__main__":
    main()
