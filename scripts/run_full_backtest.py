import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.equilibrium_engine import RENOperator
from ren.training import train_ren_operator, finite_difference_grad_check
from ren.live_system import (
    backtest_ren, backtest_momentum_baseline, backtest_equal_weight, backtest_random,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def summarize(res):
    return {
        "name": res.name, "ann_return": res.ann_return, "ann_vol": res.ann_vol,
        "sharpe": res.sharpe, "max_drawdown": res.max_drawdown, "hit_rate": res.hit_rate,
        "avg_daily_turnover": res.avg_daily_turnover,
        "n_days": len(res.daily_pnl),
    }


def main():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    ds = build_snapshot_dataset(panel, tickers)
    n = ds.X.shape[0]
    split = int(n * 0.8)
    X_train, X_test = ds.X[:split], ds.X[split:]
    fwd_train, fwd_test = ds.fwd_ret_1d[:split], ds.fwd_ret_1d[split:]
    dates_test = ds.dates[split:]
    print(f"train n={split} ({ds.dates[0].date()}..{ds.dates[split-1].date()}), "
          f"test n={n-split} ({dates_test[0].date()}..{dates_test[-1].date()})")

    print("\n[1] Gradient check on untrained op (finite-diff vs autograd through 10-step unroll)...")
    op_check = RENOperator(seed=42, damping=0.35, target_norm=0.6)
    rel_err = finite_difference_grad_check(op_check, X_train[0], fwd_train[0], n_unroll=10)
    print(f"    relative gradient error: {rel_err:.6f} (should be small, e.g. < 0.05)")

    print("\n[2] Backtesting UNTRAINED REN operator (random seeded weights) out-of-sample...")
    op_untrained = RENOperator(seed=42, damping=0.35, target_norm=0.6)
    op_untrained.eval()
    res_untrained = backtest_ren(op_untrained, X_test, fwd_test, dates_test)
    print("   ", summarize(res_untrained))

    print("\n[3] Training REN operator end-to-end on REAL training-period P&L "
          "(truncated unroll, direct P&L objective)...")
    t0 = time.time()
    op_trained, loss_hist = train_ren_operator(X_train, fwd_train, epochs=60, batch_size=64,
                                                  lr=3e-4, n_unroll=15, seed=42)
    print(f"    trained in {time.time()-t0:.1f}s, loss[0]={loss_hist[0]:.6f} -> loss[-1]={loss_hist[-1]:.6f}")
    op_trained.eval()

    print("\n[4] Backtesting TRAINED REN operator out-of-sample (held-out test period, "
          "strictly after training window, no lookahead)...")
    res_trained = backtest_ren(op_trained, X_test, fwd_test, dates_test)
    print("   ", summarize(res_trained))

    print("\n[5] Also backtesting TRAINED operator IN-SAMPLE (train period) for overfitting check...")
    res_trained_in_sample = backtest_ren(op_trained, X_train, fwd_train, ds.dates[:split])
    print("   ", summarize(res_trained_in_sample))

    print("\n[6] Baselines on the SAME held-out test period...")
    res_naive = backtest_momentum_baseline(X_test, fwd_test, dates_test)
    res_eqw = backtest_equal_weight(fwd_test, dates_test)
    res_rand = backtest_random(X_test, fwd_test, dates_test, seed=0)
    print("   ", summarize(res_naive))
    print("   ", summarize(res_eqw))
    print("   ", summarize(res_rand))

    report = {
        "gradient_check_relative_error": rel_err,
        "train_period": [str(ds.dates[0].date()), str(ds.dates[split-1].date())],
        "test_period": [str(dates_test[0].date()), str(dates_test[-1].date())],
        "training_loss_curve": loss_hist,
        "results": {
            "ren_untrained_oos": summarize(res_untrained),
            "ren_trained_oos": summarize(res_trained),
            "ren_trained_in_sample": summarize(res_trained_in_sample),
            "naive_belief_baseline_oos": summarize(res_naive),
            "equal_weight_oos": summarize(res_eqw),
            "random_baseline_oos": summarize(res_rand),
        },
    }
    with open(RESULTS / "full_backtest_report.json", "w") as f:
        json.dump(report, f, indent=2)

    np.save(RESULTS / "ren_trained_oos_pnl.npy", res_trained.daily_pnl)
    np.save(RESULTS / "ren_untrained_oos_pnl.npy", res_untrained.daily_pnl)
    torch.save(op_trained.state_dict(), RESULTS / "ren_trained_operator.pt")

    print("\n=== SUMMARY TABLE (all out-of-sample except noted) ===")
    for k, v in report["results"].items():
        print(f"{k:32s} sharpe={v['sharpe']:+.3f}  ann_ret={v['ann_return']:+.2%}  "
              f"ann_vol={v['ann_vol']:.2%}  maxDD={v['max_drawdown']:+.2%}  hit={v['hit_rate']:.1%}")


if __name__ == "__main__":
    main()
