import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.composite_training import train_composite
from ren.live_system import backtest_ren, backtest_equal_weight

RESULTS = Path(__file__).resolve().parent.parent / "results"

LOSS_KWARGS = dict(sharpe_weight=1.0, drawdown_weight=0.15, turnover_weight=0.08,
                     concentration_weight=0.05, l2_weight=0.01)


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
    dates_train, dates_test = ds.dates[:split], ds.dates[split:]

    print(f"train n={split} ({dates_train[0].date()}..{dates_train[-1].date()}), "
          f"test n={n-split} ({dates_test[0].date()}..{dates_test[-1].date()})")
    print(f"loss_kwargs = {LOSS_KWARGS}")

    eqw = backtest_equal_weight(fwd_test, dates_test)
    print(f"\nbenchmark equal-weight buy&hold, TEST: {summarize(eqw)}")

    seed_results = []
    for seed in range(5):
        t0 = time.time()
        op, hist = train_composite(X_train, fwd_train, panel, dates_train, window_len=21,
                                      n_windows_per_epoch=50, epochs=20, lr=5e-4, seed=seed,
                                      loss_kwargs=LOSS_KWARGS)
        train_time = time.time() - t0

        test_res = backtest_ren(op, X_test, fwd_test, dates_test)
        train_res = backtest_ren(op, X_train, fwd_train, dates_train)

        row = {"seed": seed, "train_time_s": train_time,
                "final_epoch_loss": hist[-1]["loss"], "final_epoch_window_sharpe": hist[-1]["window_sharpe"],
                "test": summarize(test_res), "train_in_sample": summarize(train_res)}
        seed_results.append(row)
        print(f"\nseed={seed} ({train_time:.0f}s) final_window_sharpe={hist[-1]['window_sharpe']:+.3f}")
        print(f"  TEST (held-out):     {summarize(test_res)}")
        print(f"  TRAIN (in-sample):   {summarize(train_res)}")

        torch.save(op.state_dict(), RESULTS / f"ren_composite_trained_seed{seed}.pt")

    test_sharpes = np.array([r["test"]["sharpe"] for r in seed_results])
    test_returns = np.array([r["test"]["ann_return"] for r in seed_results])
    tstat, pval = stats.ttest_1samp(test_sharpes, 0)

    print("\n=== 5-SEED SUMMARY, HELD-OUT TEST ===")
    print(f"Sharpe per seed: {test_sharpes}")
    print(f"mean={test_sharpes.mean():.3f} std={test_sharpes.std():.3f} "
          f"t-test vs 0: t={tstat:.3f} p={pval:.4f}")
    print(f"ann_return per seed: {test_returns}")
    print(f"fraction of seeds beating buy-and-hold (Sharpe {eqw.sharpe:.3f}): "
          f"{(test_sharpes > eqw.sharpe).mean():.2f}")
    print(f"fraction of seeds with positive Sharpe: {(test_sharpes > 0).mean():.2f}")

    report = {
        "loss_kwargs": LOSS_KWARGS,
        "benchmark_equal_weight_test": summarize(eqw),
        "seed_results": seed_results,
        "test_sharpe_mean": float(test_sharpes.mean()),
        "test_sharpe_std": float(test_sharpes.std()),
        "test_sharpe_ttest_stat": float(tstat),
        "test_sharpe_ttest_pvalue": float(pval),
        "fraction_seeds_beating_buyhold": float((test_sharpes > eqw.sharpe).mean()),
        "fraction_seeds_positive_sharpe": float((test_sharpes > 0).mean()),
    }
    with open(RESULTS / "composite_training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved:", RESULTS / "composite_training_report.json")


if __name__ == "__main__":
    main()
