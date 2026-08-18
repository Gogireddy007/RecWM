"""
Proper train/val/test protocol for the IFT-trained REN operator:

  - TRAIN: 2016-03-14 .. ~2022-11 (first 64% of all snapshots)
  - VAL:   ~2022-11 .. 2024-07-09 (next 16%) -- used ONLY to pick
           hyperparameters (l2_weight x lr grid), never touched by
           training itself
  - TEST:  2024-07-10 .. 2026-08-11 (final 20%, same held-out window
           used throughout this project) -- touched exactly once,
           after the winning hyperparameters are locked in from VAL.

This replaces the previous protocol (train on 80%, evaluate 3 ad hoc
hyperparameter settings directly on the real test set), which risked
implicitly overfitting the reported test number to the test set by
comparing multiple configs against it.
"""
import sys, time, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.ift_training import train_ren_operator_ift
from ren.live_system import backtest_ren

RESULTS = Path(__file__).resolve().parent.parent / "results"


def summarize(res):
    return {"sharpe": res.sharpe, "ann_return": res.ann_return, "ann_vol": res.ann_vol,
            "max_drawdown": res.max_drawdown, "hit_rate": res.hit_rate,
            "avg_daily_turnover": res.avg_daily_turnover, "n_days": len(res.daily_pnl)}


def main():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    ds = build_snapshot_dataset(panel, tickers)
    n = ds.X.shape[0]

    train_end = int(n * 0.64)
    val_end = int(n * 0.80)  # == old 80% split point, so TEST is unchanged from before

    X_train, fwd_train = ds.X[:train_end], ds.fwd_ret_1d[:train_end]
    X_val, fwd_val = ds.X[train_end:val_end], ds.fwd_ret_1d[train_end:val_end]
    X_test, fwd_test = ds.X[val_end:], ds.fwd_ret_1d[val_end:]
    dates_val, dates_test = ds.dates[train_end:val_end], ds.dates[val_end:]

    print(f"train n={train_end} ({ds.dates[0].date()}..{ds.dates[train_end-1].date()})")
    print(f"val   n={val_end-train_end} ({dates_val[0].date()}..{dates_val[-1].date()})")
    print(f"test  n={n-val_end} ({dates_test[0].date()}..{dates_test[-1].date()})")

    grid = list(itertools.product([0.0, 0.005, 0.01, 0.02, 0.05], [1e-4, 3e-4, 1e-3]))
    print(f"\nSearching {len(grid)} (l2_weight, lr) configs, selecting by VAL sharpe...\n")

    search_results = []
    for l2, lr in grid:
        t0 = time.time()
        op, hist = train_ren_operator_ift(X_train, fwd_train, epochs=15, batch_size=32,
                                             lr=lr, seed=42, l2_weight=l2, solve_max_iter=100)
        val_res = backtest_ren(op, X_val, fwd_val, dates_val)
        dt = time.time() - t0
        row = {"l2_weight": l2, "lr": lr, "val_sharpe": val_res.sharpe,
               "val_ann_return": val_res.ann_return, "val_ann_vol": val_res.ann_vol,
               "val_turnover": val_res.avg_daily_turnover,
               "train_loss_final": hist[-1], "train_time_s": dt}
        search_results.append(row)
        print(f"l2={l2:<6} lr={lr:<7} val_sharpe={val_res.sharpe:+.3f} "
              f"val_ret={val_res.ann_return:+.2%} val_vol={val_res.ann_vol:.2%} "
              f"turnover={val_res.avg_daily_turnover:.4f}  ({dt:.1f}s)")

    search_df = pd.DataFrame(search_results)
    search_df.to_csv(RESULTS / "ift_hparam_search.csv", index=False)

    best = search_df.loc[search_df["val_sharpe"].idxmax()]
    print(f"\nBest by VAL sharpe: l2={best.l2_weight} lr={best.lr} val_sharpe={best.val_sharpe:.3f}")

    print("\nRetraining winning config on TRAIN+VAL combined, evaluating ONCE on TEST...")
    X_trainval = torch.cat([X_train, X_val], dim=0)
    fwd_trainval = torch.cat([fwd_train, fwd_val], dim=0)
    op_final, hist_final = train_ren_operator_ift(X_trainval, fwd_trainval, epochs=15, batch_size=32,
                                                      lr=float(best.lr), seed=42,
                                                      l2_weight=float(best.l2_weight), solve_max_iter=100)

    test_res = backtest_ren(op_final, X_test, fwd_test, dates_test)
    trainval_res = backtest_ren(op_final, X_trainval, fwd_trainval, ds.dates[:val_end])

    print("\n=== FINAL, HONEST, SINGLE TEST-SET EVALUATION ===")
    print("TEST (never touched during hyperparameter selection):", summarize(test_res))
    print("TRAIN+VAL (in-sample, for overfitting comparison):   ", summarize(trainval_res))

    # multi-seed robustness of the WINNING config
    print("\nMulti-seed robustness of winning hyperparameters (5 training seeds)...")
    seed_rows = []
    for seed in range(5):
        op_s, _ = train_ren_operator_ift(X_trainval, fwd_trainval, epochs=15, batch_size=32,
                                            lr=float(best.lr), seed=seed,
                                            l2_weight=float(best.l2_weight), solve_max_iter=100)
        res_s = backtest_ren(op_s, X_test, fwd_test, dates_test)
        seed_rows.append({"seed": seed, **summarize(res_s)})
        print(f"  seed={seed} test_sharpe={res_s.sharpe:+.3f} ann_ret={res_s.ann_return:+.2%}")
    seed_df = pd.DataFrame(seed_rows)

    report = {
        "protocol": {"train_end": str(ds.dates[train_end-1].date()),
                      "val_range": [str(dates_val[0].date()), str(dates_val[-1].date())],
                      "test_range": [str(dates_test[0].date()), str(dates_test[-1].date())]},
        "hparam_grid_results": search_results,
        "winning_config": {"l2_weight": float(best.l2_weight), "lr": float(best.lr),
                             "val_sharpe": float(best.val_sharpe)},
        "final_test_result": summarize(test_res),
        "final_trainval_in_sample_result": summarize(trainval_res),
        "multi_seed_test_results": seed_rows,
        "multi_seed_test_sharpe_mean": float(seed_df["sharpe"].mean()),
        "multi_seed_test_sharpe_std": float(seed_df["sharpe"].std()),
    }
    with open(RESULTS / "ift_final_report.json", "w") as f:
        json.dump(report, f, indent=2)

    torch.save(op_final.state_dict(), RESULTS / "ren_ift_trained_operator.pt")
    print("\nSaved:", RESULTS / "ift_final_report.json")


if __name__ == "__main__":
    main()
