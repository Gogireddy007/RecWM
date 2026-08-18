import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.equilibrium_engine import RENOperator, Z_DIM
from ren.atlas_train import compute_ground_truth_fixed_points, train_warm_start, benchmark_warm_starts

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    print("Building real snapshot dataset...")
    ds = build_snapshot_dataset(panel, tickers)
    n = ds.X.shape[0]
    split = int(n * 0.8)
    X_train, X_test = ds.X[:split], ds.X[split:]
    dates_train, dates_test = ds.dates[:split], ds.dates[split:]
    print(f"n={n}, train={split} ({dates_train[0].date()}..{dates_train[-1].date()}), "
          f"test={n-split} ({dates_test[0].date()}..{dates_test[-1].date()})")

    op = RENOperator(seed=42, damping=0.35, target_norm=0.6)
    op.eval()

    print("Computing ground-truth fixed points for train set (offline, high precision)...")
    t0 = time.time()
    Z_train, conv_train = compute_ground_truth_fixed_points(op, X_train, tol=1e-7, max_iter=500)
    print(f"  train: {conv_train.sum()}/{len(conv_train)} converged, {time.time()-t0:.1f}s")

    print("Computing ground-truth fixed points for test set...")
    t0 = time.time()
    Z_test, conv_test = compute_ground_truth_fixed_points(op, X_test, tol=1e-7, max_iter=500)
    print(f"  test: {conv_test.sum()}/{len(conv_test)} converged, {time.time()-t0:.1f}s")

    print("Training ATLAS warm-start network on TRAIN snapshots only...")
    net, loss_hist = train_warm_start(X_train, Z_train, epochs=400)
    print(f"  final train MSE: {loss_hist[-1]:.6f} (initial: {loss_hist[0]:.6f})")

    print("Benchmarking zero-init vs random-init vs trained-warm-start on HELD-OUT test snapshots...")
    bench = benchmark_warm_starts(op, net, X_test, Z_test, tol=1e-5, max_iter=300)

    def stats(a):
        return dict(mean=float(np.mean(a)), p50=float(np.percentile(a, 50)),
                    p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                    max=float(np.max(a)))

    report = {
        "n_snapshots_total": n,
        "n_train": int(split),
        "n_test": int(n - split),
        "train_date_range": [str(dates_train[0].date()), str(dates_train[-1].date())],
        "test_date_range": [str(dates_test[0].date()), str(dates_test[-1].date())],
        "ground_truth_convergence_rate_train": float(conv_train.mean()),
        "ground_truth_convergence_rate_test": float(conv_test.mean()),
        "atlas_final_train_mse": float(loss_hist[-1]),
        "atlas_warm_start_test_mse": bench.warm_start_mse,
        "fixed_point_mean_norm_test": bench.fixed_point_norm_mean,
        "iterations": {
            "zero_init": stats(bench.zero_init_iters),
            "random_init": stats(bench.random_init_iters),
            "trained_warm_start": stats(bench.trained_init_iters),
        },
        "latency_ms": {
            "zero_init": stats(bench.zero_init_ms),
            "random_init": stats(bench.random_init_ms),
            "trained_warm_start": stats(bench.trained_init_ms),
        },
        "speedup_iterations_trained_vs_zero": float(np.mean(bench.zero_init_iters) / max(np.mean(bench.trained_init_iters), 1e-9)),
        "speedup_latency_trained_vs_zero": float(np.mean(bench.zero_init_ms) / max(np.mean(bench.trained_init_ms), 1e-9)),
    }

    with open(RESULTS / "atlas_benchmark.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
