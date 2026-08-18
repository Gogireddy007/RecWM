import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.seismograph_training import compute_real_stress_targets, train_seismograph_supervised
from ren.regime_transition import compute_rho_trajectory, label_stress_windows, lead_lag_correlation, KNOWN_STRESS_WINDOWS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    ds = build_snapshot_dataset(panel, tickers)
    n = ds.X.shape[0]
    split = int(n * 0.8)
    X_train, X_test = ds.X[:split], ds.X[split:]
    dates_train, dates_test = ds.dates[:split], ds.dates[split:]

    target_rho_all = compute_real_stress_targets(panel, ds.dates)
    target_rho_train = torch.tensor(target_rho_all.iloc[:split].to_numpy(), dtype=torch.float32)

    print(f"Training operator with rho directly supervised toward real forward SPY stress, "
          f"TRAIN period only ({dates_train[0].date()}..{dates_train[-1].date()}, n={split})...")
    t0 = time.time()
    op, hist = train_seismograph_supervised(X_train, target_rho_train, epochs=15, samples_per_epoch=150,
                                               lr=1e-3, seed=42)
    print(f"trained in {time.time()-t0:.1f}s, loss[0]={hist[0]:.5f} -> loss[-1]={hist[-1]:.5f}")

    print(f"\nEvaluating TRUE rho(D Phi) (standard SEISMOGRAPH definition) on the full real history "
          f"with the TRAINED operator...")
    t0 = time.time()
    rho_df = compute_rho_trajectory(op, ds.X, ds.dates, max_iter=150, tol=1e-5)
    print(f"done in {time.time()-t0:.1f}s, convergence rate={rho_df['converged'].mean():.3f}")

    rho_train_seg = rho_df.iloc[:split]
    rho_test_seg = rho_df.iloc[split:]

    stress_all = label_stress_windows(rho_df.index)
    stress_train = stress_all.iloc[:split]
    stress_test = stress_all.iloc[split:]

    def stress_stats(rho_seg, stress_seg, label):
        rho_in = rho_seg.loc[stress_seg.to_numpy() == 1, "rho"]
        rho_out = rho_seg.loc[stress_seg.to_numpy() == 0, "rho"]
        if len(rho_in) < 3 or len(rho_out) < 3:
            return {"label": label, "note": "insufficient stress-window samples in this segment"}
        t, p = stats.ttest_ind(rho_in, rho_out, equal_var=False)
        return {"label": label, "rho_mean_in_stress": float(rho_in.mean()),
                 "rho_mean_out_stress": float(rho_out.mean()), "t_stat": float(t), "p_value": float(p),
                 "direction": "CORRECT (higher in stress)" if rho_in.mean() > rho_out.mean() else "WRONG (lower in stress)"}

    train_result = stress_stats(rho_train_seg, stress_train, "TRAIN (in-sample)")
    test_result = stress_stats(rho_test_seg, stress_test, "TEST (held-out, never used for training)")

    spy_close = panel[("SPY", "Close")]
    fwd_vol = spy_close.pct_change().rolling(21).std().shift(-21)
    fwd_vol_z_test = ((fwd_vol - fwd_vol.expanding().mean()) / (fwd_vol.expanding().std() + 1e-9)).reindex(dates_test)
    corr_test = lead_lag_correlation(rho_test_seg["rho"], fwd_vol_z_test, max_lag=20)
    best_lag_test = corr_test.loc[corr_test["pearson_r"].abs().idxmax()].to_dict() if len(corr_test) else None

    print("\n=== TRAIN (in-sample) ===")
    print(json.dumps(train_result, indent=2))
    print("\n=== TEST (held-out, the real test of the hypothesis) ===")
    print(json.dumps(test_result, indent=2))
    print("\nBest lag correlation vs real forward SPY vol, TEST period:", best_lag_test)

    report = {
        "training_loss_curve": hist,
        "train_result": train_result,
        "test_result": test_result,
        "best_lag_corr_vs_fwd_vol_test": best_lag_test,
        "convergence_rate": float(rho_df["converged"].mean()),
        "known_stress_windows": KNOWN_STRESS_WINDOWS,
    }
    with open(RESULTS / "seismograph_training_experiment.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    rho_df.to_csv(RESULTS / "rho_trajectory_seismograph_trained.csv")
    torch.save(op.state_dict(), RESULTS / "ren_seismograph_trained_operator.pt")
    print("\nSaved:", RESULTS / "seismograph_training_experiment.json")


if __name__ == "__main__":
    main()
