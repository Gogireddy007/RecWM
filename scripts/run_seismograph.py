import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.equilibrium_engine import RENOperator
from ren.regime_transition import compute_rho_trajectory, label_stress_windows, lead_lag_correlation, KNOWN_STRESS_WINDOWS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    ds = build_snapshot_dataset(panel, tickers)

    import sys as _sys
    config = _sys.argv[1] if len(_sys.argv) > 1 else "contractive"
    if config == "contractive":
        op = RENOperator(seed=42, damping=0.35, target_norm=0.6)
        max_iter, tol = 100, 1e-6
    else:
        op = RENOperator(seed=42, damping=0.6, target_norm=1.8)
        max_iter, tol = 300, 1e-5
    op.eval()

    print(f"[{config}] Computing rho(D Phi) for all {ds.X.shape[0]} real snapshots...")
    t0 = time.time()
    rho_df = compute_rho_trajectory(op, ds.X, ds.dates, max_iter=max_iter, tol=tol)
    print(f"done in {time.time()-t0:.1f}s")

    rho_df.to_csv(RESULTS / f"rho_trajectory_{config}.csv")

    stress = label_stress_windows(rho_df.index)
    # real, objective structural-break proxy: forward 21d realized vol z-score of SPY
    spy_close = panel[("SPY", "Close")]
    spy_ret = spy_close.pct_change()
    fwd_vol = spy_ret.rolling(21).std().shift(-21).reindex(rho_df.index)
    fwd_vol_z = (fwd_vol - fwd_vol.expanding().mean()) / (fwd_vol.expanding().std() + 1e-9)

    corr_known = lead_lag_correlation(rho_df["rho"], stress.astype(float), max_lag=20)
    corr_vol = lead_lag_correlation(rho_df["rho"], fwd_vol_z, max_lag=20)

    # rho level inside vs outside known stress windows -- real, direct comparison
    rho_in = rho_df.loc[stress == 1, "rho"]
    rho_out = rho_df.loc[stress == 0, "rho"]

    from scipy import stats
    tstat, pval = stats.ttest_ind(rho_in, rho_out, equal_var=False)

    report = {
        "n_snapshots": int(len(rho_df)),
        "rho_stats_overall": {"mean": float(rho_df["rho"].mean()), "std": float(rho_df["rho"].std()),
                                "min": float(rho_df["rho"].min()), "max": float(rho_df["rho"].max())},
        "rho_mean_inside_known_stress_windows": float(rho_in.mean()),
        "rho_mean_outside_known_stress_windows": float(rho_out.mean()),
        "welch_ttest_stat": float(tstat), "welch_ttest_pvalue": float(pval),
        "known_stress_windows": KNOWN_STRESS_WINDOWS,
        "best_lag_vs_known_stress": corr_known.loc[corr_known["pearson_r"].abs().idxmax()].to_dict() if len(corr_known) else None,
        "best_lag_vs_fwd_realized_vol": corr_vol.loc[corr_vol["pearson_r"].abs().idxmax()].to_dict() if len(corr_vol) else None,
        "corr_vs_known_stress_by_lag": corr_known.to_dict(orient="records"),
        "corr_vs_fwd_vol_by_lag": corr_vol.to_dict(orient="records"),
    }
    report["config"] = config
    with open(RESULTS / f"seismograph_report_{config}.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if "by_lag" not in k}, indent=2, default=str))


if __name__ == "__main__":
    main()
