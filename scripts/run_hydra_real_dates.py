"""
Re-runs HYDRA's multi-restart multiplicity search on ACTUAL real market
snapshots (ds.X at real historical dates), not synthetic random test
vectors -- closing the precision gap flagged in the review: the
original 768-search sweep used torch.randn(X_DIM)*0.5 vectors matching
real data's scale, not real dates themselves.

Two pools of real dates:
  1. Every 10th real snapshot date across the full 2016-2026 history
     (broad coverage, ~262 dates)
  2. EVERY real date inside the 6 known historical stress windows
     (COVID crash, 2022 bear market, SVB, etc.) at full resolution --
     if REN ever produces multiple equilibria anywhere, a real crisis
     is the most plausible place per the thesis's own framing.

Tested across 3 representative operator configs already used elsewhere
in this project: contractive (target_norm=0.6), sensitive
(target_norm=1.8), and sensitive+leaky_relu.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.equilibrium_engine import RENOperator, search_multiple_equilibria
from ren.regime_transition import KNOWN_STRESS_WINDOWS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    ds = build_snapshot_dataset(panel, tickers)
    n = ds.X.shape[0]

    broad_idxs = list(range(0, n, 10))
    stress_idxs = []
    for start, end, name in KNOWN_STRESS_WINDOWS:
        mask = (ds.dates >= pd.Timestamp(start)) & (ds.dates <= pd.Timestamp(end))
        stress_idxs.extend(np.where(mask)[0].tolist())
    stress_idxs = sorted(set(stress_idxs))
    all_idxs = sorted(set(broad_idxs) | set(stress_idxs))

    print(f"Broad-coverage dates: {len(broad_idxs)}, stress-window dates: {len(stress_idxs)}, "
          f"union tested: {len(all_idxs)}")

    configs = [
        {"name": "contractive", "damping": 0.35, "target_norm": 0.6, "hidden_activation": "tanh"},
        {"name": "sensitive", "damping": 0.6, "target_norm": 1.8, "hidden_activation": "tanh"},
        {"name": "sensitive_leaky_relu", "damping": 0.6, "target_norm": 1.8, "hidden_activation": "leaky_relu"},
    ]

    report = {}
    for cfg in configs:
        name = cfg.pop("name")
        op = RENOperator(seed=42, **cfg)
        op.eval()

        t0 = time.time()
        max_found = 0
        multi_dates = []
        counts = []
        for idx in all_idxs:
            x = ds.X[idx]
            roots = search_multiple_equilibria(op, x, n_restarts=16, tol=1e-5)
            counts.append(len(roots))
            if len(roots) > max_found:
                max_found = len(roots)
            if len(roots) > 1:
                multi_dates.append(str(ds.dates[idx].date()))
        dt = time.time() - t0

        n_stress_tested = sum(1 for idx in all_idxs if idx in stress_idxs)
        stress_counts = [c for idx, c in zip(all_idxs, counts) if idx in stress_idxs]

        result = {
            "n_dates_tested": len(all_idxs),
            "n_stress_dates_tested": n_stress_tested,
            "max_equilibria_found_anywhere": max_found,
            "dates_with_multiplicity": multi_dates,
            "fraction_dates_with_multiplicity": len(multi_dates) / len(all_idxs),
            "mean_equilibria_count": float(np.mean(counts)),
            "mean_equilibria_count_in_stress_windows": float(np.mean(stress_counts)) if stress_counts else None,
            "time_s": dt,
            "n_searches_run": len(all_idxs) * 16,
        }
        report[name] = result
        print(f"[{name}] tested {len(all_idxs)} real dates ({n_stress_tested} in known stress windows), "
              f"{len(all_idxs)*16} total restarts, {dt:.1f}s: "
              f"max_equilibria_found={max_found}, dates_with_multiplicity={len(multi_dates)}")

    with open(RESULTS / "hydra_real_dates_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved:", RESULTS / "hydra_real_dates_report.json")


if __name__ == "__main__":
    main()
