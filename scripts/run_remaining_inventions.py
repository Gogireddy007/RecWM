"""Runs and saves the real, measured results for Scenario Branching,
Composition Algebra, Counterfactual Engine, Crowding Radar, and
Execution Camouflage -- consolidated into results/*.json for the
final report."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.equilibrium_engine import RENOperator, anderson_acceleration, Z_DIM
from ren.scenario_branching import run_scenario_branching
from ren.composition_algebra import evaluate_composition_algebra
from ren.counterfactual import run_intervention
from ren.signals_history import HistoryCache
from ren.adversarial_defense import crowding_predicts_bad_outcomes, run_camouflage_study

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    ds = build_snapshot_dataset(panel, tickers)
    op = RENOperator(seed=42, damping=0.35, target_norm=0.6)

    # Scenario branching + composition algebra, averaged over 10 real dates
    rng_idxs = np.linspace(len(ds.dates) - 300, len(ds.dates) - 1, 10).astype(int)
    all_comp_errs, all_comp_speedups = [], []
    scenario_summ = []
    for idx in rng_idxs:
        x_base = ds.X[idx]
        z0 = torch.zeros(Z_DIM)
        base_res = anderson_acceleration(op, z0, x_base, max_iter=200, tol=1e-6)
        scenarios = run_scenario_branching(op, x_base, tickers, z0=z0)
        for s in scenarios:
            scenario_summ.append({"date": str(ds.dates[idx].date()), "scenario": s.name,
                                    "plausibility": s.plausibility, "impact": s.portfolio_impact,
                                    "converged": s.converged})
        comp = evaluate_composition_algebra(op, x_base, base_res.z_star, scenarios)
        all_comp_errs.extend([c.approx_error_relative for c in comp])
        all_comp_speedups.extend([c.speedup for c in comp])

    # Counterfactual, averaged over same dates, all 5 agent types
    cf_summ = []
    for idx in rng_idxs:
        x_base = ds.X[idx]
        base_res = anderson_acceleration(op, torch.zeros(Z_DIM), x_base, max_iter=200, tol=1e-6)
        for ai in range(5):
            cf = run_intervention(op, x_base, base_res.z_star, agent_type_index=ai,
                                    intervention_sigma=3.0, real_fwd_return=ds.fwd_ret_1d[idx].numpy(),
                                    date=ds.dates[idx])
            cf_summ.append({"date": str(ds.dates[idx].date()), "agent_type": cf.intervened_agent_type,
                              "factual_corr": cf.factual_direction_agreement,
                              "cf_corr": cf.counterfactual_direction_agreement})

    # Crowding radar
    hist = HistoryCache(panel, tickers)
    crowd_res = crowding_predicts_bad_outcomes(hist, panel, tickers)

    # Execution camouflage
    cam = run_camouflage_study(n_trials=300)
    twap_det = float(cam.loc[cam.schedule == "TWAP", "detectability"].iloc[0])
    rand_det = cam.loc[cam.schedule != "TWAP", "detectability"]

    report = {
        "scenario_branching": {
            "n_dates_tested": len(rng_idxs),
            "all_converged": all(s["converged"] for s in scenario_summ),
            "sample": scenario_summ[:15],
        },
        "composition_algebra": {
            "mean_relative_error": float(np.mean(all_comp_errs)),
            "median_relative_error": float(np.median(all_comp_errs)),
            "max_relative_error": float(np.max(all_comp_errs)),
            "mean_speedup_x": float(np.mean(all_comp_speedups)),
            "n_pairs_tested": len(all_comp_errs),
        },
        "counterfactual_engine": {
            "n_interventions_tested": len(cf_summ),
            "mean_factual_corr": float(np.mean([c["factual_corr"] for c in cf_summ])),
            "mean_cf_corr": float(np.mean([c["cf_corr"] for c in cf_summ])),
            "sample": cf_summ[:10],
        },
        "crowding_radar": {
            "spearman_corr_crowding_vs_fwd_sharpe": crowd_res.spearman_corr,
            "spearman_pvalue": crowd_res.spearman_pvalue,
            "quantile_table": crowd_res.quantile_returns.reset_index().to_dict(orient="records"),
        },
        "execution_camouflage": {
            "twap_detectability": twap_det,
            "randomized_detectability_mean": float(rand_det.mean()),
            "randomized_detectability_p90": float(rand_det.quantile(0.9)),
            "reduction_factor": float(twap_det / rand_det.mean()),
            "n_trials": len(rand_det),
        },
    }
    with open(RESULTS / "remaining_inventions_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "sample" not in kk and "table" not in kk}
                       for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
