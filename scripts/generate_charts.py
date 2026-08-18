"""
Generates every chart for the investor PDF, from real, already-saved
result data (regenerating a few raw daily-pnl series from saved model
checkpoints where only summary stats were persisted). Uses a fixed,
colorblind-validated categorical palette (dataviz skill reference
palette, light-mode hex values).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ren.snapshot_pipeline import build_snapshot_dataset
from ren.equilibrium_engine import RENOperator
from ren.live_system import backtest_ren, backtest_equal_weight
from ren.baselines import backtest_inverse_vol

RESULTS = Path(__file__).resolve().parent.parent / "results"
CHARTS = RESULTS / "charts"
CHARTS.mkdir(exist_ok=True)

# ---- palette (dataviz skill reference palette, light mode) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dc"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def save(fig, name):
    fig.savefig(CHARTS / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------------
# 1. Sharpe comparison bar chart with bootstrap CI
# ---------------------------------------------------------------------
def chart_sharpe_comparison():
    with open(RESULTS / "comparison_and_pressure_tests.json") as f:
        data = json.load(f)
    table = {r["model"]: r for r in data["comparison_table"]}
    ci = data["bootstrap_sharpe_ci"]

    order = ["Equal_weight_buyhold", "Risk_parity_inverse_vol", "Random_forest_ML", "60_40",
             "REN_composite_BEST_seed", "REN_ift_trained", "TS_momentum_12m", "REN_untrained",
             "Ridge_ML", "Mean_reversion_5d"]
    labels = {"Equal_weight_buyhold": "Buy & Hold", "Risk_parity_inverse_vol": "Risk Parity",
              "Random_forest_ML": "Random Forest (ML)", "60_40": "60/40",
              "REN_composite_BEST_seed": "REN (best trained seed)", "REN_ift_trained": "REN (P&L-trained)",
              "TS_momentum_12m": "Momentum (12m)", "REN_untrained": "REN (untrained)",
              "Ridge_ML": "Ridge Regression (ML)", "Mean_reversion_5d": "Mean Reversion (5d)"}

    sharpes = [table[m]["sharpe"] for m in order]
    colors = [VIOLET if "REN" in m else BLUE for m in order]
    err_lo, err_hi = [], []
    for m in order:
        if m in ci:
            err_lo.append(max(0, ci[m]["point_estimate"] - ci[m]["ci_5"]))
            err_hi.append(max(0, ci[m]["ci_95"] - ci[m]["point_estimate"]))
        else:
            err_lo.append(0); err_hi.append(0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(order))[::-1]
    ax.barh(y, sharpes, color=colors, height=0.6, zorder=3)
    ax.errorbar(sharpes, y, xerr=[err_lo, err_hi], fmt="none", ecolor=TEXT_PRIMARY,
                elinewidth=1.3, capsize=3, zorder=4)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in order])
    ax.set_xlabel("Sharpe ratio (real held-out test period, 2024-07 to 2026-08, 90% bootstrap CI shown where computed)")
    ax.set_title("REN vs. standard market models — held-out Sharpe ratio", fontsize=13, fontweight="bold", loc="left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=VIOLET, label="REN variant"), Patch(color=BLUE, label="Standard market model")],
               loc="lower right", frameon=False)
    save(fig, "01_sharpe_comparison.png")


# ---------------------------------------------------------------------
# 2. Equity curves
# ---------------------------------------------------------------------
def chart_equity_curves():
    panel = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet")
    tickers = sorted({c[0] for c in panel.columns})
    ds = build_snapshot_dataset(panel, tickers)
    n = ds.X.shape[0]; split = int(n * 0.8)
    X_test, fwd_test = ds.X[split:], ds.fwd_ret_1d[split:]
    dates_test = ds.dates[split:]

    op_untrained = RENOperator(seed=42, damping=0.35, target_norm=0.6); op_untrained.eval()
    res_untrained = backtest_ren(op_untrained, X_test, fwd_test, dates_test)

    op_best = RENOperator(seed=42, damping=0.35, target_norm=0.6)
    op_best.load_state_dict(torch.load(RESULTS / "ren_composite_trained_seed1.pt")); op_best.eval()
    res_best = backtest_ren(op_best, X_test, fwd_test, dates_test)

    res_eqw = backtest_equal_weight(fwd_test, dates_test)
    res_rp = backtest_inverse_vol(panel, tickers, dates_test, fwd_test)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    series = [
        ("Buy & Hold", res_eqw.daily_pnl, BLUE),
        ("Risk Parity", res_rp.daily_pnl, AQUA),
        ("REN (best trained seed)", res_best.daily_pnl, VIOLET),
        ("REN (untrained)", res_untrained.daily_pnl, MAGENTA),
    ]
    for label, pnl, color in series:
        cum = np.cumsum(pnl) * 100
        ax.plot(dates_test, cum, label=label, color=color, linewidth=1.8)

    ax.set_title("Cumulative return — real held-out test period", fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("Cumulative return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, "02_equity_curves.png")


# ---------------------------------------------------------------------
# 3. Regime-conditional breakdown
# ---------------------------------------------------------------------
def chart_regime_breakdown():
    with open(RESULTS / "comparison_and_pressure_tests.json") as f:
        data = json.load(f)
    rb = data["regime_breakdown"]
    models = ["Equal_weight_buyhold", "Risk_parity_inverse_vol", "REN_composite_BEST_seed", "REN_untrained"]
    labels = {"Equal_weight_buyhold": "Buy & Hold", "Risk_parity_inverse_vol": "Risk Parity",
              "REN_composite_BEST_seed": "REN (trained)", "REN_untrained": "REN (untrained)"}
    colors = {"Equal_weight_buyhold": BLUE, "Risk_parity_inverse_vol": AQUA,
              "REN_composite_BEST_seed": VIOLET, "REN_untrained": MAGENTA}
    regimes = ["calm", "normal", "turbulent"]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    x = np.arange(len(regimes))
    for i, m in enumerate(models):
        vals = [next(r["sharpe"] for r in rb[m] if r["regime"] == reg) for reg in regimes]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=labels[m], color=colors[m], zorder=3)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Calm", "Normal", "Turbulent"])
    ax.set_ylabel("Sharpe ratio within regime")
    ax.set_title("Performance by real market-volatility regime", fontsize=13, fontweight="bold", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)
    save(fig, "03_regime_breakdown.png")


# ---------------------------------------------------------------------
# 4. SEISMOGRAPH: rho trajectory around COVID crash, untrained vs trained
# ---------------------------------------------------------------------
def chart_seismograph():
    untrained = pd.read_csv(RESULTS / "rho_trajectory_sensitive.csv", index_col=0, parse_dates=True)
    trained = pd.read_csv(RESULTS / "rho_trajectory_seismograph_trained.csv", index_col=0, parse_dates=True)

    start, end = "2020-01-15", "2020-04-15"
    u = untrained.loc[start:end]
    t = trained.loc[start:end]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-23"), color=RED, alpha=0.12,
               label="Real COVID-19 crash window")
    ax.plot(u.index, u["rho"], label="Untrained REN (wrong direction)", color=MAGENTA, linewidth=1.8)
    ax.plot(t.index, t["rho"], label="Crisis-trained REN (correct direction)", color=GREEN, linewidth=1.8)
    ax.set_ylabel("ρ(DΦ) — spectral radius")
    ax.set_title("SEISMOGRAPH signal around the real COVID-19 crash", fontsize=13, fontweight="bold", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.legend(loc="upper left", frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, "04_seismograph_covid.png")


# ---------------------------------------------------------------------
# 5. Multi-seed Sharpe distribution (robustness)
# ---------------------------------------------------------------------
def chart_seed_distribution():
    untrained_seeds = pd.read_csv(RESULTS / "untrained_seed_sweep.csv")["sharpe"].to_numpy()
    with open(RESULTS / "composite_training_report.json") as f:
        comp = json.load(f)
    composite_seeds = np.array([r["test"]["sharpe"] for r in comp["seed_results"]])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    rng = np.random.default_rng(0)
    y0 = np.full(len(untrained_seeds), 1.0) + rng.uniform(-0.08, 0.08, len(untrained_seeds))
    y1 = np.full(len(composite_seeds), 0.0) + rng.uniform(-0.08, 0.08, len(composite_seeds))
    ax.scatter(untrained_seeds, y0, color=MAGENTA, s=55, zorder=3, label="Untrained (20 seeds)")
    ax.scatter(composite_seeds, y1, color=VIOLET, s=55, zorder=3, label="Composite-trained (5 seeds)")
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1, zorder=1)
    ax.axvline(1.61, color=BLUE, linewidth=1.5, linestyle="--", zorder=1, label="Buy & Hold Sharpe (1.61)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Composite-\ntrained", "Untrained"])
    ax.set_xlabel("Held-out test Sharpe ratio, per random seed")
    ax.set_title("Seed-to-seed variability — no configuration is consistently profitable",
                  fontsize=13, fontweight="bold", loc="left")
    ax.set_ylim(-0.5, 1.5)
    ax.legend(loc="upper left", frameon=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", visible=False)
    save(fig, "05_seed_distribution.png")


if __name__ == "__main__":
    chart_sharpe_comparison()
    chart_equity_curves()
    chart_regime_breakdown()
    chart_seismograph()
    chart_seed_distribution()
    print("all charts generated")
