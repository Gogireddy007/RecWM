import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ren.belief_fields import AGENT_TYPES, TIMESCALES
from ren.influence_kernel import compute_influence_kernel, verify_resolvent_convergence

DATA = Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet"


def build_signal_history(panel, tickers, timescale_name, window, lookback=600):
    """Real time series: for each day in the lookback window, compute each
    agent type's mean-across-assets signal. Returns (T, n_types) matrix."""
    close_cols = {t: (t, "Close") for t in tickers}
    n = len(panel)
    start = max(window + 260, n - lookback)
    rows = []
    idx = []
    for end in range(start, n):
        sub = panel.iloc[: end + 1]
        vec = []
        for theta, fn in AGENT_TYPES.items():
            sig = fn(sub, tickers, window)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            vec.append(np.mean(sig))
        rows.append(vec)
        idx.append(panel.index[end])
    return np.array(rows), idx


def test_influence_kernel_on_real_data():
    panel = pd.read_parquet(DATA)
    tickers = sorted({c[0] for c in panel.columns})
    hist, idx = build_signal_history(panel, tickers, "medium", TIMESCALES["medium"], lookback=500)
    labels = list(AGENT_TYPES.keys())
    result = compute_influence_kernel(hist, labels)

    assert result.A.shape == (5, 5)
    assert result.spectral_radius_gammaA < 1.0 + 1e-6
    assert np.all(np.isfinite(result.K_star))

    conv = verify_resolvent_convergence(result, max_terms=100)
    # error must decrease geometrically at rate ~= rho(gamma*A) (measured, not asserted a priori)
    assert conv["frobenius_error"].iloc[-1] < conv["frobenius_error"].iloc[0]
    assert conv["frobenius_error"].iloc[-1] < 1e-3  # true at 100 terms given rho(gamma*A)=0.9
    log_err = np.log(conv["frobenius_error"].to_numpy())
    empirical_rate = np.polyfit(conv["n_terms"].to_numpy(), log_err, 1)[0]
    theoretical_rate = np.log(result.spectral_radius_gammaA)
    print(conv.iloc[[0, 9, 19, 39, 59, 99]])
    print("rho(A) =", result.spectral_radius_A, "gamma =", result.gamma,
          "rho(gamma*A) =", result.spectral_radius_gammaA)
    print("empirical log-decay rate:", empirical_rate, "theoretical log(rho):", theoretical_rate)
    assert abs(empirical_rate - theoretical_rate) < 0.05
    return result, conv


if __name__ == "__main__":
    test_influence_kernel_on_real_data()
    print("influence_kernel tests passed")
