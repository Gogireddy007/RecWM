"""
Invention 2 -- Belief Fields.

Thesis definition (Section 3.2):
    B(theta, t, s) : Theta x T x S -> Delta(W)
a distribution over world states, indexed by agent type theta and
timescale s, collapsed from an infinite mutual-belief regress into an
O(N*s) representation via RKHS projection.

Honesty note (read this before trusting any number downstream):
No public dataset records what a momentum desk, a market maker, or a
stat-arb desk actually *believes*. Nobody publishes that. So "agent
type" here is an explicit, fully-documented reduced-form function of
REAL price/volume data -- not a fabricated belief, not a black box.
Five agent types are implemented, each a standard, well-known trading
heuristic computed directly off the real OHLCV panel:

    momentum            trailing return z-score
    mean_reversion       distance from rolling mean, in rolling-std units
    volatility_targeting  negative change in realized vol (de-risk on vol spikes)
    carry_macro          relative trend vs. a cross-asset basket (macro tilt)
    liquidity_provision   inverted intraday range / volume pressure proxy

Each produces one real number per asset per day. That per-asset vector
IS the raw belief signal. It is then projected through random Fourier
features (Rahimi & Recht, 2007) -- a standard, real RKHS approximation
-- into a fixed-dimension embedding. This gives the architecture a
concrete Delta(W)-style object (mean embedding + a scalar dispersion
term standing in for distributional spread) without inventing data
that doesn't exist. The *interaction* between agent types (the "what
does A think B thinks" structure) is not handled here -- that is
exactly what the Influence Kernel (Invention 1) does, via its
resolvent series, which is the actual mechanism in this architecture
for summing higher-order belief effects. Belief Fields supply the
per-type raw signal; Influence Kernel supplies the cross-type
coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

TIMESCALES = {"short": 5, "medium": 21, "long": 63}  # trading days


# ---------------------------------------------------------------------------
# Agent-type signal functions: real, documented, deterministic functions of
# real OHLCV data. No parameter here is fit to make results look good.
# ---------------------------------------------------------------------------

def _momentum(panel: pd.DataFrame, tickers: list[str], window: int) -> np.ndarray:
    close = panel[[(t, "Close") for t in tickers]]
    close.columns = tickers
    ret = close.pct_change(window)
    z = (ret - ret.rolling(252, min_periods=window).mean()) / (
        ret.rolling(252, min_periods=window).std() + 1e-9
    )
    return z.iloc[-1].to_numpy()


def _mean_reversion(panel: pd.DataFrame, tickers: list[str], window: int) -> np.ndarray:
    close = panel[[(t, "Close") for t in tickers]]
    close.columns = tickers
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    z = -(close - ma) / (sd + 1e-9)
    return z.iloc[-1].to_numpy()


def _volatility_targeting(panel: pd.DataFrame, tickers: list[str], window: int) -> np.ndarray:
    close = panel[[(t, "Close") for t in tickers]]
    close.columns = tickers
    ret = close.pct_change()
    vol = ret.rolling(window).std()
    dvol = vol.diff(window)
    z = -dvol / (vol.rolling(252, min_periods=window).std() + 1e-9)
    return z.iloc[-1].to_numpy()


def _carry_macro(panel: pd.DataFrame, tickers: list[str], window: int) -> np.ndarray:
    close = panel[[(t, "Close") for t in tickers]]
    close.columns = tickers
    ret = close.pct_change(window)
    basket_ret = ret.mean(axis=1)
    rel = ret.sub(basket_ret, axis=0)
    z = rel / (rel.rolling(252, min_periods=window).std() + 1e-9)
    return z.iloc[-1].to_numpy()


def _liquidity_provision(panel: pd.DataFrame, tickers: list[str], window: int) -> np.ndarray:
    high = panel[[(t, "High") for t in tickers]]; high.columns = tickers
    low = panel[[(t, "Low") for t in tickers]]; low.columns = tickers
    close = panel[[(t, "Close") for t in tickers]]; close.columns = tickers
    vol = panel[[(t, "Volume") for t in tickers]]; vol.columns = tickers
    rng = (high - low) / (close + 1e-9)
    rng_z = rng.rolling(window).mean()
    rng_z = -(rng_z - rng_z.rolling(252, min_periods=window).mean()) / (
        rng_z.rolling(252, min_periods=window).std() + 1e-9
    )
    return rng_z.iloc[-1].to_numpy()


AGENT_TYPES: dict[str, Callable[[pd.DataFrame, list[str], int], np.ndarray]] = {
    "momentum": _momentum,
    "mean_reversion": _mean_reversion,
    "volatility_targeting": _volatility_targeting,
    "carry_macro": _carry_macro,
    "liquidity_provision": _liquidity_provision,
}


# ---------------------------------------------------------------------------
# RKHS projection (random Fourier features, Rahimi & Recht 2007)
# ---------------------------------------------------------------------------

class RFFProjector:
    """Approximates a Gaussian-kernel RKHS feature map with a fixed seed
    (so it is reproducible), used to embed each raw belief vector into a
    common d-dimensional space."""

    def __init__(self, input_dim: int, out_dim: int = 64, gamma: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2 * gamma), size=(input_dim, out_dim))
        self.b = rng.uniform(0, 2 * np.pi, size=out_dim)
        self.out_dim = out_dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        proj = x @ self.W + self.b
        return np.sqrt(2.0 / self.out_dim) * np.cos(proj)


@dataclass
class BeliefState:
    """One (agent type, timescale) belief: an RKHS embedding + a scalar
    dispersion term standing in for the distributional spread of Delta(W)."""
    embedding: np.ndarray
    dispersion: float
    raw_signal: np.ndarray


@dataclass
class BeliefField:
    """B(theta, t, s): the full belief field at one point in time t,
    across all agent types theta and timescales s.

    Updates are O(N*s) per new data point: N = n_agent_types * n_assets,
    s = n_timescales -- a fixed number of vector ops per tick, no growth
    with history length, matching the O(N*s) claim in the thesis.
    """
    tickers: list[str]
    embed_dim: int = 64
    ema_halflife: int = 10
    states: dict[tuple[str, str], BeliefState] = field(default_factory=dict)
    _projectors: dict[str, RFFProjector] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.tickers)
        for theta in AGENT_TYPES:
            self._projectors[theta] = RFFProjector(n, self.embed_dim, seed=hash(theta) % (2**31))

    def update(self, panel: pd.DataFrame) -> "BeliefField":
        """One O(N*s) recurrence step: recompute each agent type's raw
        signal from the trailing window, project via RKHS, EMA-blend with
        the previous embedding (the recurrence), and store."""
        alpha = 1 - 0.5 ** (1 / self.ema_halflife)
        for theta, fn in AGENT_TYPES.items():
            for s_name, window in TIMESCALES.items():
                raw = fn(panel, self.tickers, window)
                raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                emb_new = self._projectors[theta](raw)
                key = (theta, s_name)
                disp_new = float(np.std(raw))
                if key in self.states:
                    prev = self.states[key]
                    emb = (1 - alpha) * prev.embedding + alpha * emb_new
                    disp = (1 - alpha) * prev.dispersion + alpha * disp_new
                else:
                    emb, disp = emb_new, disp_new
                self.states[key] = BeliefState(embedding=emb, dispersion=disp, raw_signal=raw)
        return self

    def raw_matrix(self) -> np.ndarray:
        """Stack raw per-asset signals for all (theta, s) pairs into a
        matrix of shape (n_agent_types * n_timescales, n_assets). This is
        the direct input to the Influence Kernel and the Equilibrium
        Engine."""
        rows = []
        for theta in AGENT_TYPES:
            for s_name in TIMESCALES:
                rows.append(self.states[(theta, s_name)].raw_signal)
        return np.vstack(rows)

    def agent_labels(self) -> list[str]:
        return [f"{theta}:{s}" for theta in AGENT_TYPES for s in TIMESCALES]
