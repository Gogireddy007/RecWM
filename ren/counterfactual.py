"""
Invention 8 -- Counterfactual Engine.

Pearl's causal hierarchy: Rung 1 (association), Rung 2 (intervention,
do(X)), Rung 3 (counterfactual, given what actually happened, what
would have happened had X been different).

Rung 2 here: on a REAL historical date, take the real observed
snapshot x_t, do(agent_type i's belief = v) for a specific
counterfactual value v (holding every other real, observed input
fixed), and re-solve the equilibrium. This is a genuine intervention
on one causal input, computed with the real Equilibrium Engine.

Rung 3 here: condition on the REAL realized forward return on that
date (what actually happened), then ask what the model's *own*
counterfactual equilibrium says would have happened under the
intervention, and report the gap between the counterfactual
equilibrium's implied direction and the real realized outcome -- an
honest, directly falsifiable comparison, not a fabricated "the
model would have been right" narrative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ren.equilibrium_engine import anderson_acceleration, N_ASSETS
from ren.belief_fields import AGENT_TYPES


@dataclass
class CounterfactualResult:
    date: object
    intervened_agent_type: str
    intervention_value: float
    z_factual: torch.Tensor
    z_counterfactual: torch.Tensor
    position_delta: np.ndarray
    real_forward_return: np.ndarray
    factual_direction_agreement: float        # corr(position_factual, real fwd return)
    counterfactual_direction_agreement: float  # corr(position_cf, real fwd return)


def run_intervention(op, x_factual: torch.Tensor, z_warm: torch.Tensor,
                       agent_type_index: int, intervention_sigma: float,
                       real_fwd_return: np.ndarray, date,
                       max_iter=200, tol=1e-6) -> CounterfactualResult:
    """agent_type_index indexes into the 5 agent types; the intervention
    shifts that type's contribution to agg_belief uniformly across
    assets by `intervention_sigma` standard deviations. This is a
    simplification of "do(belief_i = v)" that is exact for the linear
    consensus construction used in snapshot_pipeline.py (agg_belief is
    a mean over K*-weighted agent-type rows), so the intervention has a
    well-defined, exact effect on x, not an approximation."""
    types = list(AGENT_TYPES.keys())
    n_types = len(types)

    res_factual = anderson_acceleration(op, z_warm, x_factual, max_iter=max_iter, tol=tol)
    z_factual = res_factual.z_star

    x_cf = x_factual.clone()
    # uniform per-asset shift representing "this agent type's belief was
    # instead `intervention_sigma` std devs" -- applied through the same
    # 1/n_types averaging weight the real consensus construction uses.
    shift = torch.full((N_ASSETS,), intervention_sigma / n_types, dtype=torch.float32)
    x_cf[:N_ASSETS] = x_cf[:N_ASSETS] + shift

    res_cf = anderson_acceleration(op, z_factual, x_cf, max_iter=max_iter, tol=tol)
    z_cf = res_cf.z_star

    pos_factual = z_factual[N_ASSETS:].detach().numpy()
    pos_cf = z_cf[N_ASSETS:].detach().numpy()
    delta = pos_cf - pos_factual

    def safe_corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return CounterfactualResult(
        date=date, intervened_agent_type=types[agent_type_index],
        intervention_value=intervention_sigma,
        z_factual=z_factual, z_counterfactual=z_cf, position_delta=delta,
        real_forward_return=real_fwd_return,
        factual_direction_agreement=safe_corr(pos_factual, real_fwd_return),
        counterfactual_direction_agreement=safe_corr(pos_cf, real_fwd_return),
    )
