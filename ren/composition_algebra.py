"""
Invention 7 -- Composition Algebra.

Thesis definition: an operator (x) over typed belief fragments F_alpha,
F_beta, so complex multi-leg reasoning is built from verified simpler
components instead of re-derived from scratch.

Concrete, testable implementation: a belief "fragment" is a single
named perturbation of the belief field (exactly the Scenario objects
from scenario_branching.py -- e.g. F_rate_shock, F_risk_off). The
composition operator (x) combines two fragments' perturbation vectors
linearly (this is the natural, well-defined choice for perturbations
of an additive input slot: composing "apply perturbation A" and
"apply perturbation B" should mean "apply A+B") and predicts the
JOINT equilibrium by RE-USING each fragment's already-computed
equilibrium delta, rather than re-solving:

    z_hat(F_a (x) F_b) = z_base + (z*_a - z_base) + (z*_b - z_base)

This is a first-order (linear superposition) approximation. Because
Phi is nonlinear, it will NOT be exact -- REN is a nonlinear fixed-
point system, and the whole point of measuring this is to find out
how good linear composition actually is. The ground truth (re-solving
the joint perturbation directly, i.e. NOT reusing components) is
computed independently and the approximation error is reported
honestly, along with the real wall-clock time saved by composing
instead of re-solving.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ren.equilibrium_engine import anderson_acceleration, N_ASSETS


@dataclass
class CompositionResult:
    pair_name: str
    z_composed: torch.Tensor
    z_ground_truth: torch.Tensor
    approx_error_l2: float
    approx_error_relative: float
    time_compose_s: float
    time_ground_truth_s: float
    speedup: float


def compose_fragments(op, x_base: torch.Tensor, z_base: torch.Tensor,
                        scenario_a, scenario_b, max_iter=200, tol=1e-6,
                        timing_reps=2000) -> CompositionResult:
    """timing_reps > 1: a single perf_counter() call around a ~1-2
    microsecond vector-add is dominated by Python/timer-call overhead and
    garbage-collection jitter, not the operation itself -- verified
    directly: single-call timings on real data varied 42x-1301x pair to
    pair for what is mechanically the identical two-vector-add operation
    every time. Repeating the op `timing_reps` times and dividing gives a
    stable, reproducible per-call estimate instead."""
    # each fragment's delta was already computed when the scenario was
    # solved (scenario_a.z_star, scenario_b.z_star) -- composition reuses
    # those, no new equilibrium solve required.
    delta_a = scenario_a.z_star - z_base
    delta_b = scenario_b.z_star - z_base
    t0 = time.perf_counter()
    for _ in range(timing_reps):
        z_composed = z_base + delta_a + delta_b
    time_compose = (time.perf_counter() - t0) / timing_reps

    x_joint = x_base.clone()
    x_joint[:N_ASSETS] = x_joint[:N_ASSETS] + torch.tensor(
        scenario_a.perturbation + scenario_b.perturbation, dtype=torch.float32)
    gt_reps = max(1, timing_reps // 100)  # solve is ~1000x more expensive per-call; fewer reps needed
    t0 = time.perf_counter()
    for _ in range(gt_reps):
        res = anderson_acceleration(op, z_base, x_joint, max_iter=max_iter, tol=tol)
    time_gt = (time.perf_counter() - t0) / gt_reps

    err = float(torch.norm(z_composed - res.z_star).item())
    rel = err / (float(torch.norm(res.z_star).item()) + 1e-9)

    return CompositionResult(
        pair_name=f"{scenario_a.name}+{scenario_b.name}",
        z_composed=z_composed, z_ground_truth=res.z_star,
        approx_error_l2=err, approx_error_relative=rel,
        time_compose_s=time_compose, time_ground_truth_s=time_gt,
        speedup=time_gt / max(time_compose, 1e-9),
    )


def evaluate_composition_algebra(op, x_base, z_base, scenarios) -> list[CompositionResult]:
    results = []
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            results.append(compose_fragments(op, x_base, z_base, scenarios[i], scenarios[j]))
    return results
