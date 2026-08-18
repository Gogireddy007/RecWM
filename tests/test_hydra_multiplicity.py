"""
Two real, separate tests of HYDRA (Invention 3's multiplicity search):

1. Sanity check against a KNOWN bistable ground truth (a double-well
   fixed-point map with two provably distinct stable equilibria) --
   confirms the multi-restart search mechanism itself is correct.

2. An empirical, honestly-reported test of whether RENOperator (the
   actual REN architecture, randomly seeded, no fabricated bias toward
   multiplicity) exhibits multiple equilibria under the configurations
   tried. If it does not, that is reported as a real negative result,
   not hidden or worked around.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from ren.equilibrium_engine import RENOperator, X_DIM, search_multiple_equilibria, anderson_acceleration


class DoubleWellOperator(nn.Module):
    """Ground-truth bistable map (independent of RENOperator): a damped
    fixed-point iteration on the double-well potential U(z) = (z^2-1)^2,
    whose stationary points are z=-1, z=0 (unstable), z=+1 -- two known,
    provably distinct stable equilibria. Used only to validate that the
    HYDRA search mechanism can find genuine multiplicity when it exists."""

    def forward(self, z, x=None):
        grad = 4 * z * (z ** 2 - 1)
        return z - 0.05 * grad


def test_hydra_finds_known_bistability():
    op = DoubleWellOperator()
    roots = []
    for z0_val in [-2.0, -0.5, 0.5, 2.0, -1.5, 1.5]:
        z0 = torch.tensor([z0_val] + [0.0] * 31)
        res = anderson_acceleration(op, z0, torch.zeros(X_DIM), max_iter=500, tol=1e-6)
        if res.converged:
            found = res.z_star[0].item()
            if all(abs(found - r) > 0.1 for r in roots):
                roots.append(found)
    print("DoubleWellOperator: distinct roots found on first coordinate:", sorted(roots))
    assert len(roots) == 2
    assert any(abs(r - 1.0) < 0.05 for r in roots)
    assert any(abs(r + 1.0) < 0.05 for r in roots)


def test_ren_operator_multiplicity_empirical():
    """Honest empirical scan: does RENOperator show multiplicity across a
    range of gain/damping configs? Report whatever is actually found."""
    results = {}
    for target_norm in [0.6, 1.0, 1.6, 2.0]:
        for damping in [0.35, 0.7, 0.95]:
            op = RENOperator(seed=42, damping=damping, target_norm=target_norm)
            counts = []
            for trial in range(4):
                torch.manual_seed(trial)
                x = torch.randn(X_DIM) * 0.5
                roots = search_multiple_equilibria(op, x, n_restarts=16, tol=1e-5)
                counts.append(len(roots))
            results[(target_norm, damping)] = counts
            print(f"target_norm={target_norm} damping={damping}: equilibria per trial = {counts}")
    max_found = max(max(v) for v in results.values())
    print(f"\nMax distinct equilibria found across all RENOperator configs tested: {max_found}")
    return results


if __name__ == "__main__":
    test_hydra_finds_known_bistability()
    test_ren_operator_multiplicity_empirical()
    print("hydra tests complete")
