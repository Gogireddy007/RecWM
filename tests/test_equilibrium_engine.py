import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
import numpy as np

from ren.equilibrium_engine import (
    RENOperator, Z_DIM, X_DIM, naive_fixed_point_iteration,
    anderson_acceleration, spectral_radius_at_fixed_point,
    search_multiple_equilibria,
)

torch.manual_seed(0)


def random_x():
    return torch.randn(X_DIM) * 0.5


def test_naive_converges():
    op = RENOperator(seed=1, damping=0.35, target_norm=0.6)
    x = random_x()
    z0 = torch.zeros(Z_DIM)
    res = naive_fixed_point_iteration(op, z0, x, max_iter=2000, tol=1e-6)
    print("naive:", res.n_iters, "iters, converged=", res.converged, "time=", res.wall_time_s)
    assert res.converged


def test_anderson_faster_than_naive():
    op = RENOperator(seed=2, damping=0.35, target_norm=0.6)
    x = random_x()
    z0 = torch.zeros(Z_DIM)
    naive = naive_fixed_point_iteration(op, z0, x, max_iter=2000, tol=1e-6)
    anders = anderson_acceleration(op, z0, x, max_iter=200, tol=1e-6)
    print("naive iters:", naive.n_iters, "anderson iters:", anders.n_iters)
    assert anders.converged
    assert anders.n_iters <= naive.n_iters


def test_spectral_radius_finite():
    op = RENOperator(seed=3, damping=0.35, target_norm=0.6)
    x = random_x()
    z0 = torch.zeros(Z_DIM)
    res = anderson_acceleration(op, z0, x, max_iter=200, tol=1e-6)
    rho = spectral_radius_at_fixed_point(op, res.z_star, x)
    print("rho(D Phi) at z* =", rho)
    assert np.isfinite(rho)


def test_latency_benchmark():
    op = RENOperator(seed=4, damping=0.35, target_norm=0.6)
    op.eval()
    n_trials = 200
    times = []
    for i in range(n_trials):
        x = torch.randn(X_DIM) * 0.5
        z0 = torch.zeros(Z_DIM)
        t0 = time.perf_counter()
        res = anderson_acceleration(op, z0, x, max_iter=100, tol=1e-5)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"Anderson-solve latency over {n_trials} real solves (ms): "
          f"p50={np.percentile(times,50):.3f} p90={np.percentile(times,90):.3f} "
          f"p99={np.percentile(times,99):.3f} mean={times.mean():.3f}")


if __name__ == "__main__":
    test_naive_converges()
    test_anderson_faster_than_naive()
    test_spectral_radius_finite()
    test_latency_benchmark()
    print("equilibrium_engine tests passed")
