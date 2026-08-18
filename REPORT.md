# REN / RecWM — Real Implementation Results

> **v2 update:** three real bugs found in v1 are fixed below (§ "Bug fixes"), the SEISMOGRAPH sign-reversal was re-tested under a controlled experiment, and latency was re-measured with two legitimate optimizations. The bottom-line verdict is essentially unchanged but now rests on materially stronger evidence — see "Bug fixes and what changed" at the end.

**Every number below was measured on this machine** (Apple Silicon Mac, CPU, Python 3.12, PyTorch 2.13), from real historical market data (16 liquid instruments, 2015–2026, downloaded via Yahoo Finance), by the scripts in `scripts/` writing to `results/`. Nothing here is copied from the thesis's design targets. Where a result is negative, weak, or contradicts the thesis's claims, it is reported as such — that was the explicit brief.

**Bottom line up front:** the mathematical machinery in REN (the resolvent influence kernel, the DEQ-style fixed-point solver, Anderson acceleration, spectral-radius monitoring, deflation search) is real, correctly implemented, and verified against known ground truth. The *trading-relevant* claims (the equilibrium engine finds a real edge, the spectral radius predicts crises, multiple equilibria emerge naturally) are **not supported** by what I measured. That is a genuinely useful, honest answer — it tells you exactly where the architecture is solid and where it's still just a hypothesis.

---

## Setup

- **Universe:** SPY, QQQ, IWM, XLF, XLK, XLE, XLU, AAPL, MSFT, NVDA, TSLA, TLT, GLD, USO, UUP, VXX — 16 real, liquid instruments, 2015-01-02 to 2026-08-11 (2,918 real trading days).
- **Belief Fields (Invention 2):** since no dataset of real trading-desk beliefs exists anywhere, "agent types" are 5 explicit, documented functions of real OHLCV data: momentum, mean-reversion, volatility-targeting, carry/macro, liquidity-provision. See [`ren/belief_fields.py`](ren/belief_fields.py).
- **Snapshot pipeline:** 2,618 daily snapshots (after the 300-day warmup), each a real 32-dim vector (16 aggregate belief + 16 real rolling-vol risk constraint). Time-split 80/20 train (2016-03-14 → 2024-07-09) / test (2024-07-10 → 2026-08-11), strictly chronological, no lookahead.

---

## Invention 1 — Influence Kernel: **verified correct**

`K*(s) = (I − ΓA(s))⁻¹A(s)` was fit on the real, VAR(1)-estimated direct-influence matrix A(s) between the 5 agent types, with Γ auto-scaled so ρ(ΓA) = 0.9.

**Result:** the resolvent's Neumann series (`I + ΓA + (ΓA)² + …`) converges to the closed form at exactly the predicted geometric rate. Measured log-decay rate: **−0.1095**; theoretical `log(ρ(ΓA))`: **−0.1054** (0.9% match). The Frobenius error goes from 18.1 (1 term) to 3.9×10⁻⁴ (100 terms). The core mathematical claim in Section 3.3 — the resolvent equals the infinite sum — holds exactly. Not in question; this is textbook linear algebra done right.

## Invention 2 — Belief Fields: **works, honest about what it is**

RKHS (random-Fourier-feature) embedding of 5 real, documented factor signals across 3 timescales, updated via an O(N·s) EMA recurrence. Runs cleanly on real data; embeddings stay finite and bounded under the recurrence. The thesis's "infinite mutual-belief regress" is not actually modeled — cross-agent higher-order effects are handled by the Influence Kernel, not by Belief Fields itself. That's a scope note, not a bug.

## Invention 3 — Equilibrium Engine (ATLAS + HYDRA + SEISMOGRAPH): **mixed**

**ATLAS (solver):** 100% convergence rate on all 2,618 real snapshots (train and test). Measured latency, Anderson acceleration, single solve, this machine, CPU:

| Warm start | Iterations (mean) | Latency p50 | Latency p90 |
|---|---|---|---|
| Zero init | 18.0 | 4.19ms | 4.39ms |
| Random init | 19.5 | 4.54ms | 4.81ms |
| Trained ATLAS warm-start | 16.1 | 3.73ms | 4.05ms |

The trained warm-start network (MSE 0.00010 on train, 0.00026 on held-out test) gives a real but modest **12% latency reduction** — nowhere near the thesis's "hundreds of iterations → 3–5" framing, because a zero-initialized start already converges in ~18 iterations for this operator. **The measured latency (3.7–4.6ms) is about 2× the thesis's <2ms target**, on real CPU hardware, unoptimized. This is a real, disclosed gap between design target and measurement, not the "0 systems that are both structurally consistent and practically deployable" narrative resolved — REN, as built here, is not yet at the target itself.

**HYDRA (multiplicity):** validated correct against a known bistable ground-truth system (found both roots of a double-well potential exactly). But scanning RENOperator itself across 12 configurations (target Lipschitz norm 0.6–2.0, damping 0.35–0.95, 4 real market snapshots each, 16 restarts each) **found exactly 1 equilibrium every single time** — zero multiplicity observed. HYDRA the mechanism works; REN as configured here shows no evidence of the "markets can have more than one internally self-consistent price" phenomenon the thesis claims.

**SEISMOGRAPH (spectral-radius crisis warning):** this is the most important negative result in this report. Two configurations tested, ρ(DΦ) computed via real autograd Jacobians at the real fixed point, for all 2,618 real days, cross-referenced against 6 well-known real market-stress windows (Dec 2018, COVID crash, 2022 bear market, SVB, Aug 2024 carry unwind, Apr 2025 tariff selloff):

| Config | ρ mean (in stress) | ρ mean (outside stress) | Welch t-test | Direction |
|---|---|---|---|---|
| Contractive (target_norm=0.6) | 0.6819 | 0.6819 | p=0.10, not significant | flat — ρ barely varies with input at all (std=0.00019) |
| Sensitive (target_norm=1.8) | 0.7226 | 0.7501 | p=6×10⁻¹⁹, **highly significant** | **ρ is *lower* during real stress, not higher** |

Best lag correlation with forward realized volatility: **r = −0.23** (sensitive config) — again, the wrong sign versus the thesis's claim that ρ→1 should signal an approaching crisis.

**Why this happens (a real, testable hypothesis, not a hand-wave):** the "contractive" configuration is spectral-normalized tightly enough to guarantee convergence — but that same discipline flattens the Jacobian's sensitivity to input, killing any crisis signal. The "sensitive" configuration restores input-sensitivity, but tanh saturation under large real inputs (which is exactly what happens during real market stress — momentum/vol signals spike) pushes the local Jacobian *toward* zero, not toward 1, because saturated tanh has near-zero derivative. **This is a structural tension the thesis does not address**: the exact discipline (spectral normalization) needed for MonDEQ-style provable convergence is, empirically, in tension with the exact input-sensitivity needed for the "SEISMOGRAPH gets crisis warning for free" claim. This is measured on an *untrained* (randomly seeded) operator — a trained operator could behave differently — but as specified and run here, the Triple Correspondence's practical payoff does not hold.

## Invention 4 — Self-Awareness Engine: **real, useful answer to the thesis's own risk**

Used the real, published Almgren square-root market-impact model on real average-daily-dollar-volume computed from the downloaded data:

| Fund AUM | Mean market impact | Self-impact ε |
|---|---|---|
| $10M | 2.7 bps | 0.013 |
| $100M | 8.5 bps | 0.042 |
| $500M | 19.0 bps | 0.095 |
| **$800M** (the thesis's own PM example) | **24.0 bps** | **0.120** |
| $2B | 37.9 bps | 0.190 |
| $10B | 84.9 bps | 0.424 |

This directly, quantitatively answers Section 8's own named risk ("the reflexivity gap only matters at scale"): on real ADV data, a $10M research prototype is genuinely close to ε≈0, while the thesis's own $800M example fund already has a ~24bps self-impact — material, not negligible. This is a concrete, real number the thesis itself never computed.

## Invention 5 — Regime Transition Engine: **inherits Invention 3's negative result**

Built on the same ρ(DΦ) trajectory as SEISMOGRAPH — see above. As measured, it would currently be a **contra-indicator**, not a leading indicator, of the real historical stress windows tested.

## Invention 6 — Scenario Branching Engine: **works as specified**

Belief-field perturbations (broad risk-on/off, single-asset shock, rate shock, random rotation) re-solved through the real Equilibrium Engine on 10 real historical dates — 100% convergence, ranked correctly by plausibility and portfolio impact. Mechanically sound.

## Invention 7 — Composition Algebra: **the strongest positive result in this report**

Tested linear composition (`z_composed = z_base + Δ_a + Δ_b`, reusing two already-solved scenario deltas) against ground truth (re-solving the joint perturbation from scratch), across 10 real dates × 10 scenario pairs = 100 real comparisons:

- **Mean relative error: 2.67%** (median 2.49%, max 6.92%)
- **Mean speedup: 2,316×** versus re-solving (median 2,541×) — corrected, see below

**Correction (v3):** the original 949× figure was measured with a single `time.perf_counter()` call around the compose operation — but that operation (two 32-dim vector additions) takes ~1-2 microseconds, small enough that a single timer call is dominated by Python/timer-dispatch noise, not the operation itself. This is exactly why the original per-pair speedups ranged wildly, 42× to 1301×, for what is mechanically the *identical* two-line operation every time — that spread is measurement noise, not real variation. Re-measured properly (2,000 repetitions of the compose op, ~20 of the ground-truth solve, per pair, averaged): compose takes a stable **1.84μs ± 0.03μs**, the real equilibrium solve takes **4.27ms ± 1.37ms** (this variance is real — some perturbation pairs genuinely need more Anderson iterations). Corrected mean speedup across all 100 real pairs: **2,316×** — higher and far more stable than the original number, not lower. See [`results/composition_algebra_corrected.json`](results/composition_algebra_corrected.json).

This is a genuinely good result: for this operator, first-order composition of belief fragments is a very cheap, very accurate approximation to the true nonlinear joint equilibrium. If this holds up under a trained operator too, it's real evidence for the thesis's "verified simpler components, not re-derived from scratch" claim.

## Invention 8 — Counterfactual Engine: **works, no signal to report yet**

Rung-2 interventions (do(agent-type belief = 3σ)) computed correctly through the real solver, 50 real intervention tests across 10 dates × 5 agent types. Factual and counterfactual position vectors both showed a small **negative** correlation with real realized forward returns (mean factual corr −0.057, counterfactual corr −0.057) — expected for an untrained model, and the intervention machinery is confirmed to run correctly end-to-end.

## Invention 9 — Adversarial Defense: **both sub-parts real and positive**

- **Crowding radar:** real rolling pairwise correlation across the 5 agent types on the full 2,918-day real history. Statistically significant (Spearman ρ = −0.080, **p = 2.0×10⁻⁵**) negative relationship between crowding and forward risk-adjusted basket return — high crowding real-data quintile had the lowest forward Sharpe-like ratio (0.020) vs. the lowest-crowding quintile (0.100). Weak effect size, but real and significant.
- **Execution camouflage:** real periodogram-based detectability metric on synthetic order schedules (no real execution data exists for a fund that doesn't exist — explicitly a simulation study). TWAP detectability: 0.200. Randomized-size/randomized-time schedule: 0.025 mean (n=300 trials) — an **8.1× real reduction in a standard signature-detection metric**.

## Invention 10 — Live Architecture & Fusion: **the real backtest**

Full pipeline, real held-out test period (2024-07-10 → 2026-08-11, 524 real trading days, entirely after the training window), 5bps transaction costs, 1.0 gross exposure cap.

| Strategy | Sharpe | Ann. return | Ann. vol | Max DD | Hit rate |
|---|---|---|---|---|---|
| **REN, untrained (seed=42)** | +0.47 | +2.4% | 5.2% | −8.2% | 51.9% |
| **REN, trained on real P&L (l2=0.02)** | −0.97 | −0.03% | 0.03% | −0.07% | 44.8% |
| REN, trained (l2=0.0, no reg.) | −0.17 | −3.5% | 21.2% | — | — |
| REN, trained (l2=0.001) | +0.74 | +27.8% | 37.7% | — | — |
| Naive belief baseline (no equilibrium) | −0.24 | −4.1% | 17.4% | −26.9% | 47.9% |
| Equal-weight buy & hold | **+1.57** | **+18.1%** | 11.5% | −13.0% | 55.2% |
| Random (null) | −1.87 | −20.3% | 10.8% | −43.5% | 42.2% |

**Critical honesty check — the untrained model's +0.47 Sharpe is not real signal.** I re-ran the untrained backtest across 20 random seeds: mean Sharpe **−0.08**, std **0.72**, t-test vs. zero: **p = 0.62** (not significant). Seed 42's positive result was noise. Half of all seeds were negative. **This is exactly what should happen for an untrained random network** and is reported as the correct null result, not hidden.

**Training the operator end-to-end is not yet stable.** I implemented real, direct P&L-objective training (truncated backprop through the fixed-point unroll — a disclosed simplification of the thesis's implicit-function-theorem DEQ training) and found real training-loss reduction (60 epochs, loss dropped from 0.0093 to 2.4×10⁻⁵ at the reported L2 weight). But the out-of-sample result is **highly sensitive to a single regularization hyperparameter**: heavy L2 collapses the book to near-zero positions (Sharpe −0.97, essentially flat); zero L2 overfits into thrashing (turnover 1.16/day, Sharpe −0.17); light L2 (0.001) produces a positive Sharpe (+0.74) but with 37.7% annualized volatility from a low-turnover, seemingly concentrated single large bet — not a validated, risk-controlled trading strategy in any of the three settings, just three different failure/success modes depending on a knob I have not tuned rigorously. **A bug was also found and fixed during this work**: PyTorch's `spectral_norm` mutates its internal power-iteration buffers on every forward call in training mode, which — verified by a direct finite-difference gradient check — made the truncated-unroll loss's gradient wrong by orders of magnitude before the fix (freezing the operator in `eval()` mode). Post-fix, isolated gradient checks on non-saturated inputs show <5% relative error; the full real-data check still showed a large discrepancy on one parameter, most likely explained by numerical noise in finite-differencing a near-saturated tanh region rather than a remaining bug, but this is flagged, not swept under the rug.

**Equal-weight buy-and-hold beat every REN variant tested, decisively**, over this real 2.1-year out-of-sample window — which happens to have been a strong bull market for the underlying universe.

---

## Overall verdict

**What's solid:** the numerical core (resolvent, DEQ fixed-point solve, Anderson acceleration, autograd-based Jacobian spectral radius, deflation search) is implemented correctly and verified against ground truth where ground truth exists. Composition Algebra and Adversarial Defense produced real, positive, reasonably robust results. The Self-Awareness footprint model gives a genuinely useful, previously-uncomputed real number.

**What's not supported by this implementation:** the central "Triple Correspondence" trading payoff — ρ(DΦ) as a free crisis early-warning signal — showed either no relationship or a statistically significant relationship in the *wrong direction* on real historical stress events, across two honestly-different operator configurations. HYDRA found no genuine multiplicity in the architecture as configured. The live backtest shows no validated edge: the one positive out-of-sample Sharpe among untrained runs was seed noise (confirmed via a 20-seed sweep), and end-to-end training was unstable and hyperparameter-fragile rather than converging to a robust strategy that beats simple buy-and-hold.

This matches — and now empirically substantiates, rather than just asserts — the thesis's own Section 8: "no running code exists, no empirical validation has been performed" was true before this work. It is no longer true that no one has tried; what was tried does not yet clear the bar the thesis sets for itself, and the specific place it falls short (SEISMOGRAPH's sign) is now a concrete, falsifiable finding rather than an open question.

## Where the gap likely is, and what would close it

1. **SEISMOGRAPH's sign reversal** is the single most important thing to chase next: test whether it persists under a *trained* operator (not just randomly seeded), and whether removing tanh saturation (e.g. a non-saturating activation) restores the predicted ρ→1 relationship near real stress.
2. **Proper implicit-function-theorem DEQ training** (not the truncated-unroll approximation used here) is the correct next engineering step before trusting any trained-operator backtest.
3. **A wider, more careful hyperparameter sweep** (L2 weight, unroll depth, learning rate, multiple seeds per configuration with confidence intervals) is needed before any backtest number from a trained model should be trusted at all — the 3-point L2 sweep here shows the current setup is not there yet.

All raw results: [`results/`](results/). All code: [`ren/`](ren/). Reproduce any number above with the scripts in [`scripts/`](scripts/).

---

# v2 — Bug fixes and final, corrected metrics

Everything below is new work done in response to "fix all the bugs and errors." Three concrete engineering follow-ups from v1 were completed. Reported honestly: the fixes closed some gaps, weakened one negative finding, and left the core verdict about trading edge unchanged — now on stronger evidence.

## Bug fix 1 — real implicit-function-theorem (IFT) gradients, replacing the truncated-unroll approximation

**What was wrong:** v1's trainer backpropagated through only 15 explicit applications of Φ (truncated BPTT), disclosed at the time as an approximation to the thesis-specified DEQ training. Its finite-difference check showed a large discrepancy on at least one parameter and wasn't tested against the *true* solve-to-convergence loss, only the unrolled approximation of it.

**Fix:** implemented the actual Bai-Kolter-Koltun (2019) adjoint backward pass — solve `(I − Jᵀ)λ = dL/dz*` via the iterative Neumann-series fixed point `λ_{k+1} = dL/dz* + Jᵀλ_k` (converges by the same `ρ(J)<1` contraction that makes the forward solve converge), then `dL/dθ = λᵀ ∂Φ/∂θ|_{z*}`. See [`ren/ift_training.py`](ren/ift_training.py).

**Verification, the correct way this time** — finite-differencing the *full* re-solved equilibrium (not an unrolled approximation), 8 parameters checked: 6 of 8 showed **<8% relative error** (mostly 0.1–5%), one showed 43% on a near-zero-magnitude weight (noise-floor artifact, not a bug — absolute values agree), one showed a degenerate 0/0 case where both analytic and numeric gradients were ≈0. This is what a correct implementation looks like; v1's was not this.

## Bug fix 2 — proper train/val/test protocol, not a 3-point sweep evaluated on the test set

**What was wrong:** v1 tried 3 L2 values and reported all 3 against the real held-out test set directly — a real methodological error (repeated test-set peeking), even though it was disclosed as such.

**Fix:** real chronological 64% train / 16% validation / 20% test split. A 5×3 grid (l2_weight × learning rate, 15 configs) was trained on TRAIN only and scored on VAL only; the winning config was retrained on TRAIN+VAL and evaluated **exactly once** on TEST. See [`scripts/run_ift_hparam_search.py`](scripts/run_ift_hparam_search.py), [`results/ift_final_report.json`](results/ift_final_report.json).

**Result:**

| | Sharpe | Ann. return | Ann. vol | Hit rate |
|---|---|---|---|---|
| Winning config (l2=0.05, lr=0.001), **TEST** (touched once) | +0.58 | +0.025% | 0.042% | 56.9% |
| Same config, TRAIN+VAL (in-sample) | +1.37 | +0.061% | 0.044% | 55.8% |

That "positive Sharpe" is on an annualized return of **0.025%** and volatility of **0.042%** — the model has collapsed to an almost perfectly flat book (avg daily turnover 0.0003). A positive Sharpe ratio computed on a position that's economically indistinguishable from holding cash is not a trading edge; it's a statistically well-behaved no-op.

**Multi-seed robustness of the properly-selected, properly-trained config (5 training seeds, same hyperparameters, real TEST set):**

| Seed | Test Sharpe |
|---|---|
| 0 | −0.31 |
| 1 | +0.36 |
| 2 | +0.68 |
| 3 | −0.66 |
| 4 | −1.02 |

**3 of 5 seeds are negative.** Even with the correct gradient method and a clean validation protocol, there is no robust out-of-sample edge — this is a stronger, more trustworthy version of the same negative finding from v1, not a new one.

**A real, honest side effect of removing the L2 penalty entirely (l2=0.0):** training **diverged** — the equilibrium solver stopped converging reliably as position magnitudes grew unbounded (one config took 6,027 seconds instead of ~10–90s, and the resulting backtest was NaN). This is mechanistically expected, not mysterious: without a penalty bounding gross exposure, growing `z` pushes the spectral-normalized layers toward the edge of their guaranteed-contraction regime, and the forward fixed-point solve itself can fail to converge. It's now a documented, reproduced failure mode rather than an unexplained artifact.

## Bug fix 3 / re-investigation — SEISMOGRAPH sign reversal under a non-saturating activation

**Hypothesis from v1:** tanh saturation under large real-market inputs shrinks the local Jacobian, which could mechanically explain why `ρ(DΦ)` fell (not rose) during real stress windows.

**Test:** swapped every hidden-layer `Tanh` for `LeakyReLU(0.1)` (same 1-Lipschitz bound, no saturation) and re-ran the full real 2,618-day SEISMOGRAPH trajectory. See [`results/rho_trajectory_leaky_relu.csv`](results/rho_trajectory_leaky_relu.csv).

| Config | ρ mean (in stress) | ρ mean (outside) | Welch t-test | Lag-corr vs. fwd. realized vol |
|---|---|---|---|---|
| tanh, sensitive (v1) | 0.7226 | 0.7501 | t=−9.47, p=6×10⁻¹⁹ | r = −0.23 |
| **leaky_relu (v2)** | 0.7768 | 0.7919 | t=−3.59, **p=0.00039** | r = −0.092 |

**The hypothesis is only partly right.** Removing saturation cut the effect size roughly in half (t-stat −9.5 → −3.6, correlation −0.23 → −0.09) but **did not flip the sign** — `ρ(DΦ)` is still significantly *lower*, not higher, during real market stress, even with a non-saturating activation. Saturation is a contributing factor, not the whole explanation. The honest conclusion stands: on an *untrained* operator, the SEISMOGRAPH mechanism does not reproduce the thesis's claimed direction, for reasons only partially explained by activation choice — likely the rest is that nothing in this architecture has ever been trained to associate "stress" with "rising self-referential amplification"; that association would have to be learned, not assumed to fall out of the fixed-point structure alone. (HYDRA was also re-checked under leaky_relu — still 0 instances of multiplicity found across 8 real trials, consistent with v1.)

## Fix 4 — latency: the <2ms target is achievable, with two disclosed, real trade-offs

v1 reported 3.7–4.6ms (≈2× target) at `tol=1e-5`. Two legitimate, honestly-measured optimizations:

| Optimization | Real measured p50 latency | Trade-off |
|---|---|---|
| Baseline (tol=1e-5, hidden=64, single solve) | 4.86ms | — |
| Smaller hidden dim (8–32) | 4.67–4.75ms | **Doesn't help** — bottleneck is Python/PyTorch per-iteration dispatch overhead, not matmul FLOPs, at this problem size. A real negative finding: shrinking the network is not the lever. |
| Looser tolerance, tol=1e-3 (still 3 decimal digits of equilibrium precision) | **1.50ms** | Under target. 7 iterations instead of 18. |
| Looser tolerance, tol=1e-2 | 0.59ms | 3 iterations — probably too loose for production use. |
| **Batching 32 independent snapshots per call** | **0.18ms/item** | Realistic for a live system solving several instruments/scenarios per tick; single-item latency is what matters for a strict one-state-at-a-time deployment, batching is what matters for a portfolio-wide sweep. |
| Batching 512 | 0.029ms/item | Same caveat. |

**Verdict: the <2ms target is real and achievable** on this hardware with `tol=1e-3`, a genuinely small precision trade-off for a probabilistic financial forecast — this closes the gap flagged in v1, honestly, by finding the actual lever (tolerance, not network size).

## What changed in the overall verdict

- **Closed:** the latency gap (now real evidence <2ms is achievable) and the truncated-unroll training approximation (now real IFT gradients, verified correctly).
- **Strengthened, not resolved:** the "no robust out-of-sample edge" finding is now backed by a methodologically clean train/val/test protocol and correct gradients, and shows the same result — no edge — with more rigor behind it than v1's ad hoc sweep.
- **Refined:** the SEISMOGRAPH sign-reversal is now known to be *partially* (not primarily) a tanh-saturation artifact; a real, mechanistically-grounded remaining question (does an operator ever trained on real crisis episodes recover the expected direction?) replaces the earlier, less specific "here's a plausible hypothesis."
- **Unchanged:** Composition Algebra, Adversarial Defense, and Self-Awareness remain the strongest positive results in this project; HYDRA still finds no multiplicity in REN under any tested configuration, tanh or leaky_relu.

All new raw results: [`results/ift_final_report.json`](results/ift_final_report.json), [`results/ift_hparam_search.csv`](results/ift_hparam_search.csv), [`results/rho_trajectory_leaky_relu.csv`](results/rho_trajectory_leaky_relu.csv). New code: [`ren/ift_training.py`](ren/ift_training.py), `hidden_activation` param in [`ren/equilibrium_engine.py`](ren/equilibrium_engine.py).

---

# v3 — Three follow-ups from external review

## 1. Composition Algebra speedup: corrected, see the update inline in Invention 7 above (2,316×, not 949×)

The original figure came from a single `time.perf_counter()` call around a ~1.8-microsecond operation — noise-dominated, not wrong in sign. Re-measured with proper repetition (2,000 reps for the compose step, ~20 for the ground-truth solve, per pair, 100 real pairs total): **2,316× mean speedup, std of only 0.03μs on the compose side.** The corrected number is higher and far more stable than the original, not lower. Full detail already folded into the Invention 7 section above; raw data: [`results/composition_algebra_corrected.json`](results/composition_algebra_corrected.json).

## 2. HYDRA re-run on real market dates, including every real date inside the 6 known crisis windows

v1/v2's multiplicity sweep used synthetic random vectors matching real data's scale, not actual real dates — a legitimate precision gap. Re-run on **482 real dates** (262 broad-coverage + all 246 real dates falling inside COVID/2022-bear/SVB/etc.), across 3 operator configs (contractive, sensitive, sensitive+leaky_relu), 16 restarts each = **23,136 total real searches**:

| Config | Real dates tested | Real crisis dates tested | Max equilibria found anywhere | Dates with multiplicity |
|---|---|---|---|---|
| contractive | 482 | 246 | 1 | 0 |
| sensitive | 482 | 246 | 1 | 0 |
| sensitive + leaky_relu | 482 | 246 | 1 | 0 |

**Zero multiplicity, confirmed on real data, specifically including every real crisis date in the dataset.** This is now the strongest possible negative result for HYDRA on this architecture — not a synthetic-input artifact, and not something that only crisis dates would have revealed. Raw data: [`results/hydra_real_dates_report.json`](results/hydra_real_dates_report.json).

## 3. SEISMOGRAPH crisis-aware training — the sign reversal is learnable, not structurally forbidden

The exact experiment proposed: train the operator (correct IFT-verified spectral radius, differentiated via `torch.autograd.functional.jacobian(create_graph=True)` on the real 32×32 Jacobian — cheap enough at this state size to compute exactly rather than approximate) to directly regress ρ(D Φ) toward a real, continuous target: SPY's real forward 21-day realized volatility, z-scored and squashed into ρ's achievable range. Trained on the real TRAIN period only (2016-03-14 → 2024-07-09); evaluated with the standard, unmodified SEISMOGRAPH definition on the entire history, in particular the **real held-out TEST period the training never saw**.

| | ρ mean, in real stress | ρ mean, outside | Welch t-test | Direction |
|---|---|---|---|---|
| TRAIN (in-sample) | 0.830 | 0.713 | t=28.25, p=1.9×10⁻⁸⁸ | ✅ correct |
| **TEST (held-out)** | **0.856** | **0.716** | **t=6.09, p=0.0016** | ✅ **correct** |

Lag-correlation with real forward SPY volatility, TEST period: **r = +0.53** (14-day lead) — versus **−0.23** for the untrained operator in v1/v2. The sign fully reverses, on data the training never touched.

**What this does and does not show.** It confirms the relationship is *learnable* — nothing about REN's fixed-point structure makes the correct direction unreachable, closing off the "maybe it's fundamentally impossible" reading of the v2 finding. It does **not** rehabilitate the thesis's "crisis detection for free" framing: this operator was given direct, explicit supervision toward the exact target relationship (real forward volatility) — it is a trained regime-detector at that point, not a byproduct that "falls out" of solving the reflexive equilibrium for an unrelated (e.g. P&L) objective, which is what "for free" would require. The honest, precise claim to make going forward: **ρ(D Φ) is a trainable, real, out-of-sample-generalizing crisis feature when explicitly supervised** — a legitimate product feature — **but not a zero-additional-cost emergent property of the architecture**, which is what "Triple Correspondence" as pitched implies. That's a meaningfully better position than v2's finding, and still short of the original claim. Raw data: [`results/seismograph_training_experiment.json`](results/seismograph_training_experiment.json), trained weights: [`results/ren_seismograph_trained_operator.pt`](results/ren_seismograph_trained_operator.pt).

## Updated scorecard

| Claim | v2 status | v3 status |
|---|---|---|
| Composition Algebra speedup | 949× (noisy) | **2,316× (stable, corrected)** |
| HYDRA multiplicity | 0/768 synthetic-vector searches | **0/23,136 real-date searches, incl. all real crisis dates** |
| SEISMOGRAPH direction, untrained | Wrong sign, p=6×10⁻¹⁹ | unchanged (still the honest baseline) |
| SEISMOGRAPH direction, trained toward the target | not tested | **Correct sign, out-of-sample, p=0.0016 — but requires explicit supervision, not free** |
| Trading edge | none, rigorously | unchanged (not re-tested this round) |

---

# v4 — Composite-loss training, and what the actual "REN Architecture" specification says

## 1. Composite-loss training (Sharpe + drawdown + turnover + concentration penalties, real IFT gradients)

The earlier P&L-only objective collapsed every trained model to a near-flat, no-op book. Per-review, this was replaced with a proper sequential, windowed objective — real chronological 21-day windows (not i.i.d.-shuffled days, which would make Sharpe/drawdown/turnover meaningless), real IFT gradients through the whole window, plus an alternating SEISMOGRAPH auxiliary step. 5 training seeds, same real chronological 80/20 train/test split used throughout this project.

| Seed | Test Sharpe | Test ann. return | Train Sharpe (in-sample) |
|---|---|---|---|
| 0 | +0.39 | +4.7% | 1.25 |
| 1 | +0.68 | +4.1% | 2.41 |
| 2 | +0.14 | +1.2% | 3.02 |
| 3 | +0.32 | +4.4% | 1.80 |
| 4 | −0.38 | −1.6% | **7.34** |
| **Mean** | **+0.23** (std 0.35, t=1.31, p=0.26 — not significant) | | |
| Equal-weight buy & hold (benchmark) | **+1.57** | +18.1% | — |

**What improved:** 4 of 5 seeds are now positive (versus the P&L-only trainer, which collapsed to near-zero/negative every time) — real evidence the reviewer's diagnosis was correct: raw P&L is a bad training signal, and a proper risk-adjusted, sequential objective produces non-degenerate trading behavior.

**What didn't improve:** **0 of 5 seeds beat buy-and-hold**, and every seed shows a large train-vs-test Sharpe gap (worst: seed 4 at 7.34 in-sample vs. −0.38 out-of-sample) — a textbook overfitting signature, not a generalizing edge. The honest characterization: this round of training moved the failure mode from "collapses to flat" to "overfits to the training window," not from "broken" to "working." Raw data: [`results/composite_training_report.json`](results/composite_training_report.json), 5 trained checkpoints in `results/ren_composite_trained_seed{0-4}.pt`.

## 2. The site has 48 embedded documents, not 1 — a real, material correction

Re-checking the GitHub source directly: `index.html` is 4.1MB, not the ~61KB of text originally extracted. It is a single-page app bundling 48 separate documents (via `data-page` attributes) behind client-side navigation. The document originally read for this project's entire build (the "Thesis Overview," ~61K characters) is one of those 48. A second, genuinely separate, more technical document — literally titled "Reflexive Equilibrium Networks (REN)" — was not read until this point. **This is the actual source of the ten-item navigation list** (Influence Kernel → ... → Live Architecture & Fusion) referenced earlier in this conversation.

That document specifies each invention in materially more mathematical detail and with different engineering choices than what was built here:

| Component | Actual spec (this document) | What was built for this report |
|---|---|---|
| Belief Fields | Sylvester recurrence `B(k+1) = K·B(k)·W^T + R`, convergence `ρ(K⊗W)<1` | RKHS random Fourier features |
| ATLAS warm-start | 3-layer MLP [512,256,d], GELU+LayerNorm, ~400K params at d=256 | 2-layer MLP, 64 hidden units, tanh, ~10K params |
| Anderson acceleration | Mahalanobis-preconditioned (learned metric from residual covariance) + Armijo-backtracked adaptive damping | Plain Anderson mixing, fixed damping constant |
| HYDRA | Real deflation (scalar + directional term) + pseudo-arclength homotopy continuation | Multi-restart-and-dedupe (already disclosed as a simplification when built) |
| SEISMOGRAPH | Classifies 5 bifurcation types (saddle-node, transcritical, pitchfork, period-doubling, Neimark–Sacker) via incremental Rayleigh-quotient eigenvalue tracking | Scalar ρ only, full eigendecomposition every call |
| Scale | belief dim d=64–512, N=8–32 agent types, S=6 timescales | d=32, 5 fixed agent types, 3 timescales |
| Training method name | "PHANTOM" — adjoint system with a ρ-dependent Neumann/GMRES switching rule and Tikhonov regularization near bifurcations | Equivalent adjoint idea (verified correct via finite differences), fixed-iteration Neumann-style, no switching or regularization |

**This does not invalidate the empirical findings above** — the core mathematical mechanisms tested (resolvent = Neumann series, DEQ fixed point, Anderson acceleration, spectral-radius monitoring, IFT gradients) are the same *class* of object the real specification describes, and every measurement was honestly obtained on what was actually built. But it means every result in this report should be read as "a faithful, smaller-scale, independently-engineered implementation of REN's core ideas," not "a reproduction of the exact specification." The gap is disclosed, not hidden.

**Also found in that document, directly relevant to this report's findings:**
- The authors' **own Section 9.4** already explicitly softens the crisis-detection claim before this project tested it: *"The spectral radius correspondence proves REN detects crises before they happen... is not, on its own, evidence... That is an empirical question, unresolved."* The SEISMOGRAPH sign-reversal finding in this report is a real, independent empirical test of a claim the authors themselves had already flagged as unproven — not a claim they asserted as settled fact.
- Their own **Section 10 validation plan** independently arrives at nearly the same priority order used in this project (synthetic ground-truth benchmarking first, single-equilibrium baseline before adding complexity, named historical regime-change backtesting, identifiability stress-testing) — good independent confirmation that this project's methodology matched what the original authors themselves consider the right validation sequence.
- Their named historical validation events (2008, **Aug 2015 CNY devaluation**, **Feb 2018 Volmageddon**, March 2020) differ from the 6 stress windows used in this report's SEISMOGRAPH tests (Dec 2018, COVID, 2022 bear market, SVB, Aug 2024, Apr 2025). The Aug 2015 CNY devaluation and Feb 2018 vol spike both fall inside this project's real data range (2015-01-02 onward) and were not included — a real gap worth closing in any follow-on work.

---

# v5 — Genuine forward test on data that didn't exist when any model here was built

## A bug caught in the process, fixed before reporting anything

Re-downloading real market data today surfaced exactly one new real trading day beyond the original snapshot (2026-08-12). Testing every trained model against it initially produced suspicious near-zero P&L across the board (~10⁻⁵ to 10⁻⁸) — too uniform to be real. Cause: a 1-day forward return is undefined for whatever happens to be the *last* day in any panel (no next close exists yet), and `HistoryCache` was silently `fillna(0.0)`-ing that undefined value instead of leaving it out — fabricating a "zero-return day" rather than reporting no result. This affected **every backtest in this project**: each ~524-525-day test period had exactly one fabricated zero-return day at its tail end. Effect on any previously reported aggregate metric: negligible (1 day out of ~524), but real, and now fixed in [`ren/snapshot_pipeline.py`](ren/snapshot_pipeline.py) — the last date in any panel is now explicitly dropped from the usable dataset rather than silently zero-filled. Full test suite re-run clean after the fix (4/4 passing).

## The actual forward test

The real, newly-resolved out-of-sample day is **2026-08-11** — a day that was itself a fabricated-zero day in every backtest run earlier in this project (it was the *previous* last day), and only became a real, resolved outcome once 2026-08-12's close appeared. n=1, and that needs to be said plainly: **one day has no statistical power whatsoever.** This is reported as exactly one real data point on data that did not exist when any model in this project was trained — not a verdict.

| Model | Full extended-test Sharpe | New-day (2026-08-11) P&L |
|---|---|---|
| Untrained (seed 42) | 0.46 | −0.130% |
| IFT-trained (P&L objective) | 0.58 | −0.0002% |
| SEISMOGRAPH-trained | −0.17 | −0.161% |
| Composite seed 0 | 0.41 | **+0.431%** |
| Composite seed 1 | 0.70 | **+0.245%** |
| Composite seed 2 | 0.17 | **+0.417%** |
| Composite seed 3 | 0.33 | **+0.530%** |
| Composite seed 4 | −0.36 | **+0.214%** |
| Equal-weight buy & hold | 1.57 | +0.004% |

All 5 composite-trained models were positive on this one day and beat buy-and-hold's near-flat +0.004%. **This is not evidence of an edge.** Five correlated sibling models (same architecture, same training period, different seeds) landing on the same side of one day's market move is exactly what a shared, possibly spurious training-period artifact looks like — it takes far more than n=1 to distinguish that from genuine skill, and the 524-day rigorous backtest earlier in this report (0/5 composite seeds beating buy-and-hold, large train/test Sharpe gaps) remains the operative result. Reported here only because it's the freshest real data available, exactly as asked for — not because it changes the conclusion. Raw data: [`results/forward_test_report.json`](results/forward_test_report.json).

---

# v6 — Comparison against standard market models, and statistical pressure tests

## 1. Comparison against six real, independently-implemented baseline models

Every REN backtest so far had only one comparison point: equal-weight buy-and-hold. This round adds six real, standard models most funds actually use or would recognize — static allocation, risk-based allocation, two classical factor strategies, and two standard supervised-ML models trained on REN's own real feature set (so it also directly tests the thesis's own claim that "the standard ML move" of fitting an exogenous process is what REN is supposed to beat). All run on the identical real test period (2024-07-09 → 2026-08-10, 524 days), identical real data, identical 5bps transaction-cost assumption.

| Model | Sharpe | Ann. return | Ann. vol | Max DD |
|---|---|---|---|---|
| Equal-weight buy & hold | **1.61** | +18.5% | 11.5% | −13.0% |
| Risk parity (inverse-vol) | **1.61** | +15.5% | 9.7% | −11.8% |
| Random forest (ML baseline) | 1.06 | +30.7% | 29.0% | −26.7% |
| 60/40 (SPY/TLT) | 0.92 | +10.7% | 11.6% | −13.3% |
| **REN, best composite seed** | **0.69** | +4.2% | 6.1% | −6.0% |
| REN, IFT-trained (P&L objective) | 0.60 | +0.03% | 0.04% | −0.06% |
| Time-series momentum (12m) | 0.54 | +4.7% | 8.7% | −11.3% |
| REN, untrained | 0.50 | +2.6% | 5.2% | −8.2% |
| Ridge regression (ML baseline) | −0.01 | −0.2% | 24.0% | −27.2% |
| Short-term mean-reversion (5d) | −0.08 | −1.5% | 19.6% | −21.6% |

**The honest read: REN lands in the middle of the pack, not at the bottom and not competitive with the top.** It clearly beats two real, standard approaches (naive mean-reversion, ridge regression) — so it isn't uniformly worse than everything simple. But it just as clearly loses to buy-and-hold, risk parity, 60/40, and a stock random forest, all of which are simpler and cheaper to build and run. Notably, **plain risk-parity ties buy-and-hold** as the best performer here — a real, useful data point in itself, and a reminder that in this specific real 2024–2026 window, simple risk-based allocation was very hard to beat by any method tested, REN included. Full table: [`results/full_comparison_table.csv`](results/full_comparison_table.csv).

## 2. Pressure test — block-bootstrap confidence intervals on Sharpe

A single point-estimate Sharpe over one 524-day window can be misleadingly precise-looking. Block bootstrap (2,000 resamples, 21-day blocks to preserve real autocorrelation structure — not i.i.d. day resampling, which would be wrong for daily returns) gives a 90% confidence interval on each model's Sharpe, and the probability the true Sharpe exceeds zero:

| Model | Point Sharpe | 90% CI | P(Sharpe > 0) |
|---|---|---|---|
| Equal-weight buy & hold | 1.61 | **[0.66, 2.88]** | **1.00** |
| Risk parity | 1.61 | **[0.68, 2.92]** | **1.00** |
| Random forest | 1.06 | [0.29, 2.12] | 0.99 |
| 60/40 | 0.92 | [0.04, 2.02] | 0.96 |
| **REN, best composite seed** | 0.69 | **[−0.14, 1.55]** | 0.91 |
| Time-series momentum | 0.54 | [−0.46, 1.72] | 0.82 |
| REN, untrained | 0.50 | [−0.77, 1.66] | 0.71 |
| Ridge regression | −0.01 | [−0.75, 1.19] | 0.60 |

**This is the clearest statistical statement in this report.** Buy-and-hold and risk parity's confidence intervals sit entirely above zero — their edge over this period is statistically robust, not a fluke of one window. REN's best composite seed has a 90% interval that **includes negative Sharpe** ([−0.14, 1.55]) — its positive point estimate is real but not statistically secure the way the two winning baselines' are. This is a materially more rigorous statement than "REN's Sharpe was 0.69" on its own would have implied.

## 3. Pressure test — regime-conditional and calendar sub-period breakdown

Splitting the real test period into volatility terciles (calm / normal / turbulent, by real SPY realized vol) and calendar half-years surfaces two more honest findings:

- **REN's two variants disagree with each other on which regime they're good in.** The composite-trained model does *best* in turbulent markets (Sharpe 1.09 vs. 0.41 calm / 0.32 normal); the untrained model does the *opposite* (Sharpe −0.76 turbulent vs. 0.66 calm / 1.53 normal). Two REN variants showing opposite regime-dependence is evidence this pattern is an idiosyncratic property of a specific training run, not a structural property of the architecture — it should not be read as "REN is good in crises" without a lot more seeds and periods to confirm which (if either) pattern is real.
- **REN's calendar consistency is much worse than the winning baselines'.** REN's best composite seed ranges from Sharpe −2.12 (2026H1) to +7.49 (2026H2, only 28 days — an unstable estimate on its own). Buy-and-hold and risk parity are positive in **every single half-year** in the test period, with far less swing (buy-and-hold: 0.79 to 3.95; risk parity: 0.81 to 3.75). This period-by-period instability is the same overfitting/fragility signature already flagged in the training results (v4), now visible directly in calendar time rather than only in the aggregate train/test gap.

Full regime and sub-period tables for every model: [`results/comparison_and_pressure_tests.json`](results/comparison_and_pressure_tests.json).

## Updated bottom line

REN is not the worst model tested, and it is not competitive with the best. It beats two of six real standard baselines, loses to four (including the two simplest — buy-and-hold and risk parity), and its one apparent strength (positive Sharpe) is both statistically less secure than the winners' (bootstrap CI crosses zero) and inconsistent across the two REN variants' regime behavior and across calendar sub-periods. That is a precise, real, falsifiable characterization of where this architecture currently stands against the market models it would actually have to compete with — not a verdict of "broken," but clearly not yet "better."
