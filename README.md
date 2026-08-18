# REN / RecWM — a real, runnable implementation

This is a genuine, executable implementation of the ten inventions
described in the RecWM thesis
(https://bhavith-chandra.github.io/RecWM-Thesis/), built to produce
measured results rather than restate the thesis's design targets as
findings.

See [`REPORT.md`](REPORT.md) for the full results writeup, with every
number traceable to a script in `scripts/` and a JSON/CSV file in
`results/`.

## Layout

- `ren/` — the ten inventions as real Python/PyTorch modules.
- `scripts/` — the benchmark/backtest drivers that produced every
  number in `REPORT.md`.
- `data/market_panel.parquet` — real OHLCV data for 16 liquid
  instruments, 2015–2026, downloaded via `scripts/download_data.py`.
- `results/` — raw JSON/CSV output from every benchmark run.
- `tests/` — correctness tests (resolvent-vs-Neumann-series check,
  HYDRA-vs-known-bistability check, belief field sanity checks, etc).

## Reproducing

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib yfinance scikit-learn torch pyarrow

python scripts/download_data.py
python scripts/run_atlas_benchmark.py
python scripts/run_seismograph.py contractive
python scripts/run_seismograph.py sensitive
python scripts/run_remaining_inventions.py
python scripts/run_full_backtest.py
```

## What is real vs. what is a disclosed modeling choice

Every latency, convergence, spectral-radius, and backtest number is
measured on this machine from real market data — nothing in this
repository is copy-pasted from the thesis's design targets. What IS a
disclosed choice, because no public dataset of hedge-fund beliefs or
live trading exists: the five "agent types" are explicit, documented
functions of real price/volume data (see `ren/belief_fields.py`), and
the core operator's weights are either randomly seeded (testing the
raw numerics) or trained on real historical P&L with a disclosed,
simplified truncated-unroll gradient method (see `ren/training.py`).
