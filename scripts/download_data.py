"""
Download REAL historical OHLCV data for the RecWM/REN universe.

This is the ONLY place real market data enters the system. Everything
downstream (belief fields, influence kernel, equilibrium engine, backtests)
is built on top of what this script fetches -- no synthetic price data
is ever substituted for it.

Universe: 16 liquid, publicly-traded instruments spanning equity indices,
sectors, single names, rates, commodities and volatility, so the belief
field has genuine cross-asset structure to work with.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

UNIVERSE = [
    "SPY", "QQQ", "IWM",        # broad equity
    "XLF", "XLK", "XLE", "XLU", # sectors: financials, tech, energy, utilities
    "AAPL", "MSFT", "NVDA", "TSLA",  # single names
    "TLT", "GLD", "USO",        # rates, gold, oil
    "UUP", "VXX",               # dollar, vol
]

START = "2015-01-01"
END = None  # up to today


def download(universe=UNIVERSE, start=START, end=END, out_path=None):
    frames = {}
    failed = []
    for i, ticker in enumerate(universe):
        for attempt in range(3):
            try:
                df = yf.download(
                    ticker, start=start, end=end, auto_adjust=True,
                    progress=False, threads=False,
                )
                if df is None or df.empty:
                    raise ValueError("empty frame")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                frames[ticker] = df[["Open", "High", "Low", "Close", "Volume"]]
                print(f"[{i+1}/{len(universe)}] {ticker}: {len(df)} rows "
                      f"{df.index[0].date()} -> {df.index[-1].date()}")
                break
            except Exception as e:
                print(f"  retry {ticker} ({attempt+1}/3): {e}")
                time.sleep(2)
        else:
            failed.append(ticker)

    if failed:
        print(f"FAILED to download: {failed}", file=sys.stderr)

    if not frames:
        raise RuntimeError("No data downloaded at all.")

    panel = pd.concat(frames, axis=1)  # columns: (ticker, field)
    panel = panel.sort_index()
    panel = panel.ffill().dropna(how="all")

    out_path = out_path or (DATA_DIR / "market_panel.parquet")
    panel.to_parquet(out_path)
    print(f"\nSaved panel: {panel.shape} -> {out_path}")
    return panel


if __name__ == "__main__":
    download()
