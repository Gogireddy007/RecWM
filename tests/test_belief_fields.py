import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from ren.belief_fields import BeliefField, TIMESCALES, AGENT_TYPES

DATA = Path(__file__).resolve().parent.parent / "data" / "market_panel.parquet"


def load_panel():
    panel = pd.read_parquet(DATA)
    tickers = sorted({c[0] for c in panel.columns})
    return panel, tickers


def test_belief_field_shapes_and_finiteness():
    panel, tickers = load_panel()
    bf = BeliefField(tickers=tickers, embed_dim=32)
    bf.update(panel.iloc[:400])
    mat = bf.raw_matrix()
    n_expected_rows = len(AGENT_TYPES) * len(TIMESCALES)
    assert mat.shape == (n_expected_rows, len(tickers))
    assert np.all(np.isfinite(mat))
    for key, st in bf.states.items():
        assert st.embedding.shape == (32,)
        assert np.all(np.isfinite(st.embedding))


def test_belief_field_recurrence_updates_smoothly():
    panel, tickers = load_panel()
    bf = BeliefField(tickers=tickers, embed_dim=32, ema_halflife=5)
    bf.update(panel.iloc[:400])
    emb1 = bf.states[("momentum", "short")].embedding.copy()
    bf.update(panel.iloc[:401])
    emb2 = bf.states[("momentum", "short")].embedding.copy()
    # should move, but not explode (EMA-bounded)
    delta = np.linalg.norm(emb2 - emb1)
    assert 0 <= delta < 5.0


if __name__ == "__main__":
    test_belief_field_shapes_and_finiteness()
    test_belief_field_recurrence_updates_smoothly()
    print("belief_fields tests passed")
