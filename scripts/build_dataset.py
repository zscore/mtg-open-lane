"""Build the wheel-prediction dataset for a set: python scripts/build_dataset.py MSH"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import ROOT, load_tensors  # noqa: E402
from openlane.features import build_examples, fit_card_stats, wheel_tensors  # noqa: E402

CHUNK = 4000


def main(set_code: str, event: str = "PremierDraft"):
    t0 = time.time()
    t = load_tensors(set_code, event)
    D = len(t.draft_id)
    print(f"{set_code} {event}: {D} complete drafts, {len(t.names)} cards  ({time.time()-t0:.0f}s)")

    passed, back = wheel_tensors(t)
    # drafts are sorted by time -> chronological split
    cut1, cut2 = int(D * 0.70), int(D * 0.85)
    splits = {"train": np.arange(0, cut1), "val": np.arange(cut1, cut2), "test": np.arange(cut2, D)}
    stats = fit_card_stats(t, splits["train"], passed, back)

    out = ROOT / "data" / "features" / f"{set_code}.{event}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({"names": t.names, "meta": t.meta, "stats": stats,
                     "draft_id": t.draft_id, "draft_time": t.draft_time, "splits": splits}, f)

    for name, idx in splits.items():
        parts = [build_examples(t, stats, passed, back, idx[i:i + CHUNK]) for i in range(0, len(idx), CHUNK)]
        df = pl.concat(parts)
        df.write_parquet(out / f"{name}.parquet")
        print(f"  {name}: {len(idx)} drafts -> {df.height:,} examples, wheel rate {df['wheeled'].mean():.3f}"
              f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(*sys.argv[1:])
