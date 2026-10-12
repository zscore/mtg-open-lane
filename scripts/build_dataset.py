"""Build the wheel-prediction dataset for a set: python scripts/build_dataset.py MSH"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.data import FEATURES, load_tensors  # noqa: E402
from openlane import ROOT  # noqa: E402
from openlane.features import build_examples, fit_card_stats, wheel_tensors  # noqa: E402

CHUNK = 4000


def main(set_code: str, event: str = "PremierDraft"):
    t0 = time.time()
    t = load_tensors(set_code, event)
    D = len(t.draft_id)
    print(f"{set_code} {event}: {D} complete drafts, {len(t.names)} cards, {t.n_picks}-card packs"
          f"{', P1P1 missing' if t.p1p1_missing else ''}  ({time.time()-t0:.0f}s)")

    passed, back = wheel_tensors(t)
    # drafts are sorted by time -> chronological split
    cut1, cut2 = int(D * 0.70), int(D * 0.85)
    splits = {"train": np.arange(0, cut1), "val": np.arange(cut1, cut2), "test": np.arange(cut2, D)}
    stats = fit_card_stats(t, splits["train"], passed, back)

    out = FEATURES / f"{set_code}.{event}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump({"names": t.names, "meta": t.meta, "stats": stats, "draft_id": t.draft_id,
                     "draft_time": t.draft_time, "splits": splits, "n_picks": t.n_picks,
                     "p1p1_missing": t.p1p1_missing}, f)

    for name, idx in splits.items():
        # stream chunks to disk: a big set's train split is ~6GB in memory
        writer, n, n_wheeled = None, 0, 0
        for i in range(0, len(idx), CHUNK):
            tbl = build_examples(t, stats, passed, back, idx[i:i + CHUNK]).to_arrow()
            writer = writer or pq.ParquetWriter(out / f"{name}.parquet", tbl.schema)
            writer.write_table(tbl)
            n, n_wheeled = n + tbl.num_rows, n_wheeled + int(np.sum(tbl["wheeled"].to_numpy()))
        writer.close()
        print(f"  {name}: {len(idx)} drafts -> {n:,} examples, wheel rate {n_wheeled / n:.3f}"
              f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(*sys.argv[1:])
