"""Download + convert + sanity-check draft files: python scripts/download.py [SET ...]

Checks per file: pack size at pick 0, picks per draft, and the wheel subset property
(pack at p+8 is contained in pack at p minus the user's pick) on a sample of drafts.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import PARQUET, download  # noqa: E402

ALL_SETS = ["LCI", "MKM", "OTJ", "BLB", "DSK", "FDN", "DFT", "TDM", "FIN", "EOE", "TLA", "TMT",
            "SOS", "ECL", "MSH"]


def check(path: Path, sample: int = 2000) -> dict:
    lf = pl.scan_parquet(path)
    names = [c for c in lf.collect_schema().names() if c.startswith("pack_card_")]
    base = lf.select("draft_id", "pack_number", "pick_number", "pick").collect()
    per_draft = base.group_by("draft_id").len()
    drafts = per_draft["draft_id"].unique().sort()[:sample]
    df = lf.filter(pl.col("draft_id").is_in(drafts.implode())).select(
        "draft_id", "pack_number", "pick_number", "pick", *names).collect()
    X = df.select(names).to_numpy().astype(np.int16)
    size0 = X[df["pick_number"].to_numpy() == 0].sum(1)
    idx = {n[len("pack_card_"):]: i for i, n in enumerate(names)}
    key = {(d, k, p): i for i, (d, k, p) in enumerate(df.select("draft_id", "pack_number", "pick_number").iter_rows())}
    ok = bad = 0
    for (d, k, p), i in key.items():
        j = key.get((d, k, p + 8))
        if p > 5 or j is None:
            continue
        passed = X[i].copy()
        passed[idx[df["pick"][i]]] -= 1
        ok, bad = (ok + 1, bad) if (X[j] <= passed).all() else (ok, bad + 1)
    return {"drafts": per_draft.height, "rows": base.height,
            "pct_42_picks": round(float((per_draft["len"] == 42).mean()), 4),
            "pack_size_p0": float(np.median(size0)), "cards": len(names),
            "subset_ok": ok, "subset_bad": bad, "MB": round(path.stat().st_size / 1e6)}


def main(sets):
    rows = []
    for s in sets:
        for e in ["PremierDraft", "TradDraft"]:
            path = download(s, e)
            rows.append({"set": s, "event": e, **check(path)})
            print(rows[-1], flush=True)
    summary = pl.DataFrame(rows)
    summary.write_csv(PARQUET / "inventory.csv")
    with pl.Config(tbl_rows=100, tbl_cols=20):
        print(summary)


if __name__ == "__main__":
    main(sys.argv[1:] or ALL_SETS)
