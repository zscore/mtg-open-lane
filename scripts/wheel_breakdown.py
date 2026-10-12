"""Wheel accuracy where it matters for a drafter: python scripts/wheel_breakdown.py MSH [model ...]

Every passed card is ranked within its pack by train-set ATA (1 = the best card you passed, i.e. the
"second-best card in the pack" when you took the best). Reports, per slice, the base wheel rate and
each model's log loss / AUC / ECE, plus a decision view: among cards a model says will wheel with
P >= 0.5 (or <= 0.2), how often they did. Models default to the ALSA baseline, lgbm_all and every
test_predictions_<name>.parquet present. Writes results/{SET}.PremierDraft/wheel_breakdown.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane import ROOT  # noqa: E402
from mtgdata.data import FEATURES  # noqa: E402
from openlane.evaluate import metrics  # noqa: E402

KEY = ["draft_idx", "pack_number", "pick_number", "card"]


def load(set_code: str, models: list[str]):
    out = ROOT / "results" / f"{set_code}.PremierDraft"
    feat = pl.read_parquet(FEATURES / f"{set_code}.PremierDraft" / "test.parquet",
                           columns=KEY + ["ata", "rarity", "pool_share_card", "pool_n", "wheeled"])
    base = pl.read_parquet(out / "test_predictions.parquet", columns=KEY + ["alsa_ata_logreg", "lgbm_all"])
    df = feat.join(base, on=KEY)
    names = ["alsa_ata_logreg", "lgbm_all"]
    extra = models or sorted(p.stem.removeprefix("test_predictions_") for p in out.glob("test_predictions_*.parquet"))
    for m in extra:
        p = out / f"test_predictions_{m}.parquet"
        if p.exists():
            df = df.join(pl.read_parquet(p, columns=KEY + [m]), on=KEY, how="left")
            names.append(m)
    df = df.with_columns(
        pl.col("ata").rank("ordinal").over(KEY[:3]).cast(pl.Int16).alias("rank_in_pack"),
        # on-colour: the drafter already holds a real share of this card's colours (pool_share_card)
        (pl.col("pool_share_card") >= 0.3).alias("on_colour"),
    )
    return df, names


def slice_stats(d: pl.DataFrame, names: list[str]) -> dict:
    y = d["wheeled"].to_numpy()
    r = {"n": int(len(d)), "base_rate": float(y.mean())}
    for m in names:
        p = d[m].to_numpy()
        ok = ~np.isnan(p)
        if ok.sum() < 100 or y[ok].min() == y[ok].max():
            continue
        mm = metrics(y[ok], p[ok])
        hi, lo = p[ok] >= 0.5, p[ok] <= 0.2
        mm.update(said_wheels=float(hi.mean()), said_wheels_hit=float(y[ok][hi].mean()) if hi.any() else None,
                  said_gone=float(lo.mean()), said_gone_wheeled=float(y[ok][lo].mean()) if lo.any() else None)
        r[m] = {k: round(v, 4) if isinstance(v, float) else v for k, v in mm.items()}
    return r


def main(set_code="MSH", *models):
    df, names = load(set_code, list(models))
    df = df.drop_nulls(names)
    print(f"{set_code}: {df.height:,} passed cards, models {names}")
    res = {"all": slice_stats(df, names)}
    for k in range(1, 6):
        res[f"rank{k}"] = slice_stats(df.filter(pl.col("rank_in_pack") == k), names)
    res["rank6+"] = slice_stats(df.filter(pl.col("rank_in_pack") >= 6), names)
    for k in range(1, 4):
        res[f"rank{k}_oncolour_p2p3"] = slice_stats(
            df.filter((pl.col("rank_in_pack") == k) & pl.col("on_colour") & (pl.col("pack_number") > 0)), names)
    res["uncertain_by_alsa"] = slice_stats(df.filter(pl.col("alsa_ata_logreg").is_between(0.2, 0.8)), names)
    for p in range(6):
        res[f"pick{p}_rank1-3"] = slice_stats(df.filter((pl.col("pick_number") == p)
                                                         & (pl.col("rank_in_pack") <= 3)), names)
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "wheel_breakdown.json"
    out.write_text(json.dumps(res, indent=1))
    cols = names
    print(f"{'slice':24s} {'n':>9s} {'base':>6s} " + " ".join(f"{m[:22]:>22s}" for m in cols))
    for s, r in res.items():
        ll = " ".join(f"{r[m]['logloss']:.4f}/{r[m]['auc']:.3f}".rjust(22) if m in r else " " * 22 for m in cols)
        print(f"{s:24s} {r['n']:9,d} {r['base_rate']:6.3f} {ll}")


if __name__ == "__main__":
    main(*sys.argv[1:])
