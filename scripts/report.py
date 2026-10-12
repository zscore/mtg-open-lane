"""Combine all test predictions, add an ensemble, and write the results table: python scripts/report.py MSH

readme=0 writes summary.json only, leaving the README's MSH table alone.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane import ROOT  # noqa: E402
from openlane.evaluate import metrics, metrics_by  # noqa: E402

KEYS = ["draft_idx", "pack_number", "pick_number", "card"]


def main(set_code: str, event: str = "PremierDraft", *kv: str):
    opts = {"readme": 1} | {k: int(v) for k, v in (a.split("=") for a in kv)}
    out = ROOT / "results" / f"{set_code}.{event}"
    df = pl.read_parquet(out / "test_predictions.parquet")
    nn_path = out / "test_predictions_wheelnet.parquet"
    if nn_path.exists():
        nn = pl.read_parquet(nn_path).drop("wheeled")
        n0 = df.height
        df = df.join(nn, on=KEYS, how="inner")
        assert df.height == n0, "wheelnet and GBM test rows must match exactly"
        df = df.with_columns(((pl.col("lgbm_all") + pl.col("wheelnet")) / 2).alias("ensemble"))

    models = [c for c in df.columns if c not in KEYS + ["wheeled"]]
    y = df["wheeled"].to_numpy()
    base = metrics(y, df["alsa_ata_logreg"].to_numpy())["logloss"]
    rows = []
    for m in models:
        r = metrics(y, df[m].to_numpy())
        by_pack = metrics_by(df, m, ["pack_number"])["logloss"].to_list()
        rows.append({"model": m, **r, "vs_alsa": 1 - r["logloss"] / base,
                     **{f"ll_pack{k + 1}": v for k, v in enumerate(by_pack)}})
    table = pl.DataFrame(rows)
    print(table)
    json.dump(table.to_dicts(), open(out / "summary.json", "w"), indent=1)

    md = ["| model | logloss | Brier | AUC | ECE | logloss vs ALSA baseline | logloss P1 / P2 / P3 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r['model']}` | {r['logloss']:.4f} | {r['brier']:.4f} | {r['auc']:.4f} | "
                  f"{r['ece']:.4f} | {r['vs_alsa']:+.1%} | "
                  f"{r['ll_pack1']:.3f} / {r['ll_pack2']:.3f} / {r['ll_pack3']:.3f} |")
    if not opts["readme"]:
        return
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = "<!-- results -->\n" + "\n".join(md) + "\n<!-- /results -->"
    if "RESULTS_TABLE" in text:
        text = text.replace("RESULTS_TABLE", block)
    else:
        text = re.sub(r"<!-- results -->.*?<!-- /results -->", block, text, flags=re.S)
    readme.write_text(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
