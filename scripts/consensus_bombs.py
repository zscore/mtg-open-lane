"""Join the hand-curated community bomb list (results/bombs/consensus_bombs.csv) to our 17Lands ratings.

For each listed card, adds its rank within the set by shrunk normalized GIH WR (Premier Draft, from the
bombs notebook's cached results/bombs/{set}.PremierDraft.cards.parquet), the rating in pp, the robust z,
and whether it clears the notebook's Tukey-fence bomb cutoff (z > 2.70). Also flags our top-10 cards that
the community list doesn't mention. Sources and method: results/bombs/consensus_bombs_sources.md.
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "bombs"
FENCE_Z = 0.6745 + 1.5 * 1.349


def ranked(s):
    d = pl.read_parquet(OUT / f"{s}.PremierDraft.cards.parquet")
    med, iqr = d["norm_eb"].median(), d["norm_eb"].quantile(0.75) - d["norm_eb"].quantile(0.25)
    return (d.sort("norm_eb", descending=True).with_row_index("data_rank", 1)
            .with_columns(((pl.col("norm_eb") - med) / (iqr / 1.349)).alias("z"), pl.lit(d.height).alias("set_cards")))


def match(card, names):
    # exact name, else the front face of a double-faced card ("Elesh Norn" -> "Elesh Norn // ...")
    if card in names:
        return card
    hits = [n for n in names if n.startswith(card + " //") or n.startswith(card + ",")]
    return hits[0] if len(hits) == 1 else None


cons = pl.read_csv(OUT / "consensus_bombs.csv")
rows, extra = [], []
for s in cons["set"].unique(maintain_order=True):
    r = ranked(s)
    names = set(r["name"])
    listed = cons.filter(pl.col("set") == s)
    for row in listed.iter_rows(named=True):
        n = match(row["card"], names)
        hit = r.filter(pl.col("name") == n) if n else None
        rows.append({**row, "data_name": n,
                     "data_rank": hit["data_rank"][0] if n else None, "set_cards": r["set_cards"][0],
                     "norm_eb_pp": round(hit["norm_eb"][0], 1) if n else None,
                     "z": round(hit["z"][0], 2) if n else None,
                     "data_bomb": bool(hit["z"][0] > FENCE_Z) if n else None})
    matched = {x["data_name"] for x in rows if x["set"] == s}
    for t in r.head(10).iter_rows(named=True):
        if t["name"] not in matched:
            extra.append({"set": s, "card": t["name"], "rarity": t["rarity"], "data_rank": t["data_rank"],
                          "norm_eb_pp": round(t["norm_eb"], 1), "z": round(t["z"], 2)})

out = pl.DataFrame(rows)
out.write_csv(OUT / "consensus_bombs_vs_data.csv")
pl.DataFrame(extra).write_csv(OUT / "data_top10_not_in_consensus.csv")
missing = out.filter(pl.col("data_name").is_null())
print(f"{out.height} consensus cards, {missing.height} not found in data:", missing.select("set", "card").rows())
print(f"{len(extra)} data top-10 cards not on the consensus list")
