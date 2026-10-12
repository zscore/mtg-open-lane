"""Prince vs pauper: how much of the spread in deck card value comes from rares?

Uses each card's win-rate value from scripts/card_impact.py (`impact_eb`, pp of game win rate per copy vs an average
card of the same deck). For every Premier Draft deck (one row per game):

    V  = sum over the deck's cards of copies * impact          (the deck's total card value, pp of game WR)
    R  = the part of V from rares and mythics
    CU = the part of V from commons and uncommons            (V = R + CU + the few cards with other rarities)

`rare_share` = Cov(R, V) / Var(V): the share of the between-deck variance in card value that comes from rares and
mythics. The shares of all rarity groups add to 1. A prince format has a high rare_share: decks differ mainly in which
rares they got. A pauper format has a low one: decks differ mainly in their commons and uncommons.

Cards without an impact estimate (fewer than 500 decks) count as 0. Bonus-sheet cards count under their printed
rarity. Output: results/bombs/prince_index.csv, one row per set.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from openlane.deckstrength import aligned
from openlane.games import game_path

OUT = ROOT / "results" / "bombs"
RARE = ["rare", "mythic"]
CU = ["common", "uncommon"]


def run_set(s, imp):
    lf = pl.scan_parquet(game_path(s, "PremierDraft"))
    names = [c[len("deck_"):] for c in lf.collect_schema().names() if c.startswith("deck_")]
    g = lf.select([f"deck_{n}" for n in names]).collect()
    deck = aligned(g, "deck_", names).astype(np.float32)
    del g
    d = imp.filter(pl.col("set") == s)
    val = dict(zip(d["name"], d["impact"])); rar = dict(zip(d["name"], d["rarity"]))
    v = np.array([val.get(n, 0.0) for n in names], np.float32)
    is_r = np.array([rar.get(n) in RARE for n in names]); is_cu = np.array([rar.get(n) in CU for n in names])
    V, R, C = deck @ v, deck @ (v * is_r), deck @ (v * is_cu)
    n_r = deck @ is_r.astype(np.float32)
    var = V.var()
    return {"set": s, "games": len(V), "sd deck value": np.sqrt(var), "sd rare part": R.std(), "sd C/U part": C.std(),
            "rare_share": np.cov(R, V)[0, 1] / var, "cu_share": np.cov(C, V)[0, 1] / var,
            "rares per deck": n_r.mean()}


if __name__ == "__main__":
    imp = pl.read_csv(OUT / "card_impact.csv").rename({"impact_eb": "impact"})
    rows = []
    for s in imp["set"].unique(maintain_order=True):
        rows.append(run_set(s, imp))
        print(f"{s}: rare share {rows[-1]['rare_share']:.2f}", flush=True)
    pl.DataFrame(rows).write_csv(OUT / "prince_index.csv")
