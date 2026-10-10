"""Deep vs shallow: how far down do decks have to reach for their last playables?

Uses each card's win-rate value from scripts/card_impact.py (`impact_eb`, pp of game win rate per copy vs an average
card of the same deck). Lands are left out; "cards" below means nonland cards with an impact estimate. For every
Premier Draft deck (one row per game):

    filler gap = mean value of the deck's cards - mean value of its 3 weakest card copies
                 How much worse the cards a drafter had to play are than the deck's typical card. Both terms are
                 values in the same deck, so the set's zero point cancels. Small = deep: even the last cards in are
                 close to the rest. Large = shallow: decks fill out with cards well below their average.
    cuts       = copies of on-color cards left in the pool (drafted but not in the deck) that are worth at least
                 the deck's 3rd-weakest card copy. On-color means the card's colors are within the deck's main +
                 splash colors (colorless counts). Deep formats leave real playables on the bench.

The pool comes from the draft logs (picks per draft_id), so cuts exist only for sets with draft logs (not NEO, ONE)
and drafts present in both files. Packs have 13 to 15 picks depending on the set, and a bigger pool has a bigger
bench, so the pool is cut to the first PICKS picks of each pack (39 cards). TLA and TMT logs lack P1P1, so their pools
miss one more card. Bench copies are clipped at 0.

`filler gap` is in pp of game win rate, so it grows with the spread of card values in the set. `filler gap (sd)`
divides it by the SD of value over the set's played card copies, which leaves how far into the tail decks reach. Output: results/bombs/depth_index.csv, one row per set.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from openlane.data import card_metadata
from openlane.deckstrength import aligned
from openlane.games import game_path

OUT = ROOT / "results" / "bombs"
COLORS = list("WUBRG")
K = 3          # weakest card copies per deck
PICKS = 13     # pool = the first 13 picks of each pack, so pools are the same size in every set
CHUNK = 100_000


def weakest(D, v):
    # mean value of the K weakest copies per row, and the value of the K-th weakest copy
    vals = np.where(D > 0, v[None, :], np.inf)
    o = np.argsort(vals, axis=1)
    vs, cs = np.take_along_axis(vals, o, 1), np.take_along_axis(D, o, 1)
    cum = cs.cumsum(1)
    take = np.clip(K - (cum - cs), 0, cs)
    mean = (take * np.where(take > 0, vs, 0)).sum(1) / K
    kth = vs[np.arange(len(D)), (cum >= K).argmax(1)]
    return mean, kth


def pools(s, names):
    p = ROOT / "data" / "parquet" / f"{s}.PremierDraft.parquet"
    if not p.exists():
        return None
    idx = {n: j for j, n in enumerate(names)}
    d = pl.scan_parquet(p).filter(pl.col("pick_number") < PICKS).select("draft_id", "pick").collect()
    ids, row = np.unique(d["draft_id"].to_numpy(), return_inverse=True)
    col = np.array([idx.get(n, -1) for n in d["pick"]])
    P = np.zeros((len(ids), len(names)), np.float32)
    np.add.at(P, (row[col >= 0], col[col >= 0]), 1)
    return dict(zip(ids, range(len(ids)))), P


def run_set(s, imp):
    lf = pl.scan_parquet(game_path(s, "PremierDraft"))
    names = [c[len("deck_"):] for c in lf.collect_schema().names() if c.startswith("deck_")]
    g = lf.select("draft_id", "main_colors", "splash_colors", *[f"deck_{n}" for n in names]).collect()
    d = imp.filter(pl.col("set") == s)
    val = dict(zip(d["name"], d["impact"]))
    meta = card_metadata(names, s)
    keep = np.array([n in val for n in names]) & ~meta["types"].str.contains("Land").to_numpy()
    names = [n for n, k in zip(names, keep) if k]
    v = np.array([val[n] for n in names], np.float32)
    ccol = meta.filter(pl.Series(keep)).select(COLORS).to_numpy().astype(np.float32)   # [C, 5]
    deck = aligned(g, "deck_", names).astype(np.float32)
    dcol = np.stack([(g["main_colors"].fill_null("") + g["splash_colors"].fill_null("")).str.contains(c).to_numpy()
                     for c in COLORS], 1)
    pool = pools(s, names)
    if pool is not None:
        pidx, P = pool
        prow = np.array([pidx.get(i, -1) for i in g["draft_id"]])
    gap, cuts = [], []
    for c0 in range(0, len(deck), CHUNK):
        D = deck[c0:c0 + CHUNK]
        n = D.sum(1)
        ok = n >= K + 5
        D = D[ok]
        mean = D @ v / D.sum(1)
        wmean, kth = weakest(D, v)
        gap.append(mean - wmean)
        if pool is None:
            continue
        pr = prow[c0:c0 + CHUNK][ok]
        has = pr >= 0
        bench = np.maximum(P[pr[has]] - D[has], 0)
        on = (ccol @ (~dcol[c0:c0 + CHUNK][ok][has]).T.astype(np.float32)).T == 0              # [rows, C]
        cuts.append((bench * on * (v[None, :] >= kth[has, None])).sum(1))
    gap = np.concatenate(gap)
    w = deck.sum(0)
    sd_played = np.sqrt(np.average((v - np.average(v, weights=w)) ** 2, weights=w))
    row = {"set": s, "decks": len(gap), "filler gap": gap.mean(), "sd played": sd_played,
           "filler gap (sd)": gap.mean() / sd_played}
    if pool is not None:
        cuts = np.concatenate(cuts)
        row |= {"cuts": cuts.mean(), "P(cuts ≥1)": (cuts >= 1).mean(), "decks with pool": len(cuts)}
    return row


if __name__ == "__main__":
    imp = pl.read_csv(OUT / "card_impact.csv").rename({"impact_eb": "impact"})
    rows = []
    for s in imp["set"].unique(maintain_order=True):
        rows.append(run_set(s, imp))
        print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(OUT / "depth_index.csv")
