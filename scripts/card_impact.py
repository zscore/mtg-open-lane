"""What is each card worth in game win rate? Per-card regression on Premier Draft games.

For every set and every rated card i, a linear probability model on games:

    won (0/1) = beta_i * copies_i + gamma_i * score_{-i} + controls + e

* copies_i  copies of card i in the main deck.
* score_{-i}  mean cross-fitted GIH WR rating (shrunk, normalized, pp) of the deck's *other* rated cards,
            i.e. the bombs notebook's deck score with card i left out. Holding it fixed means beta_i compares
            card i with an average card of the same deck, so beta_i is "pp of game win rate from having
            this card in the deck instead of a typical card of that deck".
* controls  spec A: none (intercept only), matching the deck-score slope the conversion uses.
            spec B: the drafter's overall game win rate bucket (17Lands `user_game_win_rate_bucket`,
            one dummy per bucket, plus unknown), main colors (one dummy per pair/triple), on the play, and
            the number of rated cards in the deck. The skill control is what separates "this card wins"
            from "good drafters take this card".

Ratings for score_{-i} are cross-fitted exactly as in the bombs notebook (draft_id hash halves, seed 7), and
computed here from the game files, so the script doesn't need the notebook to have run first. Games
come from drafts where every game has cover >= 0.9, the notebook's sample. Standard errors are clustered by
draft_id. beta_i is then shrunk within set x rarity (empirical Bayes, as for the ratings) to `impact_eb`;
bonus-sheet reprints share one prior pooled over all sets instead.

Also written: `implied`, what the per-set deck-score slope says the card is worth,
    slope_set * (norm_eb_i - mean deck score) / mean rated cards per deck,
the "convert rating units to win rate" shortcut, for comparison with the direct estimate.

Output: results/bombs/card_impact.csv (one row per set x card), results/bombs/card_impact_sets.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from openlane.data import card_metadata
from openlane.deckstrength import aligned, flag
from openlane.games import game_path

OUT = ROOT / "results" / "bombs"
SETS = ["NEO", "ONE", "MOM", "WOE", "LCI", "MKM", "OTJ", "BLB", "DSK", "FDN", "DFT", "TDM", "FIN", "EOE", "TLA",
        "ECL", "TMT", "SOS", "MSH"]
BASICS = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
MIN_GIH = 200
CHUNK = 20


def eb_shrink(x, se2, groups):
    out = np.empty_like(x)
    for g in np.unique(groups):
        m = groups == g
        w = 1 / se2[m]
        mu = (w * x[m]).sum() / w.sum()
        tau2 = max(x[m].var() - se2[m].mean(), 0.25)
        out[m] = mu + tau2 / (tau2 + se2[m]) * (x[m] - mu)
    return out


def half_ratings(won, seen, deck, rarity, keep):
    # shrunk normalized GIH WR (pp) per card from the games in `keep`; NaN for cards below MIN_GIH (as the notebook)
    w, s, d = won[keep], seen[keep], deck[keep] > 0
    unseen = d & ~s
    n_gih, n_gns = s.sum(0), unseen.sum(0)
    ok = (n_gih >= MIN_GIH) & (n_gns >= MIN_GIH) & (rarity != "basic")
    p = s[w].sum(0)[ok] / n_gih[ok]
    x = (p - w.mean()) * 100
    out = np.full(len(rarity), np.nan)
    out[ok] = eb_shrink(x, p * (1 - p) / n_gih[ok] * 1e4, rarity[ok])
    return out


def dummies(s: pl.Series, min_n=200):
    v = s.cast(pl.Utf8).fill_null("?")
    keep = v.value_counts().filter(pl.col("count") >= min_n)[v.name].to_list()
    v = v.to_numpy()
    lev = sorted(set(keep))[1:]   # first level is the baseline
    return np.stack([(v == l) for l in lev], 1).astype(np.float64) if lev else np.zeros((len(v), 0))


def run_set(s):
    t0 = time.time()
    lf = pl.scan_parquet(game_path(s, "PremierDraft"))
    cols = lf.collect_schema().names()
    names = [c[len("drawn_"):] for c in cols if c.startswith("drawn_")]
    g = lf.select("draft_id", "won", "on_play", "main_colors", "user_game_win_rate_bucket",
                  *[c for c in cols if c.startswith(("deck_", "drawn_", "opening_hand_"))]).collect()
    deck = aligned(g, "deck_", names)
    seen = (aligned(g, "opening_hand_", names) + aligned(g, "drawn_", names)) > 0
    meta = g.select("draft_id", "won", "on_play", "main_colors", "user_game_win_rate_bucket")
    del g
    cm = card_metadata(names, s)
    rarity = cm["rarity"].to_numpy().astype(object)
    rarity[np.isin(names, BASICS)] = "basic"
    won = flag(meta["won"])
    half = meta["draft_id"].hash(seed=7).to_numpy() % 2
    norm_eb = half_ratings(won, seen, deck, rarity, np.ones(len(won), bool))   # full-data ratings, for `implied`

    # cross-fitted ratings: rows in half h use ratings from half 1-h
    rh = np.stack([half_ratings(won, seen, deck, rarity, half != h) for h in (0, 1)])   # [2, C]
    del seen
    rated = ~np.isnan(rh)
    rz = np.nan_to_num(rh).astype(np.float32)
    nonbasic = ~np.isin(names, BASICS)
    S = np.zeros(len(won), np.float64); K = np.zeros(len(won), np.float64); NB = deck[:, nonbasic].sum(1)
    for h in (0, 1):
        m = half == h
        d = deck[m].astype(np.float32)
        S[m] = d @ rz[h]
        K[m] = d @ rated[h].astype(np.float32)
    cover = K / np.maximum(NB, 1)
    keep = (pl.DataFrame({"d": meta["draft_id"], "c": cover}).select(pl.col("c").min().over("d") >= 0.9)
            .to_series().to_numpy()) & (K > 0)
    deck, S, K, half, won = deck[keep], S[keep], K[keep], half[keep], won[keep].astype(np.float64)
    meta = meta.filter(pl.Series(keep))
    _, cl = np.unique(meta["draft_id"].to_numpy(), return_inverse=True)
    G = len(won)
    score = S / K
    slope = np.cov(score, won)[0, 1] / score.var() * 100

    # controls for spec B
    C = np.hstack([np.ones((G, 1)), dummies(meta["user_game_win_rate_bucket"]), dummies(meta["main_colors"]),
                   flag(meta["on_play"])[:, None].astype(np.float64), K[:, None]])
    P = np.linalg.pinv(C.T @ C)
    C32, P32 = C.astype(np.float32), P.astype(np.float32)

    def resid(X, spec):
        # residualize the columns of X on the spec's controls (Frisch-Waugh-Lovell)
        if spec == "A":
            return X - X.mean(0)
        return X - C32 @ (P32 @ (C32.T @ X))

    y = {sp: resid(won[:, None].astype(np.float32), sp)[:, 0].astype(np.float64) for sp in "AB"}
    cand = np.flatnonzero(~np.isnan(norm_eb)).tolist()
    res = {sp: {} for sp in "AB"}
    for c0 in range(0, len(cand), CHUNK):
        J = cand[c0:c0 + CHUNK]
        N = deck[:, J].astype(np.float32)
        Nr = N * rated[:, J][half]                     # copies that count toward score (rated in this row's half)
        rest = K[:, None] - Nr
        Sm = np.where(rest > 0, (S[:, None] - Nr * rz[:, J][half]) / np.maximum(rest, 1),
                      score.mean()).astype(np.float32)   # score_{-i}; set mean if card i is the deck's only rated card
        for sp in "AB":
            Nt, St = resid(N, sp).astype(np.float64), resid(Sm, sp).astype(np.float64)
            # 2-regressor OLS per column, then FWL again: n residualized on s
            g_ns = (Nt * St).sum(0) / (St * St).sum(0)
            nh = Nt - St * g_ns
            beta = (nh * y[sp][:, None]).sum(0) / (nh * nh).sum(0)
            gam = ((St * y[sp][:, None]).sum(0) - beta * (Nt * St).sum(0)) / (St * St).sum(0)
            e = y[sp][:, None] - Nt * beta - St * gam
            sc = np.stack([np.bincount(cl, nh[:, k] * e[:, k]) for k in range(len(J))], 1)
            se = np.sqrt((sc ** 2).sum(0)) / (nh * nh).sum(0)
            for k, j in enumerate(J):
                res[sp][j] = (beta[k] * 100, se[k] * 100)
        del N, Nr, Sm
    out = pl.DataFrame({
        "set": s, "name": [names[j] for j in cand], "rarity": rarity[cand].tolist(),
        "games_in_deck": (deck[:, cand] > 0).sum(0).astype(np.int64),
        "beta_A": [res["A"][j][0] for j in cand], "se_A": [res["A"][j][1] for j in cand],
        "beta_B": [res["B"][j][0] for j in cand], "se_B": [res["B"][j][1] for j in cand],
        "norm_eb": norm_eb[cand]})
    out = out.with_columns((slope * (pl.col("norm_eb") - score.mean()) / K.mean()).alias("implied"),
                           pl.col("games_in_deck").truediv(G).alias("deck_share"),
                           pl.Series("is_bonus", cm["is_bonus"].to_numpy()[cand]))
    out = out.filter(pl.col("games_in_deck") >= 500)
    info = {"set": s, "games": G, "drafts": int(cl.max() + 1), "slope": slope, "mean_score": score.mean(),
            "mean_rated_cards": K.mean(), "cards": out.height}
    print(f"{s}: {G:,} games, {out.height} cards, slope {slope:.2f}, {time.time() - t0:.0f}s", flush=True)
    return out, info


if __name__ == "__main__":
    sets = sys.argv[1:] or SETS
    outs, infos = [], []
    for s in sets:
        o, i = run_set(s)
        outs.append(o); infos.append(i)
    suffix = "" if sets == SETS else "_" + "_".join(sets)
    # Shrink within set x rarity, except bonus-sheet reprints: they're few per set and seen in few decks, so a
    # per-set prior is unstable and the set's mythic mean would carry them. They share one prior across all sets.
    out = pl.concat(outs).with_columns(
        pl.when(pl.col("is_bonus")).then(pl.lit("bonus")).otherwise(pl.col("set") + "|" + pl.col("rarity")).alias("prior"))
    pr = out["prior"].to_numpy()
    out = out.with_columns(
        pl.Series("impact_eb", eb_shrink(out["beta_B"].to_numpy(), out["se_B"].to_numpy() ** 2, pr)),
        pl.Series("impact_eb_A", eb_shrink(out["beta_A"].to_numpy(), out["se_A"].to_numpy() ** 2, pr))).drop("prior")
    out.write_csv(OUT / f"card_impact{suffix}.csv")
    pl.DataFrame(infos).write_csv(OUT / f"card_impact_sets{suffix}.csv")
