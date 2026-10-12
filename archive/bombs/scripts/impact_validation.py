"""Checks on card impact (scripts/card_impact.py) against GIH WR, for notebooks/bombs.ipynb section 4c.

Per set, Premier Draft games are split into the same draft_id hash halves as card_impact.py (seed 7). Within each
half, every card gets
* its GIH WR rating (raw and shrunk, pp above the half's win rate, as card_impact.half_ratings), and
* its impact (spec B of card_impact.py: copies, score_{-i}, skill bucket, colors, play/draw, rated cards), with
  score_{-i} cross-fitted inside the half (its two hash quarters rate each other), so nothing from the other half
  is used.

That gives three things:
1. Held-out prediction. Values from half h predict game wins in half 1-h. Each model is a linear probability model
   fit on half h and scored on half 1-h (Brier, log loss), with two sets of controls: "none" (intercept, rated
   cards) and "skill" (spec B's controls). Models: controls only, + GIH deck score (mean rating, as section 2),
   + flexible GIH (deck sums of a piecewise-linear basis of the rating, knots at rating deciles 1,3,5,7,9),
   + impact (deck sum of impact per copy), and + impact + flexible GIH.
2. Split-half reliability: raw ratings and raw impacts from the two halves, per card, for the notebook to correlate.
3. Sealed: the same impact regression on Sealed games, with score_{-i} from the full Premier ratings. Sealed pools are
   random, so "good drafters take this card" mostly drops out.
Also per card: the mean skill bucket of the decks that play it and its mana value, for the residual check.

Output: results/bombs/impact_validation_cards.csv, results/bombs/impact_validation_pred.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from card_impact import BASICS, CHUNK, MIN_GIH, SETS, dummies, eb_shrink
from openlane.data import card_metadata
from openlane.deckstrength import aligned, flag
from openlane.games import game_path

OUT = ROOT / "results" / "bombs"
CLIP = 0.01


def raw_ratings(won, seen, deck, rarity, keep):
    # normalized GIH WR (pp) and its sampling variance per card on the games in `keep`; NaN below MIN_GIH
    w, s, d = won[keep], seen[keep], deck[keep] > 0
    n_gih, n_gns = s.sum(0), (d & ~s).sum(0)
    ok = (n_gih >= MIN_GIH) & (n_gns >= MIN_GIH) & (rarity != "basic")
    x, se2 = np.full(len(rarity), np.nan), np.full(len(rarity), np.nan)
    p = s[w].sum(0)[ok] / n_gih[ok]
    x[ok], se2[ok] = (p - w.mean()) * 100, p * (1 - p) / n_gih[ok] * 1e4
    return x, se2


def shrunk(x, se2, rarity):
    out, ok = np.full(len(x), np.nan), ~np.isnan(x)
    out[ok] = eb_shrink(x[ok], se2[ok], rarity[ok])
    return out


def deck_score(deck, R, grp):
    # S = sum of rated copies' ratings, K = rated copies, per row; row r uses ratings R[grp[r]]
    rated, rz = ~np.isnan(R), np.nan_to_num(R).astype(np.float32)
    S, K = np.zeros(len(deck)), np.zeros(len(deck))
    for g in np.unique(grp):
        m = grp == g
        d = deck[m].astype(np.float32)
        S[m], K[m] = d @ rz[g], d @ rated[g].astype(np.float32)
    return S, K


def controls(meta, K):
    return np.hstack([np.ones((len(K), 1)), dummies(meta["user_game_win_rate_bucket"]), dummies(meta["main_colors"]),
                      flag(meta["on_play"])[:, None].astype(np.float64), K[:, None]])


def fit_betas(deck, won, S, K, R, grp, C, draft_id, cand, fill):
    # card_impact.py's spec B: per card j in cand, won ~ copies_j + score_{-j} + C, SEs clustered by draft. pp.
    rated, rz = ~np.isnan(R), np.nan_to_num(R).astype(np.float32)
    _, cl = np.unique(draft_id, return_inverse=True)
    P32, C32 = np.linalg.pinv(C.T @ C).astype(np.float32), C.astype(np.float32)
    resid = lambda X: X - C32 @ (P32 @ (C32.T @ X))
    y = resid(won[:, None].astype(np.float32))[:, 0].astype(np.float64)
    beta, se = np.full(len(cand), np.nan), np.full(len(cand), np.nan)
    for c0 in range(0, len(cand), CHUNK):
        J = cand[c0:c0 + CHUNK]
        N = deck[:, J].astype(np.float32)
        Nr = N * rated[:, J][grp]
        rest = K[:, None] - Nr
        Sm = np.where(rest > 0, (S[:, None] - Nr * rz[:, J][grp]) / np.maximum(rest, 1), fill).astype(np.float32)
        Nt, St = resid(N).astype(np.float64), resid(Sm).astype(np.float64)
        g_ns = (Nt * St).sum(0) / (St * St).sum(0)
        nh = Nt - St * g_ns
        b = (nh * y[:, None]).sum(0) / (nh * nh).sum(0)
        gam = ((St * y[:, None]).sum(0) - b * (Nt * St).sum(0)) / (St * St).sum(0)
        e = y[:, None] - Nt * b - St * gam
        sc = np.stack([np.bincount(cl, nh[:, k] * e[:, k]) for k in range(len(J))], 1)
        beta[c0:c0 + len(J)] = b * 100
        se[c0:c0 + len(J)] = np.sqrt((sc ** 2).sum(0)) / (nh * nh).sum(0) * 100
    return beta, se


def load(s, event):
    lf = pl.scan_parquet(game_path(s, event))
    cols = lf.collect_schema().names()
    names = [c[len("drawn_"):] for c in cols if c.startswith("drawn_")]
    g = lf.select("draft_id", "won", "on_play", "main_colors", "user_game_win_rate_bucket",
                  *[c for c in cols if c.startswith(("deck_", "drawn_", "opening_hand_"))]).collect()
    deck = aligned(g, "deck_", names)
    seen = (aligned(g, "opening_hand_", names) + aligned(g, "drawn_", names)) > 0
    meta = g.select("draft_id", "won", "on_play", "main_colors", "user_game_win_rate_bucket")
    return names, deck, seen, meta


def cover_keep(deck, R, grp, nonbasic, draft_id):
    # card_impact.py's sample: drafts where every game has >= 90% of nonbasic copies rated
    S, K = deck_score(deck, R, grp)
    cover = K / np.maximum(deck[:, nonbasic].sum(1), 1)
    return (pl.DataFrame({"d": draft_id, "c": cover}).select(pl.col("c").min().over("d") >= 0.9)
            .to_series().to_numpy()) & (K > 0)


def lpm(X, y, Xt):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.clip(Xt @ b, CLIP, 1 - CLIP)


def run_set(s, imp_set):
    t0 = time.time()
    names, deck, seen, meta = load(s, "PremierDraft")
    idx = {n: j for j, n in enumerate(names)}
    cm = card_metadata(names, s)
    rarity = cm["rarity"].to_numpy().astype(object)
    rarity[np.isin(names, BASICS)] = "basic"
    nonbasic = ~np.isin(names, BASICS)
    won = flag(meta["won"])
    q = meta["draft_id"].hash(seed=7).to_numpy() % 4
    half, sub = q % 2, q // 2          # half matches card_impact.py's draft_id hash % 2
    did = meta["draft_id"].to_numpy()

    # same game sample as card_impact.py (cross-fitted full-data halves)
    R_full = np.stack([shrunk(*raw_ratings(won, seen, deck, rarity, half != h), rarity) for h in (0, 1)])
    keep = cover_keep(deck, R_full, half, nonbasic, did)
    norm_eb_full = shrunk(*raw_ratings(won, seen, deck, rarity, np.ones(len(won), bool)), rarity)   # for Sealed

    cand_all = [idx[n] for n in imp_set["name"] if n in idx]
    raw = {h: raw_ratings(won, seen, deck, rarity, keep & (half == h)) for h in (0, 1)}
    cand = [j for j in cand_all if not np.isnan(raw[0][0][j]) and not np.isnan(raw[1][0][j])]
    prior = np.where(cm["is_bonus"].to_numpy(), "bonus", rarity).astype(object)

    est, pred = {}, []
    for h in (0, 1):
        m = keep & (half == h)
        # score_{-i} inside half h: quarter k rated by the other quarter of the same half
        R = np.stack([shrunk(*raw_ratings(won, seen, deck, rarity, m & (sub != k)), rarity) for k in (0, 1)])
        d, w, grp, mm = deck[m], won[m].astype(np.float64), sub[m], meta.filter(pl.Series(m))
        S, K = deck_score(d, R, grp)
        beta, se = fit_betas(d, w, S, K, R, grp, controls(mm, K), did[m], cand, (S / K).mean())
        gih = shrunk(*raw[h], rarity)
        impact = np.full(len(names), np.nan)
        impact[cand] = eb_shrink(beta, se ** 2, prior[cand])
        est[h] = (beta, se)

        # held-out prediction: values from half h, model fit on half h, scored on half 1-h
        val = impact.copy()   # cards without an impact estimate: their rarity's mean impact; basics 0
        for r in np.unique(rarity[cand]):
            val[np.isnan(val) & (rarity == r)] = np.nanmean(impact[cand][rarity[cand] == r])
        val[~nonbasic | np.isnan(val)] = 0
        rated = ~np.isnan(gih)
        knots = np.quantile(gih[rated], [0.1, 0.3, 0.5, 0.7, 0.9])
        g0 = np.nan_to_num(gih)
        B = np.column_stack([g0] + [np.maximum(g0 - k, 0) for k in knots]) * rated[:, None]

        def feats(rows):
            dd = deck[rows].astype(np.float32)
            Kr = dd @ rated.astype(np.float32)
            return {"K": Kr, "score": (dd @ g0.astype(np.float32)) / np.maximum(Kr, 1),
                    "flex": dd @ B.astype(np.float32), "V": dd @ val.astype(np.float32)}

        te = keep & (half == 1 - h)
        ftr, fte = feats(m), feats(te)
        mte = meta.filter(pl.Series(te))
        # skill dummies must have the same columns in both halves: build them on the stacked rows
        both = pl.concat([mm, mte])
        Cs = controls(both, np.concatenate([ftr["K"], fte["K"]]).astype(np.float64))
        Cs_tr, Cs_te = Cs[:m.sum()], Cs[m.sum():]
        yt = won[te].astype(np.float64)
        _, clt = np.unique(did[te], return_inverse=True)
        for cs in ("none", "skill"):
            plain = lambda f: np.column_stack([np.ones(len(f["K"])), f["K"]])
            c_tr, c_te = (plain(ftr), plain(fte)) if cs == "none" else (Cs_tr, Cs_te)
            models = {"controls only": [], "+ GIH score": ["score"], "+ GIH flexible": ["flex"],
                      "+ impact": ["V"], "+ impact + GIH flexible": ["V", "flex"]}
            loss = {}
            for name, fs in models.items():
                Xtr = np.column_stack([c_tr] + [np.asarray(ftr[f], np.float64).reshape(len(w), -1) for f in fs])
                Xte = np.column_stack([c_te] + [np.asarray(fte[f], np.float64).reshape(len(yt), -1) for f in fs])
                p = lpm(Xtr, w, Xte)
                loss[name] = ((p - yt) ** 2, -(yt * np.log(p) + (1 - yt) * np.log(1 - p)))
            for name, (sq, ll) in loss.items():
                row = {"set": s, "fit_half": h, "controls": cs, "model": name, "games": len(yt),
                       "brier": sq.mean(), "logloss": ll.mean(), "var_won": yt.var()}
                for k, a, b in (("brier", sq, loss["controls only"][0]), ("logloss", ll, loss["controls only"][1])):
                    dlt = a - b
                    cs_ = np.bincount(clt, dlt - dlt.mean())
                    row[f"d_{k}"], row[f"se_d_{k}"] = dlt.mean(), np.sqrt((cs_ ** 2).sum()) / len(dlt)
                pred.append(row)
    del seen

    # per-card extras on the kept sample: mean skill bucket of the decks that play the card
    sk = meta["user_game_win_rate_bucket"].cast(pl.Utf8).cast(pl.Float64, strict=False).to_numpy()
    ok = keep & ~np.isnan(sk)
    inn = deck[ok][:, cand] > 0
    skill_in_deck = (inn * sk[ok][:, None]).sum(0) / np.maximum(inn.sum(0), 1) - sk[ok].mean()
    out = pl.DataFrame({
        "set": s, "name": [names[j] for j in cand], "rarity": rarity[cand].tolist(),
        "is_bonus": cm["is_bonus"].to_numpy()[cand], "mana_value": cm["mana_value"].to_numpy()[cand],
        "gih_0": raw[0][0][cand], "gih_se2_0": raw[0][1][cand], "gih_1": raw[1][0][cand], "gih_se2_1": raw[1][1][cand],
        "beta_0": est[0][0], "se_0": est[0][1], "beta_1": est[1][0], "se_1": est[1][1],
        "skill_in_deck": skill_in_deck * 100})
    del deck

    # Sealed: same regression, score_{-i} from the full Premier ratings
    sn, sdeck, _, smeta = load(s, "Sealed")
    sidx = {n: j for j, n in enumerate(sn)}
    R_s = np.array([norm_eb_full[idx[n]] if n in idx else np.nan for n in sn])[None, :]
    snb = ~np.isin(sn, BASICS)
    sdid = smeta["draft_id"].to_numpy()
    zero = np.zeros(len(sdid), np.int64)
    skeep = cover_keep(sdeck, R_s, zero, snb, sdid)
    sdeck, smeta, sdid = sdeck[skeep], smeta.filter(pl.Series(skeep)), sdid[skeep]
    S, K = deck_score(sdeck, R_s, zero[skeep])
    scand = [sidx[names[j]] for j in cand if names[j] in sidx]
    sb, sse = fit_betas(sdeck, flag(smeta["won"]).astype(np.float64), S, K, R_s, zero[skeep],
                        controls(smeta, K), sdid, scand, (S / K).mean())
    sealed = pl.DataFrame({"name": [sn[j] for j in scand], "sealed_beta": sb, "sealed_se": sse,
                           "sealed_games_in_deck": (sdeck[:, scand] > 0).sum(0).astype(np.int64)})
    out = out.join(sealed, on="name", how="left").with_columns(pl.lit(len(sdid)).alias("sealed_games"))
    print(f"{s}: {len(cand)} cards, sealed {len(sdid):,} games, {time.time() - t0:.0f}s", flush=True)
    return out, pl.DataFrame(pred)


if __name__ == "__main__":
    sets = sys.argv[1:] or SETS
    imp = pl.read_csv(OUT / "card_impact.csv")
    outs, preds = [], []
    for s in sets:
        o, p = run_set(s, imp.filter(pl.col("set") == s))
        outs.append(o); preds.append(p)
    suffix = "" if sets == SETS else "_" + "_".join(sets)
    pl.concat(outs).write_csv(OUT / f"impact_validation_cards{suffix}.csv")
    pl.concat(preds).write_csv(OUT / f"impact_validation_pred{suffix}.csv")
