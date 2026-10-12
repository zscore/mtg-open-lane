"""Does reading signals pay? python scripts/open_lane.py MSH [SET ...]

Lane openness from what the drafter was passed. For pack k and colour j,
    flow[d, k, j] = sum over picks q = 1..7 of the cards c in the pack shown at q of
                    colour_share[c, j] * max(0, q - ALSA[c])
i.e. how many good cards of colour j arrived later than they usually last (the `late` signal of
features.py, upstream picks only: picks 8+ are packs that went round the whole table). It is
z-scored per (pack, colour) within each draft week, so set-wide shifts in colour popularity cancel.
The drafter's own picks never enter it: it is fixed by the opened packs and the upstream drafters.

Packs pass left in P1 and P3 and right in P2, so P1 and P3 share the upstream neighbour.
1. Persistence: corr(flow P1, flow P3) vs corr(flow P1, flow P2), per colour.
2. Following: P(colour j is one of the drafter's two main colours | P1 flow of j), by rank/skill.
3. Payoff: binomial logit of the event record (p^W (1-p)^L, as in deckstrength.py) on the openness of
   the drafter's main colours, with colour-pair fixed effects, rank and the 17Lands skill bucket.
Writes results/{SET}.PremierDraft/open_lane.json.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.data import COLORS, FEATURES, download, load_tensors  # noqa: E402
from openlane import ROOT  # noqa: E402

PAIRS = [a + b for i, a in enumerate(COLORS) for b in COLORS[i + 1:]]


def binom_logit(X, w, l, ridge=1e-4):
    """Max-likelihood p = sigmoid(X b) for W wins / L losses per row; returns b and standard errors."""
    def f(b):
        z = X @ b
        ll = w * -np.logaddexp(0, -z) + l * -np.logaddexp(0, z)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (w * (1 - p) - l * p)
        return -ll.sum() + ridge * b @ b, -g + 2 * ridge * b
    b = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
    p = 1 / (1 + np.exp(-(X @ b)))
    H = (X * ((w + l) * p * (1 - p))[:, None]).T @ X + 2 * ridge * np.eye(X.shape[1])
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def onehot(v, n=None):
    v = np.asarray(v)
    n = n or int(v.max()) + 1
    return np.eye(n, dtype=np.float64)[v]


def run(set_code: str) -> dict:
    t = load_tensors(set_code)
    meta = pickle.load(open(FEATURES / f"{set_code}.PremierDraft" / "meta.pkl", "rb"))
    assert (meta["draft_id"] == t.draft_id).all()
    alsa = meta["stats"].alsa
    col = t.color_matrix
    share = col / np.maximum(col.sum(1), 1)[:, None]                      # [C,5]
    D, K, P, C = t.pack.shape
    flow = np.zeros((D, K, 5), np.float64)
    for q in range(1, min(8, P)):
        if t.p1p1_missing and q == 0:
            continue
        wq = share * np.maximum(0, q - alsa)[:, None]                     # [C,5]
        for k in range(K):
            flow[:, k] += t.pack[:, k, q].astype(np.float32) @ wq
    week = ((t.draft_time.astype("datetime64[s]") - t.draft_time.astype("datetime64[s]").min())
            .astype(np.int64) // (7 * 86400)).astype(np.int64)
    z = np.zeros_like(flow)
    for wk in np.unique(week):
        m = week == wk
        mu, sd = flow[m].mean(0), flow[m].std(0) + 1e-9
        z[m] = (flow[m] - mu) / sd

    # main colours: colour mass of all 42 picks, weighted toward cards that get picked early
    picks = t.pick.reshape(D, -1).astype(np.int64)
    val = np.maximum(0, 10 - meta["stats"].ata)                           # crude playability
    mass = np.zeros((D, 5))
    for i in range(picks.shape[1]):
        mass += share[picks[:, i]] * val[picks[:, i], None]
    top2 = np.argsort(-mass, 1)[:, :2]
    main = np.zeros((D, 5), bool)
    main[np.arange(D)[:, None], top2] = True
    pair = np.array([PAIRS.index("".join(sorted((COLORS[a], COLORS[b]), key=COLORS.index)))
                     for a, b in top2])
    res = {"set": set_code, "drafts": int(D)}

    # 1. persistence across rounds, per colour
    cc = lambda a, b: float(np.mean([np.corrcoef(z[:, a, j], z[:, b, j])[0, 1] for j in range(5)]))
    res["persistence"] = {"P1-P3 (same neighbour)": cc(0, 2), "P1-P2": cc(0, 1), "P2-P3": cc(1, 2)}

    # 2. following: logit of main colour on P1 flow, colour fixed effects; slope by rank and skill
    def follow(mask):
        y = main[mask].ravel().astype(float)
        x = z[mask, 0].ravel()
        X = np.column_stack([x, np.tile(np.eye(5), (int(mask.sum()), 1))])
        b, se = binom_logit(X, y, 1 - y)
        return {"slope": float(b[0]), "se": float(se[0]), "n": int(mask.sum())}
    res["follow_all"] = follow(np.ones(D, bool))
    res["follow_by_rank"] = {r: follow(t.rank == i) for i, r in enumerate(
        ["bronze", "silver", "gold", "platinum", "diamond", "mythic"]) if (t.rank == i).sum() > 500}
    wr = t.user_wr
    buckets = {"<0.50": (wr >= 0) & (wr < 0.5), "0.50-0.56": (wr >= 0.5) & (wr < 0.56),
               "0.56-0.62": (wr >= 0.56) & (wr < 0.62), ">=0.62": wr >= 0.62}
    res["follow_by_skill"] = {k: follow(m) for k, m in buckets.items() if m.sum() > 500}

    # 3. payoff: event record on openness of the main colours
    # records straight from the draft file (load_drafts keeps only max-length drafts, which on ECL
    # is one malformed draft)
    rec = (pl.scan_parquet(download(set_code, "PremierDraft"))
           .group_by("draft_id").agg(pl.col("event_match_wins").max().alias("W"),
                                     pl.col("event_match_losses").max().alias("L")).collect())
    rec = pl.DataFrame({"draft_id": t.draft_id}).join(rec, on="draft_id", how="left", maintain_order="left")
    W, L = rec["W"].fill_null(0).to_numpy(), rec["L"].fill_null(0).to_numpy()
    ok = (W + L) > 0
    res["drafts_with_record"] = int(ok.sum())
    if ok.sum() < 1000:  # e.g. a set whose files carry no event records
        return res
    open_main = {k: (z[:, k] * main).sum(1) / 2 for k in range(K)}            # mean over 2 main colours
    open_other = {k: (z[:, k] * ~main).sum(1) / 3 for k in range(K)}
    has_skill = wr >= 0
    ctrl = [onehot(pair, 10), onehot(t.rank + 1, 7)[:, 1:],
            np.column_stack([np.where(has_skill, wr - 0.54, 0) / 0.09, ~has_skill])]
    pay = {}
    for name, cols in {
        "P1 openness of main colours": [open_main[0]],
        "P1 + P3 (same neighbour)": [open_main[0] + open_main[2]],
        "P1, P2, P3 separately": [open_main[0], open_main[1], open_main[2]],
        "P1 main vs P1 other colours": [open_main[0], open_other[0]],
    }.items():
        for skill in (False, True):
            X = np.column_stack([*cols, *ctrl[:2], *(ctrl[2:] if skill else [])])[ok]
            b, se = binom_logit(X, W[ok], L[ok])
            pay[f"{name}{' | +skill' if skill else ''}"] = {
                "coef": [round(float(v), 4) for v in b[:len(cols)]],
                "se": [round(float(v), 4) for v in se[:len(cols)]]}
    res["payoff"] = pay
    # descriptive: game win rate by quintile of P1 openness of main colours
    o = open_main[0][ok]
    qs = np.quantile(o, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(qs, o)
    res["winrate_by_P1_openness_quintile"] = [round(float(W[ok][qi == i].sum() / (W[ok][qi == i] + L[ok][qi == i]).sum()), 4)
                                              for i in range(5)]
    # did they end in the most-open P1 pair?
    zp = np.stack([z[:, 0, COLORS.index(p[0])] + z[:, 0, COLORS.index(p[1])] for p in PAIRS], 1)
    in_best = zp.argmax(1) == pair
    res["share_in_most_open_P1_pair"] = float(in_best.mean())
    res["winrate_in_vs_not_most_open_pair"] = [
        round(float(W[ok & in_best].sum() / (W[ok & in_best] + L[ok & in_best]).sum()), 4),
        round(float(W[ok & ~in_best].sum() / (W[ok & ~in_best] + L[ok & ~in_best]).sum()), 4)]
    return res


def main(*sets):
    for s in sets or ["MSH"]:
        r = run(s)
        (ROOT / "results" / f"{s}.PremierDraft" / "open_lane.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
