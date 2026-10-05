"""Should you read signals or force a colour pair? python scripts/force_vs_follow.py MSH [n_tables] [model] [device]

Counterfactual for the open-lane analysis: 8-seat tables of PickNet bots (scripts/simulate_draft.py:
real opened packs, each bot samples its pick from the model). Seat 0 plays a strategy, seats 1-7 are
always the human-like bot. Every strategy runs on the same tables and the same random stream.

  follow      seat 0 is the human-like bot too (reads signals as people do)
  force       seat 0 samples from the same model restricted to cards whose colours fit a fixed pair
              (colourless always fits; if nothing fits it picks normally); table g forces pair g mod 10
  force_late  the same, but only from pack 1 pick 6: take P1's first five picks freely, then lock the
              pair that is most open by then (flow of good cards, as in open_lane.py), i.e. a drafter
              who reads early signals and then commits hard

Pools are scored as decks: for each of the 10 pairs take the 23 best cards (by GIH WR from training
games) whose colours fit, missing slots get a replacement-level card (10th percentile GIH WR); the
deck score is the best pair's mean GIH WR. Writes results/{SET}.PremierDraft/force_vs_follow.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlane.data import COLORS, N_SEATS, ROOT, load_tensors  # noqa: E402
from openlane.nn import draft_context, sparse  # noqa: E402
from openlane.picknet import load_model  # noqa: E402
from train_picknet import splits_for  # noqa: E402

PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


@torch.no_grad()
def simulate(model, opened, ctx, n_picks, rule, colbin, alsa, dev, chunk=512, seed=0):
    """simulate_draft.simulate with a pick rule for seat 0: rule(k, p, flow) -> [G, 5] allowed colours or None."""
    rng = np.random.default_rng(seed)
    G, S, K, C = opened.shape
    T, B, P = K * n_picks, G * S, n_picks + 1
    seen = np.zeros((B, K, n_picks, C), np.int8)
    pick = np.zeros((B, K, n_picks), np.int64)
    idx = np.full((B, T, P), C, np.int64)
    cnt = np.zeros((B, T, P), np.int64)
    oidx = np.full((B, T, N_SEATS + 2), C, np.int64)
    ocnt = np.zeros((B, T, N_SEATS + 2), np.int64)
    pk = np.zeros((B, T), np.int64)
    ctx_b = torch.from_numpy(ctx.reshape(B, -1).astype(np.float32))
    seat, s0 = np.arange(S), np.arange(G) * S
    share = colbin / np.maximum(colbin.sum(1, keepdims=True), 1)
    flow = np.zeros((G, 5))                                  # seat 0's openness signal so far
    for k in range(K):
        cur = opened[:, :, k].astype(np.int16).copy()
        direction = 1 if k % 2 == 0 else -1
        for p in range(n_picks):
            t = k * n_picks + p
            holder = (seat - direction * p) % S
            here = cur[:, holder].reshape(B, C)
            seen[:, k, p] = here
            i, c = sparse(here, P)
            idx[:, t], cnt[:, t] = i, c
            if 1 <= p < N_SEATS:
                flow += here[s0].astype(np.float64) @ (share * np.maximum(0, p - alsa)[:, None])
            if p >= N_SEATS:
                prev = seen[:, k, p - N_SEATS].astype(np.int16)
                prev[np.arange(B), pick[:, k, p - N_SEATS]] -= 1
                oi, oc = sparse(np.clip(prev - here, 0, None).astype(np.int8), N_SEATS + 2)
                oidx[:, t], ocnt[:, t] = oi, oc
            logits = []
            for a in range(0, B, chunk):
                s = slice(a, a + chunk)
                x = [torch.from_numpy(v[s, :t + 1]).to(dev) for v in (idx, cnt, pk, oidx, ocnt)]
                lg = model(*x, ctx_b[s].to(dev))[:, t]
                logits.append(lg.masked_fill(x[1][:, t] == 0, float("-inf")).float().cpu())
            lg = torch.cat(logits).numpy().astype(np.float64)
            allowed = rule(k, p, flow)
            if allowed is not None:                          # seat 0: best card that fits the pair
                cand = idx[s0, t]                            # [G, P]
                cb = np.vstack([colbin, np.zeros((1, 5))])[cand]                  # [G, P, 5]
                fits = (cb * ~allowed[:, None, :]).sum(-1) == 0
                fits &= cnt[s0, t] > 0
                # sample like the others, but only among cards that fit (if any do)
                lg[s0] = np.where(fits.any(1, keepdims=True) & ~fits, -np.inf, lg[s0])
            prob = np.exp(lg - lg.max(1, keepdims=True))
            prob /= prob.sum(1, keepdims=True)
            u = rng.random((B, 1))
            j = np.minimum((prob.cumsum(1) < u).sum(1), P - 1)
            chosen = idx[np.arange(B), t, j]
            pick[:, k, p] = chosen
            pk[:, t] = chosen
            g, h = np.divmod(np.arange(B), S)
            cur[g, holder[h], chosen] -= 1
    return pick.reshape(G, S, -1)


def deck_scores(pick, colbin, gih, n_deck=23):
    """[G, S, 42] picks -> best-pair deck score [G, S] and the pair index."""
    G, S, T = pick.shape
    repl = np.quantile(gih, 0.10)
    v = gih[pick]                                            # [G, S, T]
    cb = colbin[pick]                                        # [G, S, T, 5]
    out = np.full((G, S, len(PAIRS)), -np.inf)
    for q, (a, b) in enumerate(PAIRS):
        ok = np.ones(5, bool)
        ok[[a, b]] = False
        fits = (cb[..., ok].sum(-1) == 0)
        vv = np.sort(np.where(fits, v, repl), -1)[..., ::-1][..., :n_deck]
        out[..., q] = vv.mean(-1)
    return out.max(-1), out.argmax(-1)


def main(set_code="MSH", n_tables="2000", name="picknet", dev=None):
    n_tables = int(n_tables)
    dev = dev or ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    t = load_tensors(set_code)
    model, ck = load_model(ROOT / "results" / f"{set_code}.PremierDraft" / "picknet" / "models" / f"{name}.pt", dev)
    test = splits_for(t, set_code, "PremierDraft")["test"]
    ctx = draft_context(t) if ck["train"].get("ctx", 1) else np.zeros((len(t.draft_id), 0), np.float32)
    colbin = t.color_matrix > 0
    cv = pl.read_csv(ROOT / "results" / f"{set_code}.PremierDraft" / "deckstrength" / "card_values.csv")
    gih = dict(zip(cv["name"], cv["gih_wr"]))
    gih = np.array([gih.get(n, np.nan) for n in t.names])
    gih = np.where(np.isnan(gih), np.nanquantile(gih, 0.10), gih)
    import pickle
    alsa = pickle.load(open(ROOT / "data" / "features" / f"{set_code}.PremierDraft" / "meta.pkl", "rb"))["stats"].alsa
    rng = np.random.default_rng(0)
    tables = rng.choice(test, (n_tables, N_SEATS), replace=True)
    opened = t.pack[tables][:, :, :, 0]
    G = n_tables
    forced = np.zeros((G, 5), bool)
    for g in range(G):
        forced[g, list(PAIRS[g % 10])] = True
    late = {}

    def force_late(k, p, flow):
        if k == 0 and p < 5:
            return None
        if "pair" not in late:
            late["pair"] = np.array([[flow[g, a] + flow[g, b] for a, b in PAIRS] for g in range(G)]).argmax(1)
            late["mask"] = np.zeros((G, 5), bool)
            for g in range(G):
                late["mask"][g, list(PAIRS[late["pair"][g]])] = True
        return late["mask"]

    rules = {"follow": lambda k, p, f: None, "force": lambda k, p, f: forced, "force_late": force_late}
    res = {"set": set_code, "tables": G, "model": name}
    picks = {}
    for strat, rule in rules.items():
        pick = simulate(model, opened, ctx[tables], t.n_picks, rule, colbin, alsa, dev, seed=1)
        sc, pair = deck_scores(pick, colbin, gih)
        picks[strat] = pick
        r = {"seat0": float(sc[:, 0].mean()), "seat0_se": float(sc[:, 0].std() / np.sqrt(G)),
             "others": float(sc[:, 1:].mean()),
             "seat0_minus_others": float((sc[:, 0] - sc[:, 1:].mean(1)).mean()),
             "neighbours_minus_others": float((sc[:, [1, 7]].mean(1) - sc[:, 2:7].mean(1)).mean())}
        if strat == "force":
            r["by_pair"] = {"".join(COLORS[c] for c in PAIRS[q]): float(sc[np.arange(G) % 10 == q, 0].mean())
                            for q in range(10)}
        if strat == "follow":
            res["follow_scores"] = sc[:, 0]
        else:
            d = sc[:, 0] - res["follow_scores"]
            r["vs_follow"] = float(d.mean())
            r["vs_follow_se"] = float(d.std() / np.sqrt(G))
        res[strat] = r
        print(f"{strat}: {json.dumps(r)} [{time.time() - t0:.0f}s]", flush=True)
    res.pop("follow_scores")
    res["seconds"] = time.time() - t0
    (ROOT / "results" / f"{set_code}.PremierDraft" / "force_vs_follow.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(ROOT / "results" / f"{set_code}.PremierDraft" / "force_vs_follow_picks.npz",
                        tables=tables, **{k: v.astype(np.int16) for k, v in picks.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
