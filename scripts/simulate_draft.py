"""Simulate whole 8-seat drafts with a pick model at every seat, and compare the tables to real drafts.

    python scripts/simulate_draft.py MSH picknet [n_tables] [temperature] [seed]

Packs are real: each table takes 8 random test drafts and gives seat s the three packs that draft's
drafter opened (pack shown at pick 1 of each round). Seat s also gets that drafter's context (rank,
skill, date). Packs go left in rounds 1 and 3 and right in round 2. Each bot samples its pick from the
model's softmax (temperature 1 = the model's own probabilities), so this is a simulation of human
choices, not a best-pick bot.

Compared with the real test drafts (which are one seat at real 8-person tables):
  - average taken at (ATA) per card, and average last seen at (ALSA)
  - per-card wheel rate at picks 1-6 (the rate by pick is fixed by pack size: 6 of 13 come back at
    pick 1), and by rarity
  - colour commitment: share of the last 14 picks in the drafter's top two colours, and the
    distribution of main colour pairs
Writes results/{SET}.PremierDraft/picknet/sim_{model}.json.
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import COLORS, N_PACKS, N_SEATS, ROOT, load_tensors  # noqa: E402
from openlane.nn import draft_context, sparse  # noqa: E402
from openlane.picknet import load_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_picknet import splits_for  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def simulate(model, opened, ctx, n_picks, temperature=1.0, chunk=512, rng=None):
    """opened [G, 8, K, C] packs each seat opens, ctx [G, 8, F]. Returns seen [G*8, K, N, C], pick [G*8, K, N]."""
    G, S, K, C = opened.shape
    T = K * n_picks
    B = G * S
    seen = np.zeros((B, K, n_picks, C), np.int8)
    pick = np.zeros((B, K, n_picks), np.int64)
    P = n_picks + 1
    idx = np.full((B, T, P), C, np.int64)
    cnt = np.zeros((B, T, P), np.int64)
    oidx = np.full((B, T, N_SEATS + 2), C, np.int64)
    ocnt = np.zeros((B, T, N_SEATS + 2), np.int64)
    pk = np.zeros((B, T), np.int64)
    ctx_b = torch.from_numpy(ctx.reshape(B, -1).astype(np.float32))
    seat = np.arange(S)
    for k in range(K):
        cur = opened[:, :, k].astype(np.int16).copy()   # [G, opener, C]
        direction = 1 if k % 2 == 0 else -1
        for p in range(n_picks):
            t = k * n_picks + p
            holder = (seat - direction * p) % S          # opener of the pack seat s holds
            here = cur[:, holder].reshape(B, C)         # [B, C]
            seen[:, k, p] = here
            i, c = sparse(here, P)
            idx[:, t], cnt[:, t] = i, c
            if p >= N_SEATS:
                prev = seen[:, k, p - N_SEATS].astype(np.int16)
                prev[np.arange(B), pick[:, k, p - N_SEATS]] -= 1
                oi, oc = sparse(np.clip(prev - here, 0, None).astype(np.int8), N_SEATS + 2)
                oidx[:, t], ocnt[:, t] = oi, oc
            logits = []
            for a in range(0, B, chunk):
                s = slice(a, a + chunk)
                x = [torch.from_numpy(v[s, :t + 1]).to(DEV) for v in (idx, cnt, pk, oidx, ocnt)]
                lg = model(*x, ctx_b[s].to(DEV))[:, t]
                logits.append(lg.masked_fill(x[1][:, t] == 0, float("-inf")).float().cpu())
            prob = F.softmax(torch.cat(logits) / temperature, -1).numpy().astype(np.float64)
            prob /= prob.sum(1, keepdims=True)
            u = rng.random((B, 1))
            j = np.minimum((prob.cumsum(1) < u).sum(1), P - 1)
            chosen = idx[np.arange(B), t, j]
            pick[:, k, p] = chosen
            pk[:, t] = chosen
            g, h = np.divmod(np.arange(B), S)
            cur[g, holder[h], chosen] -= 1
    return seen, pick


def stats(pack, pick, color, n_picks):
    """Per-card ATA / ALSA, wheel rate by pick and per card, colour commitment, colour-pair shares."""
    D, K, N, C = pack.shape
    pos = np.broadcast_to(np.arange(1, N + 1)[None, None], pick.shape)
    taken_n = np.bincount(pick.ravel(), minlength=C)
    ata = np.bincount(pick.ravel(), weights=pos.ravel(), minlength=C) / np.maximum(taken_n, 1)
    # last seen: the latest pick (within a round) at which each card copy in the pack was visible
    seen_any = pack > 0
    last = np.where(seen_any, pos[..., None], 0).max(2)  # [D, K, C]
    alsa = last.sum((0, 1)) / np.maximum((last > 0).sum((0, 1)), 1)
    # wheel: of the cards passed at p (pack minus pick), which copies are in the pack at p+8
    W = N - N_SEATS
    passed = pack[:, :, :W].astype(np.int16).copy()
    np.put_along_axis(passed, pick[:, :, :W, None], np.take_along_axis(passed, pick[:, :, :W, None], -1) - 1, -1)
    back = pack[:, :, N_SEATS:N_SEATS + W]
    came = (np.minimum(passed, back) > 0)
    pas = passed > 0
    wheel_card = came.sum((0, 1, 2)) / np.maximum(pas.sum((0, 1, 2)), 1)
    n_pass = pas.sum((0, 1, 2))
    # colour commitment from the picks: colour weights of picked cards (mono-coloured counts 1)
    oh = np.zeros((D, K * N, C), np.float32)
    np.put_along_axis(oh, pick.reshape(D, -1)[..., None], 1, -1)
    col = oh @ color                                          # [D, T, 5]
    totals = col.sum(1)
    top2 = np.argsort(-totals, 1)[:, :2]
    last14 = col[:, -N:]
    in_top2 = np.take_along_axis(last14, top2[:, None, :], -1).sum(-1) > 0
    colourless_or = last14.sum(-1) > 0
    commit = float((in_top2 & colourless_or).sum() / max(colourless_or.sum(), 1))
    pairs = ["".join(COLORS[i] for i in sorted(tp)) for tp in top2]
    order = ["".join(c) for c in combinations(COLORS, 2)]
    pair_share = {pr: pairs.count(pr) / D for pr in order}
    return dict(ata=ata, alsa=alsa, taken=taken_n, wheel_card=wheel_card, n_pass=n_pass,
                commit=commit, pair_share=pair_share, p1p1=np.bincount(pick[:, 0, 0], minlength=C) / D)


def compare(sim, real, names, rarity):
    ok = (real["taken"] >= 200) & (sim["taken"] >= 50)
    okw = (real["n_pass"] >= 200) & (sim["n_pass"] >= 50)
    corr = lambda a, b, m: float(np.corrcoef(a[m], b[m])[0, 1])  # noqa: E731
    pr = np.array(list(real["pair_share"].values()))
    ps = np.array(list(sim["pair_share"].values()))
    top = np.argsort(-real["p1p1"])[:15]
    return {
        "ata_corr": corr(sim["ata"], real["ata"], ok), "ata_mae": float(np.abs(sim["ata"] - real["ata"])[ok].mean()),
        "alsa_corr": corr(sim["alsa"], real["alsa"], ok),
        "alsa_mae": float(np.abs(sim["alsa"] - real["alsa"])[ok].mean()),
        "wheel_card_corr": corr(sim["wheel_card"], real["wheel_card"], okw),
        "wheel_card_mae": float(np.abs(sim["wheel_card"] - real["wheel_card"])[okw].mean()),
        "wheel_by_rarity": {r: (float(real["wheel_card"][okw & (rarity == r)].mean()),
                                float(sim["wheel_card"][okw & (rarity == r)].mean()))
                            for r in ("common", "uncommon", "rare", "mythic") if (okw & (rarity == r)).any()},
        "commit_sim": sim["commit"], "commit_real": real["commit"],
        "pair_share_sim": sim["pair_share"], "pair_share_real": real["pair_share"],
        "pair_share_tv": float(0.5 * np.abs(pr - ps).sum()),
        "p1p1_corr": corr(sim["p1p1"], real["p1p1"], real["p1p1"] >= 0),
        "p1p1_top15": [(names[i], float(real["p1p1"][i]), float(sim["p1p1"][i])) for i in top],
        "per_card": {"name": [names[i] for i in np.flatnonzero(ok)], "ata_real": real["ata"][ok].tolist(),
                     "ata_sim": sim["ata"][ok].tolist(), "taken_real": real["taken"][ok].tolist()},
        "ata_biggest_gaps": sorted([(names[i], float(real["ata"][i]), float(sim["ata"][i]))
                                    for i in np.flatnonzero(ok)], key=lambda r: -abs(r[1] - r[2]))[:15],
    }


def main(set_code, name="picknet", n_tables="1000", temperature="1.0", seed="0"):
    n_tables, temperature = int(n_tables), float(temperature)
    rng = np.random.default_rng(int(seed))
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "picknet"
    t0 = time.time()
    t = load_tensors(set_code)
    model, ck = load_model(out / "models" / f"{name}.pt", DEV)
    test = splits_for(t, set_code, "PremierDraft")["test"]
    use_ctx = ck["train"].get("ctx", 1)
    ctx = draft_context(t) if use_ctx else np.zeros((len(t.draft_id), 0), np.float32)
    color = t.color_matrix
    color = color / np.maximum(color.sum(1, keepdims=True), 1)
    real = stats(t.pack[test], t.pick[test].astype(np.int64), color, t.n_picks)
    tables = rng.choice(test, (n_tables, N_SEATS), replace=True)
    opened = t.pack[tables][:, :, :, 0]                        # [G, 8, K, C]
    assert not t.p1p1_missing, "needs the opened packs"
    seen, pick = simulate(model, opened, ctx[tables], t.n_picks, temperature, rng=rng)
    sim = stats(seen, pick, color, t.n_picks)
    res = {"model": name, "tables": n_tables, "temperature": temperature, **compare(sim, real, t.names, t.meta["rarity"].to_numpy()),
           "seconds": time.time() - t0}
    (out / f"sim_{name}{'' if temperature == 1 else f'_T{temperature:g}'}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / f"sim_{name}_drafts.npz", pick=pick.astype(np.int16), tables=tables)
    print(f"{set_code} {name} T={temperature}: ATA corr {res['ata_corr']:.3f} (MAE {res['ata_mae']:.2f}) "
          f"ALSA corr {res['alsa_corr']:.3f} wheel/card corr {res['wheel_card_corr']:.3f} MAE {res['wheel_card_mae']:.3f} "
          f"commit {res['commit_sim']:.3f} vs {res['commit_real']:.3f} pair TV {res['pair_share_tv']:.3f} "
          f"P1P1 corr {res['p1p1_corr']:.3f} [{time.time() - t0:.0f}s]")
    print("  wheel rate by rarity (real, sim):", {k: tuple(round(x, 3) for x in v) for k, v in res["wheel_by_rarity"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
