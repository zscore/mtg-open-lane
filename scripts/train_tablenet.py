"""TableNet, step 1: a wheel head on the frozen all-sets PickNet (see paper/combined/architecture.tex).

    python scripts/train_tablenet.py [key=value ...]

Loads picknet's all-sets model (default `frahob_freeze`: all_d256_ep3 with FRA and HOB fitted), freezes it, and
trains wheel heads on its internals. Data comes straight from picknet's shards (mtg-data
features/picks_all), Premier only, with the same chronological train/val/test splits as WheelNet. Wheel labels
are computed from the shards: the cards passed at a wheel-eligible pick t (pack t minus the pick) against the
pack shown at t+8. Works for 13-, 14- and 15-card packs (n-8 wheel-eligible picks per pack); a missing P1P1
pack simply has no passed cards.

Two heads train side by side on the same frozen forward pass: L=0 (per-card MLP) and L=1 (plus one
self-attention layer over the passed cards of a pack). Each reads, for a passed card c at pick t, PickNet's
final candidate token for c, the history state h_t, the card taken at t, c's and the pick's log-probs, and
the copies passed; nothing after slot t (checked at startup by blanking every later slot).

Keys: init (PickNet .pt), sets (default all Premier shards), max_per_set (train drafts per set, 15000),
val_per_set (1000), epochs (2), batch (64), lr (1e-3), wd (0.01), d (head width, 256), name ("tablenet"),
test_sets (default: sets), bench (time 20 steps and exit).
Writes results/tablenet/{name}.json, models/{name}_L{0,1}.pt, test_{name}_L{0,1}_{SET}.npz.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
PICKNET = Path(__file__).resolve().parents[2] / "picknet"
sys.path.insert(0, str(PICKNET / "scripts"))
from openlane import ROOT  # noqa: E402
from openlane.evaluate import metrics  # noqa: E402
from picknet.model import PickNet, bag  # noqa: E402
from train_picknet_all import DEV, N_PICKS, SHARDS, Shard, gather, to_torch, train_batches, value  # noqa: E402

N_PACKS, N_SEATS = 3, 8
W_MAX = N_PICKS - N_SEATS  # 7 wheel-eligible picks per pack at most (15-card packs)
WP = torch.tensor([k * N_PICKS + p for k in range(N_PACKS) for p in range(W_MAX)])  # [21] history slots
CFG = dict(init=str(PICKNET / "results" / "picknet_all" / "models" / "frahob_freeze.pt"), sets="",
           max_per_set=15000, val_per_set=1000, epochs=2, batch=64, lr=1e-3, wd=0.01, d=256, name="tablenet",
           test_sets="", bench=0, seed=0)
OUT = ROOT / "results" / "tablenet"


def trunk(m: PickNet, idx, cnt, pick, oidx, ocnt, ctx, sid):
    """PickNet.forward in eval mode, up to the scoring head. Returns final candidate tokens x [B,T,P,d],
    history states h [B,T,d], the card table, and the pick logits [B,T,P]."""
    B, T, P = idx.shape
    tab = m.cards()
    e_pick = F.embedding(pick, tab)
    prev = torch.cat([torch.zeros_like(e_pick[:, :1]), e_pick[:, :-1]], 1)
    h = m.pos.weight[None, :T] + m.pack_bag(bag(tab, idx, cnt)) + m.ctx_mlp(ctx)[:, None]
    h = h + m.set_emb(sid.long())[:, None] + m.pick_in(prev) + m.oth_bag(bag(tab, oidx, ocnt))
    h = m.hist_norm(m.hist(h, mask=m.causal[:T, :T], is_causal=True))
    pad = cnt > 0
    pad[..., 0] |= ~pad.any(-1)
    x = F.embedding(idx.long(), tab) + m.copies(cnt.long().clamp(max=3)) + m.state_proj(h)[:, :, None]
    see = ~m.causal[:T, :T]
    for blk in m.blocks:
        x = blk(x, pad, h, see)
    return x, h, tab, m.head(x).squeeze(-1)


def pack_size(cnt):
    """[B] cards per pack: the non-empty slots of pack 2, which is complete even when P1P1 is missing."""
    return (cnt[:, N_PICKS:2 * N_PICKS].sum(-1) > 0).sum(-1)


def wheel_targets(idx, cnt, pick, npk):
    """Passed-card tokens at the 21 wheel slots, as candidate slots of pack t: copies passed [B,21,P],
    token mask, label (a copy is in the pack shown at t+8)."""
    wp = WP.to(idx.device)
    it, ct, ir, cr = idx[:, wp], cnt[:, wp].long(), idx[:, wp + N_SEATS], cnt[:, wp + N_SEATS]
    passed = ct - (it == pick[:, wp, None]).long()
    p = (wp % N_PICKS)[None]                                        # pick within the pack
    ok = p <= (npk[:, None] - N_SEATS - 1)                          # wheel-eligible for this pack size
    mask = (passed > 0) & ok[..., None]
    label = ((it[..., None] == ir[:, :, None]) & (cr[:, :, None] > 0)).any(-1)
    return passed, mask, label


class WheelHead(nn.Module):
    def __init__(self, d_in: int, d: int = 256, layers: int = 0, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(3 * d_in + 3, d)
        self.pick_pos = nn.Embedding(N_PACKS * W_MAX, d)
        self.layers = layers
        if layers:
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, feats, mask):
        """feats [B,21,P,3d+3] -> logits [B,21,P]; self-attention (if any) within one pack's passed cards."""
        B, S, P, _ = feats.shape
        z = self.inp(feats) + self.pick_pos.weight[None, :, None]
        if self.layers:
            m = mask.reshape(B * S, P)
            pad = ~m
            pad[:, 0] &= m.any(-1)
            z = self.enc(z.reshape(B * S, P, -1), src_key_padding_mask=pad).reshape(B, S, P, -1)
        return self.head(z).squeeze(-1)


def head_inputs(m, x7, npk, trunk_out=None):
    """Trunk -> per-token head inputs [B,21,P,3d+3], mask, label. Differentiable (the caller decides on
    no_grad); pass trunk_out to reuse a trunk() pass that is already computed."""
    idx, cnt, pick = x7[0], x7[1], x7[2]
    x, h, tab, logit = trunk(m, *x7) if trunk_out is None else trunk_out
    wp = WP.to(idx.device)
    passed, mask, label = wheel_targets(idx, cnt, pick, npk)
    lp = logit.masked_fill(cnt == 0, float("-inf")).log_softmax(-1)[:, wp]       # [B,21,P]
    took = (idx[:, wp] == pick[:, wp, None]).float().argmax(-1, keepdim=True)
    lpc = lp.nan_to_num(nan=-30.0, neginf=-30.0).clamp(min=-30)
    lpt = torch.gather(lpc, 2, took).expand_as(lpc)
    P = idx.shape[-1]
    ctx_t = torch.cat([h[:, wp], F.embedding(pick[:, wp], tab)], -1)[:, :, None].expand(-1, -1, P, -1)
    f = torch.cat([x[:, wp], ctx_t, lpc[..., None], lpt[..., None], passed.clamp(max=3).float()[..., None]], -1)
    # an empty pack (P1P1 missing) has an all -inf log-softmax, i.e. NaN; zero every unscored token, since a
    # NaN input poisons the weight gradients even when its loss is masked out
    f = f.masked_fill(~mask[..., None], 0)
    assert torch.isfinite(f).all(), "non-finite wheel-head input"
    return f, mask, label


def main(*kv):
    cfg = dict(CFG)
    cfg.update({k: value(v) for k, v in (a.split("=", 1) for a in kv)})
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    name = cfg["name"]
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cards = np.load(SHARDS / "cards.npz")
    pad = len(cards["name_id"])
    want = set(cfg["sets"].split(",")) if cfg["sets"] else None
    shards = [Shard(p, pad) for p in sorted(SHARDS.iterdir())
              if p.is_dir() and (p / "info.json").exists() and p.name.endswith(".PremierDraft")
              and (want is None or p.name.split(".")[0] in want)]
    names = [s.info["set"] for s in shards]

    def pairs(split, cap=0):
        out = []
        for i, s in enumerate(shards):
            rows = np.flatnonzero(s.split == split)
            if cap and len(rows) > cap:
                rows = np.sort(rng.choice(rows, cap, replace=False))
            out.append(np.stack([np.full(len(rows), i), rows], 1))
        return np.concatenate(out).astype(np.int64)

    train, val = pairs(0, cfg["max_per_set"]), pairs(1, cfg["val_per_set"])
    test_sets = set(cfg["test_sets"].split(",")) if cfg["test_sets"] else set(names)
    print(f"{name}: sets {names}, train {len(train):,} drafts, val {len(val):,} [{time.time() - t0:.0f}s]", flush=True)

    ck = torch.load(cfg["init"], map_location="cpu", weights_only=False)
    pc = ck["cfg"]
    feats = (cards["feats"] - ck["feat_mu"]) / ck["feat_sd"]
    pn = PickNet(pad, N_PICKS, d=pc["d"], heads=pc["heads"], hist_layers=pc["hist_layers"],
                 pack_layers=pc["pack_layers"], dropout=pc["dropout"], feats=feats, id_map=cards["name_id"],
                 n_ctx=ck["n_ctx"], n_sets=len(cards["sets"]), deck_head=pc["deck_w"] > 0)
    pn.load_state_dict(ck["state"])
    pn = pn.to(DEV).eval()
    for p in pn.parameters():
        p.requires_grad_(False)
    d_in = pc["d"]

    def batch_of(pr):
        x7 = [v.to(DEV) for v in to_torch(gather(shards, pr))[:7]]
        return x7, pack_size(x7[1])

    # checks: trunk() == PickNet.forward, and blanking every slot after t leaves the wheel inputs at t unchanged
    x7, npk = batch_of(val[:16])
    with torch.no_grad():
        gap = (trunk(pn, *x7)[3] - pn(*x7))[x7[1] > 0].abs().max().item()
        assert gap < 1e-3, f"trunk() drifted from PickNet.forward: {gap}"
        f_full = head_inputs(pn, x7, npk)[0]
        for t in (3, 18):
            cut = [v.clone() for v in x7]
            cut[0][:, t + 1:] = pad
            cut[1][:, t + 1:] = 0
            cut[2][:, t + 1:] = pad
            cut[3][:, t + 1:] = pad
            cut[4][:, t + 1:] = 0
            j = int((WP == t).nonzero())
            g = (head_inputs(pn, cut, npk)[0][:, j] - f_full[:, j]).abs().max().item()
            assert g < 1e-3, f"wheel inputs at slot {t} depend on later slots: {g}"
    print(f"checks passed (trunk gap {gap:.1e}, no dependence on slots after t)", flush=True)

    heads = {L: WheelHead(d_in, cfg["d"], L).to(DEV) for L in (0, 1)}
    per_epoch = len(train) // cfg["batch"]
    steps = 20 if cfg["bench"] else cfg["epochs"] * per_epoch
    opts = {L: torch.optim.AdamW(h.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) for L, h in heads.items()}
    scheds = {L: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=cfg["lr"], total_steps=cfg["epochs"] * per_epoch, pct_start=0.05)
              for L, o in opts.items()}
    n_par = {L: sum(p.numel() for p in h.parameters()) for L, h in heads.items()}
    print(f"frozen PickNet {sum(p.numel() for p in pn.parameters()):,} params; heads {n_par}; steps {steps} "
          f"({per_epoch}/epoch, batch {cfg['batch']})", flush=True)

    @torch.no_grad()
    def score(pr, keep=False):
        for h in heads.values():
            h.eval()
        out = {L: [] for L in heads}
        lab, sid, pos = [], [], []
        for i in range(0, len(pr), 128):
            x7, npk = batch_of(pr[i:i + 128])
            f, mask, label = head_inputs(pn, x7, npk)
            for L, h in heads.items():
                out[L].append(torch.sigmoid(h(f, mask))[mask].cpu().numpy())
            lab.append(label[mask].cpu().numpy())
            if keep:
                sid.append(x7[6][:, None, None].expand_as(mask)[mask].cpu().numpy())
                pos.append(WP.to(DEV)[None, :, None].expand_as(mask)[mask].cpu().numpy())
        for h in heads.values():
            h.train()
        res = {L: np.concatenate(v) for L, v in out.items()}
        return (res, np.concatenate(lab)) if not keep else (res, np.concatenate(lab), np.concatenate(pos))

    best = {L: (1e9, None) for L in heads}
    hist = []
    tt = time.time()
    gen = train_batches(shards, train, cfg["batch"], cfg["seed"], 0)
    tot = {L: 0.0 for L in heads}
    for step in range(1, steps + 1):
        b = next(gen)
        x7 = [v.to(DEV) for v in b[:7]]
        npk = pack_size(x7[1])
        with torch.no_grad():
            f, mask, label = head_inputs(pn, x7, npk)
        for L, h in heads.items():
            loss = F.binary_cross_entropy_with_logits(h(f, mask)[mask], label[mask].float())
            opts[L].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
            opts[L].step()
            scheds[L].step()
            tot[L] += loss.item()
        if cfg["bench"] and step == steps:
            dt = time.time() - tt
            print(f"bench: {dt / steps:.2f}s/step, {steps * cfg['batch'] / dt:.0f} drafts/s; "
                  f"epoch ~{per_epoch * dt / steps / 60:.0f} min", flush=True)
            return
        if step % 500 == 0:
            print(f"  step {step}/{steps} loss " + " ".join(f"L{L} {tot[L] / 500:.4f}" for L in heads)
                  + f" [{(time.time() - tt) / 60:.1f} min]", flush=True)
            tot = {L: 0.0 for L in heads}
        if step % per_epoch == 0 or step == steps:
            p, y = score(val)
            v = {L: metrics(y, p[L])["logloss"] for L in heads}
            hist.append({"step": step, **{f"val_L{L}": v[L] for L in heads}})
            print(f"VAL step {step}: " + " ".join(f"L{L} {v[L]:.4f}" for L in heads), flush=True)
            for L, h in heads.items():
                if v[L] < best[L][0]:
                    best[L] = (v[L], {k: w.clone() for k, w in h.state_dict().items()})
    for L, h in heads.items():
        h.load_state_dict(best[L][1])
        torch.save({"cfg": cfg, "layers": L, "state": best[L][1]}, OUT / "models" / f"{name}_L{L}.pt")

    res = {"cfg": cfg, "params": n_par, "history": hist, "best_val": {L: best[L][0] for L in heads}, "test": {}}
    for i, s in enumerate(shards):
        if names[i] not in test_sets:
            continue
        rows = np.flatnonzero(s.split == 2)
        p, y, pos = score(np.stack([np.full(len(rows), i), rows], 1), keep=True)
        res["test"][names[i]] = {f"L{L}": metrics(y, p[L]) for L in heads}
        res["test"][names[i]]["n"] = int(len(y))
        for L in heads:
            np.savez_compressed(OUT / f"test_{name}_L{L}_{names[i]}.npz", prob=p[L], label=y, slot=pos)
        print(f"TEST {names[i]}: n={len(y):,} " + " ".join(
            f"L{L} logloss {res['test'][names[i]][f'L{L}']['logloss']:.4f} auc {res['test'][names[i]][f'L{L}']['auc']:.4f}"
            for L in heads) + f" [{(time.time() - t0) / 60:.0f} min]", flush=True)
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"done [{(time.time() - t0) / 3600:.2f} h]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
