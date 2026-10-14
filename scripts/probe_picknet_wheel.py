"""Frozen-PickNet wheel probe: does the pick model already know what will wheel?

    python scripts/probe_picknet_wheel.py MSH [epochs] [key=value ...]

Loads picknet's features-only `multiset_plus_target` model for the target set, freezes it, and trains
only a small wheel head on its internals. Same sets, per-set draft samples, 5% early-stopping split
and test drafts as train_wheelnet_multiset.py with_target=1, so the result compares directly with
`wheelnet_multiset_plus_target`.

What the head sees for a passed card c at pick t (a wheel-eligible pick, 0..5 of a round):
  - PickNet's final candidate token for c (pack self-attention + cross-attention to history <= t)
  - the history state at t (packs <= t, picks < t)
  - the card you took at t (its frozen embedding and log-prob), c's pick log-prob, copies passed
Nothing after pick t, so the returning pack at t+8 cannot leak in. That matches what WheelNet sees.

Keys: layers (0 = per-card MLP, 1+ = that many self-attention layers over the passed cards),
d (head width), lr, batch, wd, name, limit (first N drafts per split, for smoke tests),
ckpt (path to the PickNet checkpoint).
Writes results/{SET}.PremierDraft/results_{name}.json and test_predictions_{name}.parquet.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtgdata.data import N_PACKS, N_SEATS, load_tensors  # noqa: E402
from mtgdata.encode import draft_context, pack_tokens, splits_for  # noqa: E402
from openlane import ROOT  # noqa: E402
from openlane.evaluate import metrics  # noqa: E402
from openlane.features import wheel_tensors  # noqa: E402
from picknet.model import PickDataset, PickNet, bag  # noqa: E402
from train_nn import DEV, flat, value  # noqa: E402

SETS14 = ["OTJ", "DSK", "FDN", "DFT", "TDM", "FIN", "SOS", "MSH"]
PICKNET = ROOT.parent / "picknet" / "results"
N_PICKS, W = 14, 14 - N_SEATS
WP = [k * N_PICKS + p for k in range(N_PACKS) for p in range(W)]  # history positions of wheel picks


class ProbeDataset(PickDataset):
    """PickNet's inputs plus WheelNet's passed-card tokens and labels [3, 6, 13]."""

    def __init__(self, t, drafts, ctx, passed, back):
        super().__init__(t, drafts, ctx)
        sel = slice(drafts[0], drafts[-1] + 1) if (np.diff(drafts) == 1).all() else drafts
        self.widx, self.wcop, self.label, self.mask = pack_tokens(passed[sel], back[sel], self.C)

    def __getitem__(self, i):
        return (*super().__getitem__(i)[:6], self.widx[i], self.wcop[i], self.label[i], self.mask[i])


def collate(batch):
    return [torch.from_numpy(np.stack(c)) for c in zip(*batch)]


def remap(ds: ProbeDataset, off: int, pad: int):
    """Shift a set's local card indices into the global table (local padding C -> global pad)."""
    for a in ("idx", "oidx", "widx"):
        x = getattr(ds, a).astype(np.int64)
        setattr(ds, a, np.where(x == ds.C, pad, x + off).astype(np.int32 if a != "widx" else np.int64))
    ds.pick = ds.pick + off
    return ds


@torch.no_grad()
def trunk(m: PickNet, idx, cnt, pick, oidx, ocnt, ctx):
    """PickNet.forward (eval mode, history + others on, no set embedding) up to the head, keeping
    the candidate tokens. Returns candidates x [B,T,P,d], states h [B,T,d], card table, pick logits."""
    B, T, P = idx.shape
    tab = m.cards()
    e_pick = F.embedding(pick, tab)
    prev = torch.cat([torch.zeros_like(e_pick[:, :1]), e_pick[:, :-1]], 1)
    h = m.pos.weight[None, :T] + m.pack_bag(bag(tab, idx, cnt))
    if m.n_ctx:
        h = h + m.ctx_mlp(ctx)[:, None]
    h = h + m.pick_in(prev) + m.oth_bag(bag(tab, oidx, ocnt))
    h = m.hist_norm(m.hist(h, mask=m.causal[:T, :T], is_causal=True))
    pad = cnt > 0
    pad[..., 0] |= ~pad.any(-1)
    x = F.embedding(idx.long(), tab) + m.copies(cnt.long().clamp(max=3)) + m.state_proj(h)[:, :, None]
    see = ~m.causal[:T, :T]
    for blk in m.blocks:
        x = blk(x, pad, h, see)
    return x, h, tab, m.head(x).squeeze(-1)


class WheelProbe(nn.Module):
    def __init__(self, d_in: int, d: int = 192, layers: int = 0, heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(3 * d_in + 3, d)
        self.pick_pos = nn.Embedding(N_PACKS * W, d)
        self.layers = layers
        if layers:
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, m: PickNet, idx, cnt, pick, oidx, ocnt, ctx, widx, wcop, mask):
        B = idx.shape[0]
        x, h, tab, logit = trunk(m, idx, cnt, pick, oidx, ocnt, ctx)
        wp = torch.tensor(WP, device=idx.device)
        lp = logit.masked_fill(cnt == 0, float("-inf")).log_softmax(-1)[:, wp]       # [B, 18, P]
        pidx = idx[:, wp].long()                                                     # [B, 18, P]
        widx = widx.reshape(B, len(WP), -1)                                          # [B, 18, M]
        slot = (widx[..., None] == pidx[:, :, None]).float().argmax(-1)              # [B, 18, M]
        xc = torch.gather(x[:, wp], 2, slot[..., None].expand(-1, -1, -1, x.shape[-1]))
        lpc = torch.gather(lp, 2, slot).clamp(min=-30)
        took = (pidx == pick[:, wp, None]).float().argmax(-1, keepdim=True)          # slot of your pick
        lpt = torch.gather(lp, 2, took).expand_as(lpc)
        M = widx.shape[-1]
        ctx_t = torch.cat([h[:, wp], F.embedding(pick[:, wp], tab)], -1)[:, :, None].expand(-1, -1, M, -1)
        cop = wcop.reshape(B, len(WP), M).float().clamp(max=3)
        z = torch.cat([xc, ctx_t, lpc[..., None], lpt[..., None], cop[..., None]], -1)
        z = self.inp(z) + self.pick_pos.weight[None, :, None]                        # [B, 18, M, d]
        if self.layers:
            m_ = mask.reshape(B * len(WP), M)
            pad = ~m_
            pad[:, 0] &= m_.any(-1)
            z = self.enc(z.reshape(B * len(WP), M, -1), src_key_padding_mask=pad).reshape(B, len(WP), M, -1)
        return self.head(z).squeeze(-1).reshape(B, N_PACKS, W, M)


def main(target="MSH", epochs="3", *kv):
    epochs = int(epochs)
    cfg = {k: value(v) for k, v in (a.split("=") for a in kv)}
    lr, wd, batch = cfg.pop("lr", 1e-3), cfg.pop("wd", 0.01), int(cfg.pop("batch", 128))
    layers, d = int(cfg.pop("layers", 0)), int(cfg.pop("d", 192))
    limit = int(cfg.pop("limit", 0))
    name = cfg.pop("name", f"probe_picknet_L{layers}")
    ckpt = Path(cfg.pop("ckpt", PICKNET / f"{target}.PremierDraft" / "picknet" / "models" / "multiset_plus_target.pt"))
    assert not cfg, f"unknown keys {cfg}"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sets, per_set = ck["sets"], 20000
    assert sets[-1] == target, "the probe expects a with_target=1 PickNet"
    rng = np.random.default_rng(0)  # same draws as both multiset trainers
    torch.manual_seed(0)
    t0 = time.time()
    parts = []
    for s in dict.fromkeys(sets + [target]):
        t = load_tensors(s)
        assert t.n_picks == N_PICKS and not t.p1p1_missing
        ctx = draft_context(t)
        passed, back = wheel_tensors(t)
        if s == target:
            sp = splits_for(t, s, "PremierDraft")
            use = {"test": sp["test"], "train": sp["train"]}
        else:
            use = {"train": np.sort(rng.choice(len(t.draft_id), min(per_set, len(t.draft_id)), replace=False))}
        if limit:
            use = {k: v[:limit] for k, v in use.items()}
        parts.append((s, {k: ProbeDataset(t, v, ctx, passed, back) for k, v in use.items()}, len(t.names)))
        print(f"  {s}: {len(t.names)} cards, " + ", ".join(f"{k} {len(v)}" for k, v in use.items())
              + f" [{time.time() - t0:.0f}s]", flush=True)
        del t, passed, back
    offs = np.cumsum([0] + [c for _, _, c in parts])
    pad = int(offs[-1])
    train, val, test, test_off = [], [], None, 0
    for (s, dss, C), off in zip(parts, offs[:-1]):
        for k, ds in dss.items():
            remap(ds, int(off), pad)
            if k == "test":
                test, test_off = ds, int(off)
            else:
                cut = rng.permutation(len(ds))
                val.append(torch.utils.data.Subset(ds, np.sort(cut[: len(ds) // 20])))
                train.append(torch.utils.data.Subset(ds, np.sort(cut[len(ds) // 20:])))
    dl = lambda d_, sh=False: torch.utils.data.DataLoader(d_, batch_size=batch, shuffle=sh, collate_fn=collate)
    loaders = {"train": dl(torch.utils.data.ConcatDataset(train), True),
               "val": dl(torch.utils.data.ConcatDataset(val)), "test": dl(test)}

    feats = ck["state"]["cards.feats"][:-1].numpy()
    assert len(feats) == pad, f"card table {len(feats)} != {pad}: sets or card data changed since training"
    pn = PickNet(pad, N_PICKS, feats=feats, ids=False, n_ctx=test.ctx.shape[1], **ck["cfg"])
    pn.load_state_dict(ck["state"])
    pn = pn.to(DEV).eval()
    for p in pn.parameters():
        p.requires_grad_(False)
    b = next(iter(loaders["test"]))
    x6 = [v.to(DEV) for v in b[:6]]
    gap = (trunk(pn, *x6)[3] - pn(*x6)).abs()[x6[1] > 0].max().item()
    assert gap < 1e-3, f"trunk() drifted from PickNet.forward: {gap}"

    probe = WheelProbe(pn.pos.weight.shape[1], d, layers).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    n_par = sum(p.numel() for p in probe.parameters())
    print(f"{name}: frozen {ckpt.name} ({sum(p.numel() for p in pn.parameters()):,} params), "
          f"probe params={n_par:,} layers={layers} train drafts={sum(len(x) for x in train)} steps={steps} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    def run(loader):
        probe.eval()
        out = []
        with torch.no_grad():
            for bt in loader:
                *x, widx, wcop, _, mask = (v.to(DEV) for v in bt)
                out.append(torch.sigmoid(probe(pn, *x, widx, wcop, mask)).cpu().numpy())
        return np.concatenate(out)

    def val_loss():
        p = run(loaders["val"])
        y = np.concatenate([ds.dataset.label[ds.indices] for ds in loaders["val"].dataset.datasets])
        m = np.concatenate([ds.dataset.mask[ds.indices] for ds in loaders["val"].dataset.datasets])
        return metrics(y[m], p[m])["logloss"]

    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        probe.train()
        te, tot, n = time.time(), 0.0, 0
        for i, bt in enumerate(loaders["train"]):
            *x, widx, wcop, label, mask = (v.to(DEV) for v in bt)
            logit = probe(pn, *x, widx, wcop, mask)
            loss = F.binary_cross_entropy_with_logits(logit[mask], label[mask].float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
            if i % 500 == 0:
                print(f"  ep {ep} step {i}/{len(loaders['train'])} loss {tot / n:.4f} ({time.time() - te:.0f}s)",
                      flush=True)
        vl = val_loss()
        hist.append({"epoch": ep, "train": tot / n, "val": vl})
        print(f"epoch {ep}: train_loss={tot / n:.4f} val_logloss={vl:.4f} ({time.time() - te:.0f}s)", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    out = ROOT / "results" / f"{target}.PremierDraft"
    torch.save({"layers": layers, "d": d, "ckpt": str(ckpt), "state": best_state}, out / "models" / f"{name}.pt")
    prob = run(loaders["test"])
    widx = np.where(test.mask, test.widx - test_off, test.widx)  # local card ids in the parquet
    pred = flat(SimpleNamespace(mask=test.mask, drafts=test.drafts, idx=widx, label=test.label), prob, name)
    pred.write_parquet(out / f"test_predictions_{name}.parquet")
    res = metrics(pred["wheeled"].to_numpy(), pred[name].to_numpy())
    res.update(sets=sets, picknet=str(ckpt), layers=layers, d=d, params=n_par, best_val=best, history=hist,
               n_test=pred.height, seconds=time.time() - t0)
    if not limit:
        json.dump(res, open(out / f"results_{name}.json", "w"), indent=1)
    print(f"{target} {name} TEST logloss {res['logloss']:.4f} auc {res['auc']:.4f} ece {res['ece']:.4f} "
          f"n={pred.height} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
