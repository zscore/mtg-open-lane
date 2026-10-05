"""Train a pick model: python scripts/train_picknet.py MSH picknet [epochs] [key=value ...]

Models: rating, pool, picknet (see openlane.picknet). key=value overrides model kwargs (d, heads,
hist_layers, pack_layers, dropout, history, others, cross, ids) and these switches:
  meta=1 rich=1 text=0 ctx=1   card meta (colour/rarity/mv/types), Scryfall rules features, rules-text
                               embedding, rank/skill/time context
  lr batch wd name val_only
Writes results/{SET}.PremierDraft/picknet/{name}.json (+ models/{name}.pt, test_{name}.npz).
Split: the same chronological 70/15/15 drafts as the wheel models (data/features/.../meta.pkl).
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import N_PACKS, ROOT, load_tensors  # noqa: E402
from openlane.nn import card_features, draft_context  # noqa: E402
from openlane.picknet import MODELS, PickDataset, collate, others_taken, pick_loss  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_KEYS = {"lr": 1e-3, "batch": 64, "wd": 0.01, "val_only": 0, "name": "", "meta": 1, "rich": 1, "text": 0, "ctx": 1,
              "event": "PremierDraft", "max_train": 0}


def value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def splits_for(t, set_code, event):
    p = ROOT / "data" / "features" / f"{set_code}.{event}" / "meta.pkl"
    if p.exists():
        sp = pickle.load(open(p, "rb"))["splits"]
        if sum(len(v) for v in sp.values()) == len(t.draft_id):
            return sp
    D = len(t.draft_id)  # chronological 70/15/15
    a, b = int(0.7 * D), int(0.85 * D)
    return {"train": np.arange(a), "val": np.arange(a, b), "test": np.arange(b, D)}


def card_inputs(t, meta: bool, rich: bool, text: bool = False):
    parts = []
    if meta:
        parts.append(card_features(t.meta))
    if rich:
        from openlane.cardinfo import card_info, card_matrix
        parts.append(card_matrix(card_info(t.names, t.set_code)))
    if text:  # scripts/card_text_embeddings.py
        parts.append(np.load(ROOT / "data" / "features" / f"{t.set_code}.PremierDraft" / "text_emb.npy"))
    return np.hstack(parts).astype(np.float32) if parts else None


@torch.no_grad()
def evaluate(model, loader, n_picks):
    """NLL and top-1 per pick slot, plus the chosen card's probability for every test pick."""
    model.eval()
    nll, top1, pchosen, valid = [], [], [], []
    for b in loader:
        x, (target, ok) = [v.to(DEV) for v in b[:6]], [v.to(DEV) for v in b[6:]]
        _, lp = pick_loss(model(*x), x[1], target, ok)
        lt = lp.gather(-1, target[..., None]).squeeze(-1)
        nll.append((-lt).cpu().numpy())
        top1.append((lp.argmax(-1) == target).cpu().numpy())
        valid.append(ok.cpu().numpy())
    nll, top1, valid = (np.concatenate(a) for a in (nll, top1, valid))
    T = nll.shape[1]
    res = {"nll": float(nll[valid].mean()), "top1": float(top1[valid].mean()),
           "nll_by_pick": [float(nll[valid[:, i], i].mean()) if valid[:, i].any() else None for i in range(T)],
           "top1_by_pick": [float(top1[valid[:, i], i].mean()) if valid[:, i].any() else None for i in range(T)]}
    for k in range(N_PACKS):
        s = slice(k * n_picks, (k + 1) * n_picks)
        res[f"top1_p{k + 1}"] = float(top1[:, s][valid[:, s]].mean())
    return res, nll, top1


def fit(model, loaders, epochs, lr, wd, n_picks):
    """AdamW + one-cycle; keeps the best validation-NLL state."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        model.train()
        t1, tot, n = time.time(), 0.0, 0
        for b in loaders["train"]:
            x, (target, ok) = [v.to(DEV) for v in b[:6]], [v.to(DEV) for v in b[6:]]
            loss, _ = pick_loss(model(*x), x[1], target, ok)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        vm, _, _ = evaluate(model, loaders["val"], n_picks)
        hist.append({"epoch": ep, "train_nll": tot / n, "val_nll": vm["nll"], "val_top1": vm["top1"]})
        print(f"  epoch {ep}: train {tot / n:.4f} val nll {vm['nll']:.4f} top1 {vm['top1']:.4f} "
              f"({time.time() - t1:.0f}s)", flush=True)
        if vm["nll"] < best:
            best, best_state = vm["nll"], {k: v.detach().clone() for k, v in model.state_dict().items()}
    return best, best_state, hist


def main(set_code: str, model_name: str = "picknet", epochs: str = "6", *kv: str):
    epochs = int(epochs)
    cfg = {k: value(v) for k, v in (a.split("=") for a in kv)}
    tr = {k: type(d)(cfg.pop(k, d)) for k, d in TRAIN_KEYS.items()}
    name = tr["name"] or model_name
    out = ROOT / "results" / f"{set_code}.{tr['event']}" / "picknet"
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()
    t = load_tensors(set_code, tr["event"])
    sp = splits_for(t, set_code, tr["event"])
    if tr["max_train"]:
        sp["train"] = sp["train"][-tr["max_train"]:]
    feats = card_inputs(t, tr["meta"], tr["rich"], tr["text"])
    ctx = draft_context(t) if tr["ctx"] else None
    oth = others_taken(t)
    ds = {k: PickDataset(t, v, ctx, oth) for k, v in sp.items()}
    n_cards, n_picks = len(t.names), t.n_picks
    del t, oth
    loaders = {k: torch.utils.data.DataLoader(v, batch_size=tr["batch"], shuffle=(k == "train"), collate_fn=collate)
               for k, v in ds.items()}
    for k in ("history", "others", "cross", "ids"):
        if k in cfg:
            cfg[k] = bool(cfg[k])
    model = MODELS[model_name](n_cards, n_picks=n_picks, feats=feats,
                               n_ctx=0 if ctx is None else ctx.shape[1], **cfg).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{set_code} {name}: model={model_name} cfg={cfg} train={tr} feats={None if feats is None else feats.shape} "
          f"params={n_par:,} train drafts={len(ds['train'])} device={DEV} [{time.time() - t0:.0f}s]", flush=True)

    best, best_state, hist = fit(model, loaders, epochs, tr["lr"], tr["wd"], n_picks)
    if tr["val_only"]:
        print(f"best val nll {best:.4f}")
        return
    model.load_state_dict(best_state)
    torch.save({"model": model_name, "cfg": cfg, "n_cards": n_cards, "n_picks": n_picks, "feats": feats,
                "train": tr, "state": best_state}, out / "models" / f"{name}.pt")
    res, nll, top1 = evaluate(model, loaders["test"], n_picks)
    res.update(model=model_name, cfg=cfg, train=tr, params=n_par, best_val_nll=best, history=hist,
               seconds=time.time() - t0)
    np.savez_compressed(out / f"test_{name}.npz", nll=nll.astype(np.float32), top1=top1)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"{set_code} {name} TEST nll {res['nll']:.4f} top1 {res['top1']:.4f} "
          f"(P1 {res['top1_p1']:.3f} P2 {res['top1_p2']:.3f} P3 {res['top1_p3']:.3f}) [{time.time() - t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
