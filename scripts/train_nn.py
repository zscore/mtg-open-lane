"""Train WheelNet: python scripts/train_nn.py MSH [epochs] [event] [key=value ...]

key=value pairs override WheelNet kwargs (d, heads, hist_layers, pack_layers, dropout, cross,
others, tied_bag), input switches (meta, types, rank, skill, time; all on by default) or training settings (lr, batch, wd). val_only=1 skips the test set and writes
nothing, for architecture search. name=X writes models/X.pt, test_predictions_X.parquet and
results_X.json instead of the default wheelnet files.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import ROOT, load_tensors  # noqa: E402
from openlane.evaluate import metrics  # noqa: E402
from openlane.features import wheel_tensors  # noqa: E402
from openlane.nn import DraftDataset, WheelNet, card_features, collate, draft_context  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for *x, label, mask, ctx in loader:
            logit = model(*(v.to(DEV) for v in x), mask.to(DEV), ctx.to(DEV))
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def flat(ds, prob, col="wheelnet"):
    """Long-format (draft_idx, pack, pick, card, wheeled, prob) for valid tokens."""
    d, k, p, j = np.nonzero(ds.mask)
    return pl.DataFrame({
        "draft_idx": ds.drafts[d], "pack_number": k.astype(np.int8), "pick_number": p.astype(np.int8),
        "card": ds.idx[d, k, p, j].astype(np.int16), "wheeled": ds.label[d, k, p, j].astype(np.int8),
        col: prob[d, k, p, j].astype(np.float32),
    })


TRAIN_KEYS = {"lr": 1e-3, "batch": 128, "wd": 0.01, "val_only": 0, "name": "wheelnet"}


def value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse(kv: list[str]):
    cfg = {k: value(v) for k, v in (a.split("=") for a in kv)}
    train = {k: type(d)(cfg.pop(k, d)) for k, d in TRAIN_KEYS.items()}
    return cfg, train


def main(set_code: str, epochs: str = "8", event: str = "PremierDraft", *kv: str):
    epochs = int(epochs)
    cfg, tr = parse(list(kv))
    torch.manual_seed(0)
    feat_dir = ROOT / "data" / "features" / f"{set_code}.{event}"
    out = ROOT / "results" / f"{set_code}.{event}"
    splits = pickle.load(open(feat_dir / "meta.pkl", "rb"))["splits"]
    t = load_tensors(set_code, event)
    passed, back = wheel_tensors(t)
    types = bool(cfg.pop("types", 1))
    feats = card_features(t.meta, types=types) if cfg.pop("meta", 1) else None
    inputs = {k: bool(cfg.pop(k, 1)) for k in ("rank", "skill", "time")}
    ctx = draft_context(t, **inputs)
    n_cards = len(t.names)
    cfg["n_picks"], cfg["n_ctx"] = t.n_picks, ctx.shape[1]
    ds = {k: DraftDataset(t, passed, back, v, ctx) for k, v in splits.items()}
    del t, passed, back
    loaders = {k: torch.utils.data.DataLoader(v, batch_size=tr["batch"], shuffle=(k == "train"),
                                              collate_fn=collate) for k, v in ds.items()}

    model = WheelNet(n_cards, card_feats=feats, **cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["wd"])
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr["lr"], total_steps=steps, pct_start=0.05)
    print(f"cfg={cfg} train={tr} inputs={dict(inputs, types=types)} "
          f"card_feats={None if feats is None else feats.shape[1]}")
    print(f"device={DEV} params={sum(p.numel() for p in model.parameters()):,} steps={steps}")

    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in loaders["train"]:
            *x, label, mask, ctx = (v.to(DEV) for v in batch)
            logit = model(*x, mask, ctx)
            loss = F.binary_cross_entropy_with_logits(logit[mask], label[mask].float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        pv = predict(model, loaders["val"])
        vm = metrics(ds["val"].label[ds["val"].mask], pv[ds["val"].mask])
        print(f"epoch {ep}: train_loss={tot/n:.4f} val_logloss={vm['logloss']:.4f} "
              f"val_auc={vm['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vm["logloss"] < best:
            best, best_state = vm["logloss"], {k: v.clone() for k, v in model.state_dict().items()}

    print(f"best val_logloss={best:.4f}")
    if tr["val_only"]:
        return
    model.load_state_dict(best_state)
    name = tr["name"]
    torch.save({"cfg": cfg, "card_feats": feats, "inputs": inputs, "state": best_state},
               out / "models" / f"{name}.pt")
    test = flat(ds["test"], predict(model, loaders["test"]), name)
    test.write_parquet(out / f"test_predictions_{name}.parquet")
    res = metrics(test["wheeled"].to_numpy(), test[name].to_numpy())
    print(f"{name} test", res)
    json.dump(res, open(out / f"results_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
