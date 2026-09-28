"""Train WheelNet: python scripts/train_nn.py MSH [epochs]"""
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
from openlane.nn import WHEEL_PICKS, DraftDataset, WheelNet, collate  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for pack, pick, idx, copies, label, mask in loader:
            logit = model(pack.to(DEV), pick.to(DEV), idx.to(DEV), copies.to(DEV), mask.to(DEV))
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def flat(ds, prob):
    """Long-format (draft_idx, pack, pick, card, wheeled, prob) for valid tokens."""
    d, k, p, j = np.nonzero(ds.mask)
    return pl.DataFrame({
        "draft_idx": ds.drafts[d], "pack_number": k.astype(np.int8), "pick_number": p.astype(np.int8),
        "card": ds.idx[d, k, p, j].astype(np.int16), "wheeled": ds.label[d, k, p, j].astype(np.int8),
        "wheelnet": prob[d, k, p, j].astype(np.float32),
    })


def main(set_code: str, epochs: str = "8", event: str = "PremierDraft"):
    epochs = int(epochs)
    torch.manual_seed(0)
    feat_dir = ROOT / "data" / "features" / f"{set_code}.{event}"
    out = ROOT / "results" / f"{set_code}.{event}"
    splits = pickle.load(open(feat_dir / "meta.pkl", "rb"))["splits"]
    t = load_tensors(set_code, event)
    passed, back = wheel_tensors(t)
    ds = {k: DraftDataset(t, passed, back, v) for k, v in splits.items()}
    del t, passed, back
    loaders = {k: torch.utils.data.DataLoader(v, batch_size=128, shuffle=(k == "train"),
                                              collate_fn=collate) for k, v in ds.items()}

    model = WheelNet(ds["train"].pack.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    print(f"device={DEV} params={sum(p.numel() for p in model.parameters()):,} steps={steps}")

    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for pack, pick, idx, copies, label, mask in loaders["train"]:
            pack, pick, idx, copies, label, mask = (x.to(DEV) for x in (pack, pick, idx, copies, label, mask))
            logit = model(pack, pick, idx, copies, mask)
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

    model.load_state_dict(best_state)
    torch.save(best_state, out / "models" / "wheelnet.pt")
    test = flat(ds["test"], predict(model, loaders["test"]))
    test.write_parquet(out / "test_predictions_wheelnet.parquet")
    res = metrics(test["wheeled"].to_numpy(), test["wheelnet"].to_numpy())
    print("wheelnet test", res)
    json.dump(res, open(out / "results_wheelnet.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
