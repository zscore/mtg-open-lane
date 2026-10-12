"""Train one features-only WheelNet on several sets; score a target set (day zero, or + its own drafts).

    python scripts/train_wheelnet_multiset.py MSH [epochs] [per_set] [key=value ...]

The wheel-model twin of train_picknet_multiset.py. Cards are only their features (colour/rarity/mana
value/types, Scryfall rules features, frozen rules-text embedding), so the same weights score every
set. Train sets: the other 14-pick sets with P1P1 recorded, `per_set` random drafts each, 5% held out
for early stopping. `with_target=1` adds the target set's own training drafts (chronological split).
Scored on the target's test drafts; writes results/{SET}.PremierDraft/results_{name}.json and
test_predictions_{name}.parquet (same layout as train_nn.py, so report/breakdown scripts read it).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtgdata.data import load_tensors  # noqa: E402
from mtgdata.encode import card_inputs, draft_context, splits_for  # noqa: E402
from openlane import ROOT  # noqa: E402
from openlane.evaluate import metrics  # noqa: E402
from openlane.features import wheel_tensors  # noqa: E402
from openlane.nn import DraftDataset, WheelNet, collate  # noqa: E402
from train_nn import DEV, flat, predict, value  # noqa: E402

SETS14 = ["OTJ", "DSK", "FDN", "DFT", "TDM", "FIN", "SOS", "MSH"]


def remap(ds: DraftDataset, C: int, off: int, pad: int):
    """Shift a set's local card indices into the global table (local padding C -> global pad)."""
    for a, dt in (("pack", np.int16), ("oth", np.int16), ("idx", np.int64)):
        x = getattr(ds, a).astype(np.int64)
        setattr(ds, a, np.where(x == C, pad, x + off).astype(dt))
    ds.pick = ds.pick + off
    return ds


def main(target="MSH", epochs="3", per_set="20000", *kv):
    epochs, per_set = int(epochs), int(per_set)
    cfg = {k: value(v) for k, v in (a.split("=") for a in kv)}
    lr, wd, batch = cfg.pop("lr", 1e-3), cfg.pop("wd", 0.01), int(cfg.pop("batch", 128))
    with_target = int(cfg.pop("with_target", 0))
    text = int(cfg.pop("text", 1))
    name = cfg.pop("name", f"wheelnet_multiset{'_plus_target' if with_target else '_zeroshot'}")
    sets = [s for s in SETS14 if s != target] + ([target] if with_target else [])
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    t0 = time.time()
    parts, feats = [], []
    for s in dict.fromkeys(sets + [target]):
        t = load_tensors(s)
        assert t.n_picks == 14 and not t.p1p1_missing
        feats.append(card_inputs(t, True, True, bool(text)))
        ctx = draft_context(t)
        passed, back = wheel_tensors(t)
        if s == target:
            sp = splits_for(t, s, "PremierDraft")
            use = {"test": sp["test"], **({"train": sp["train"]} if with_target else {})}
        else:
            use = {"train": np.sort(rng.choice(len(t.draft_id), min(per_set, len(t.draft_id)), replace=False))}
        parts.append((s, {k: DraftDataset(t, passed, back, v, ctx) for k, v in use.items()}, len(t.names)))
        print(f"  {s}: {len(t.names)} cards, " + ", ".join(f"{k} {len(v)}" for k, v in use.items())
              + f" [{time.time() - t0:.0f}s]", flush=True)
        del t, passed, back
    offs = np.cumsum([0] + [c for _, _, c in parts])
    pad = int(offs[-1])
    train, val, test, test_off = [], [], None, 0
    for (s, dss, C), off in zip(parts, offs[:-1]):
        for k, ds in dss.items():
            remap(ds, C, int(off), pad)
            if k == "test":
                test, test_off = ds, int(off)
            else:
                cut = rng.permutation(len(ds))
                val.append(torch.utils.data.Subset(ds, np.sort(cut[: len(ds) // 20])))
                train.append(torch.utils.data.Subset(ds, np.sort(cut[len(ds) // 20:])))
    Fm = np.vstack(feats).astype(np.float32)
    Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-3)
    dl = lambda d, sh=False: torch.utils.data.DataLoader(d, batch_size=batch, shuffle=sh, collate_fn=collate)
    loaders = {"train": dl(torch.utils.data.ConcatDataset(train), True),
               "val": dl(torch.utils.data.ConcatDataset(val)), "test": dl(test)}
    for k in ("cross", "others", "tied_bag"):
        if k in cfg:
            cfg[k] = bool(cfg[k])
    cfg.update(n_picks=14, n_ctx=test.ctx.shape[1])
    model = WheelNet(pad, card_feats=Fm, ids=False, **cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name}: target={target} sets={sets} train drafts={sum(len(d) for d in train)} cards={pad} "
          f"feat dim={Fm.shape[1]} params={n_par:,} steps={steps} [{time.time() - t0:.0f}s]", flush=True)

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for *x, label, mask, ctx in loaders["val"]:
                logit = model(*(v.to(DEV) for v in x), mask.to(DEV), ctx.to(DEV))
                m = mask.to(DEV)
                tot += F.binary_cross_entropy_with_logits(logit[m], label.to(DEV)[m].float(), reduction="sum").item()
                n += int(m.sum())
        return tot / n

    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        model.train()
        te, tot, n = time.time(), 0.0, 0
        for b in loaders["train"]:
            *x, label, mask, ctx = (v.to(DEV) for v in b)
            loss = F.binary_cross_entropy_with_logits(model(*x, mask, ctx)[mask], label[mask].float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        vl = val_loss()
        hist.append({"epoch": ep, "train": tot / n, "val": vl})
        print(f"epoch {ep}: train_loss={tot/n:.4f} val_logloss={vl:.4f} ({time.time()-te:.0f}s)", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out = ROOT / "results" / f"{target}.PremierDraft"
    torch.save({"cfg": cfg, "sets": sets, "state": best_state}, out / "models" / f"{name}.pt")
    prob = predict(model, loaders["test"])
    test.idx = np.where(test.mask, test.idx - test_off, test.idx)  # local card ids in the parquet
    pred = flat(test, prob, name)
    pred.write_parquet(out / f"test_predictions_{name}.parquet")
    res = metrics(pred["wheeled"].to_numpy(), pred[name].to_numpy())
    res.update(sets=sets, per_set=per_set, with_target=with_target, text=text, params=n_par, best_val=best,
               history=hist, seconds=time.time() - t0)
    json.dump(res, open(out / f"results_{name}.json", "w"), indent=1)
    print(f"{target} {name} TEST logloss {res['logloss']:.4f} auc {res['auc']:.4f} ece {res['ece']:.4f} "
          f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
