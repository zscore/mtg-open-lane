"""Train one features-only WheelNet on a group of sets with the same pack size; score every set's own test drafts.

    python scripts/train_wheelnet_allsets.py SETS [epochs] [per_set] [key=value ...]

SETS is a comma list of sets that share a pack size (WheelNet is built for one): e.g. the 14-card sets. Unlike
train_wheelnet_multiset.py (several sets -> one target), every set here contributes `per_set` random drafts from
its own chronological training split, and every set is scored on its own test split. These are the same splits
TableNet uses, so the predictions line up with TableNet's and the per-set WheelNets'. Sets whose files lack P1P1
are allowed. Cards are only their features, so one set of weights scores every set. 5% of the training drafts
are held out for picking the best epoch.

Writes, per set, results/{SET}.PremierDraft/results_{name}.json and test_predictions_{name}.parquet (the
train_nn.py layout), plus a summary results/tablenet/{name}_{n_picks}.json.
Keys: name ("wheelnet_allsets"), lr (1e-3), wd (0.01), batch (128), text (1), and WheelNet options.
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
from train_wheelnet_multiset import remap  # noqa: E402


def main(sets, epochs="3", per_set="20000", *kv):
    sets, epochs, per_set = sets.split(","), int(epochs), int(per_set)
    cfg = {k: value(v) for k, v in (a.split("=") for a in kv)}
    lr, wd, batch = cfg.pop("lr", 1e-3), cfg.pop("wd", 0.01), int(cfg.pop("batch", 128))
    text = int(cfg.pop("text", 1))
    name = cfg.pop("name", "wheelnet_allsets")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    t0 = time.time()

    # phase 1: training drafts of every set (test drafts are loaded later, one set at a time, to save memory)
    train_parts, feats, sizes, n_picks, n_ctx = [], [], [], None, None
    for s in sets:
        t = load_tensors(s)
        assert n_picks in (None, t.n_picks), f"{s} has {t.n_picks}-card packs; a group must share one pack size"
        n_picks = t.n_picks
        feats.append(card_inputs(t, True, True, bool(text)))
        ctx = draft_context(t)
        n_ctx = ctx.shape[1]
        passed, back = wheel_tensors(t)
        tr = splits_for(t, s, "PremierDraft")["train"]
        if len(tr) > per_set:
            tr = np.sort(rng.choice(tr, per_set, replace=False))
        train_parts.append(DraftDataset(t, passed, back, tr, ctx))
        sizes.append(len(t.names))
        print(f"  {s}: {len(t.names)} cards, {t.n_picks}-card packs, P1P1 missing {t.p1p1_missing}, "
              f"train {len(tr)} [{time.time() - t0:.0f}s]", flush=True)
        del t, passed, back
    offs = np.cumsum([0] + sizes)
    pad = int(offs[-1])
    train, val = [], []
    for ds, C, off in zip(train_parts, sizes, offs[:-1]):
        remap(ds, C, int(off), pad)
        cut = rng.permutation(len(ds))
        val.append(torch.utils.data.Subset(ds, np.sort(cut[: len(ds) // 20])))
        train.append(torch.utils.data.Subset(ds, np.sort(cut[len(ds) // 20:])))
    Fm = np.vstack(feats).astype(np.float32)
    Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-3)
    dl = lambda d, sh=False: torch.utils.data.DataLoader(d, batch_size=batch, shuffle=sh, collate_fn=collate)
    loaders = {"train": dl(torch.utils.data.ConcatDataset(train), True), "val": dl(torch.utils.data.ConcatDataset(val))}
    for k in ("cross", "others", "tied_bag"):
        if k in cfg:
            cfg[k] = bool(cfg[k])
    cfg.update(n_picks=n_picks, n_ctx=n_ctx)
    model = WheelNet(pad, card_feats=Fm, ids=False, **cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name}: sets={sets} {n_picks}-card packs, train drafts={sum(len(d) for d in train)} cards={pad} "
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
        print(f"epoch {ep}: train_loss={tot / n:.4f} val_logloss={vl:.4f} ({time.time() - te:.0f}s)", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    del loaders, train, val, train_parts
    summary = {"sets": sets, "n_picks": n_picks, "per_set": per_set, "epochs": epochs, "params": n_par,
               "best_val": best, "history": hist, "test": {}}
    (ROOT / "results" / "tablenet" / "models").mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "sets": sets, "state": best_state, "feat_mean_sd": None},
               ROOT / "results" / "tablenet" / "models" / f"{name}_{n_picks}.pt")

    # phase 2: each set's own test drafts
    for s, C, off in zip(sets, sizes, offs[:-1]):
        t = load_tensors(s)
        passed, back = wheel_tensors(t)
        test = DraftDataset(t, passed, back, splits_for(t, s, "PremierDraft")["test"], draft_context(t))
        del t, passed, back
        remap(test, C, int(off), pad)
        prob = predict(model, dl(test))
        test.idx = np.where(test.mask, test.idx - int(off), test.idx)  # local card ids in the parquet
        pred = flat(test, prob, name)
        out = ROOT / "results" / f"{s}.PremierDraft"
        out.mkdir(parents=True, exist_ok=True)
        pred.write_parquet(out / f"test_predictions_{name}.parquet")
        res = metrics(pred["wheeled"].to_numpy(), pred[name].to_numpy())
        res.update(n=pred.height, sets=sets, per_set=per_set, text=text, params=n_par, best_val=best,
                   history=hist, seconds=time.time() - t0)
        json.dump(res, open(out / f"results_{name}.json", "w"), indent=1)
        summary["test"][s] = {k: res[k] for k in ("logloss", "auc", "ece", "n")}
        (ROOT / "results" / "tablenet" / f"{name}_{n_picks}.json").write_text(json.dumps(summary, indent=1))
        print(f"TEST {s}: logloss {res['logloss']:.4f} auc {res['auc']:.4f} ece {res['ece']:.4f} n={pred.height:,} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    print(f"done [{(time.time() - t0) / 3600:.2f} h]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
