"""Train one features-only pick model on several sets, test it on a set it never saw (day zero).

    python scripts/train_picknet_multiset.py MSH [epochs] [per_set] [key=value ...]

Cards are only their features (colour/rarity/mana value/types, Scryfall rules features, a frozen
rules-text embedding): no card ids, so an unseen set is scored by the same machinery (DraftFM's
setup). Train sets: the other 14-pick sets with P1P1 recorded (OTJ DSK FDN DFT TDM FIN SOS MSH, minus
the test set), `per_set` random drafts from each; 5% of those drafts are held out for early stopping.
`with_target=1` also adds the target set's own training drafts (same chronological split as
train_picknet.py), to see what target-set data adds on top. Scored on the target set's test drafts.
Writes results/{SET}.PremierDraft/picknet/{name}.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlane.data import ROOT, load_tensors  # noqa: E402
from openlane.nn import draft_context  # noqa: E402
from openlane.picknet import PickDataset, PickNet, collate  # noqa: E402
from train_picknet import DEV, card_inputs, evaluate, fit, splits_for, value  # noqa: E402

SETS14 = ["OTJ", "DSK", "FDN", "DFT", "TDM", "FIN", "SOS", "MSH"]


def remap(ds: PickDataset, off: int, pad: int):
    """Shift a set's local card indices into the global table (padding -> global pad)."""
    for a in ("idx", "oidx"):
        x = getattr(ds, a).astype(np.int32)
        setattr(ds, a, np.where(x == ds.C, pad, x + off).astype(np.int32))
    ds.pick = ds.pick + off
    return ds


def main(target="MSH", epochs="3", per_set="20000", *kv):
    epochs, per_set = int(epochs), int(per_set)
    cfg = {k: value(v) for k, v in (a.split("=") for a in kv)}
    lr, wd, batch = cfg.pop("lr", 1e-3), cfg.pop("wd", 0.01), int(cfg.pop("batch", 128))
    with_target = int(cfg.pop("with_target", 0))
    text = int(cfg.pop("text", 1))
    name = cfg.pop("name", f"multiset{'_plus_target' if with_target else '_zeroshot'}")
    sets = [s for s in SETS14 if s != target] + ([target] if with_target else [])
    rng = np.random.default_rng(0)
    t0 = time.time()
    # first pass: card counts per set, to fix the global table (pad = total)
    parts, feats, n_picks = [], [], None
    for s in dict.fromkeys(sets + [target]):
        t = load_tensors(s)
        n_picks = n_picks or t.n_picks
        assert t.n_picks == n_picks and not t.p1p1_missing
        feats.append(card_inputs(t, True, True, bool(text)))
        ctx = draft_context(t)
        if s == target:
            sp = splits_for(t, s, "PremierDraft")
            use = {"test": sp["test"], **({"train": sp["train"]} if with_target else {})}
        else:
            use = {"train": np.sort(rng.choice(len(t.draft_id), min(per_set, len(t.draft_id)), replace=False))}
        parts.append((s, {k: PickDataset(t, v, ctx) for k, v in use.items()}, len(t.names)))
        print(f"  {s}: {len(t.names)} cards, " + ", ".join(f"{k} {len(v)}" for k, v in use.items())
              + f" [{time.time() - t0:.0f}s]", flush=True)
        del t
    offs = np.cumsum([0] + [c for _, _, c in parts])
    pad = int(offs[-1])
    train, val, test = [], [], None
    for (s, dss, C), off in zip(parts, offs[:-1]):
        for k, ds in dss.items():
            remap(ds, int(off), pad)
            if k == "test":
                test = ds
            else:  # hold out 5% for early stopping
                n = len(ds)
                cut = rng.permutation(n)
                val.append(torch.utils.data.Subset(ds, np.sort(cut[: n // 20])))
                train.append(torch.utils.data.Subset(ds, np.sort(cut[n // 20:])))
    F = np.vstack(feats).astype(np.float32)
    mu, sd = F.mean(0), F.std(0) + 1e-3
    F = (F - mu) / sd
    loaders = {"train": torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train), batch_size=batch,
                                                    shuffle=True, collate_fn=collate),
               "val": torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val), batch_size=batch,
                                                  collate_fn=collate),
               "test": torch.utils.data.DataLoader(test, batch_size=batch, collate_fn=collate)}
    n_ctx = test.ctx.shape[1]
    for k in ("history", "others", "cross"):
        if k in cfg:
            cfg[k] = bool(cfg[k])
    model = PickNet(pad, n_picks, feats=F, ids=False, n_ctx=n_ctx, **cfg).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name}: target={target} sets={sets} train drafts={sum(len(d) for d in train)} cards={pad} "
          f"feat dim={F.shape[1]} params={n_par:,} cfg={cfg} [{time.time() - t0:.0f}s]", flush=True)
    best, state, hist = fit(model, loaders, epochs, lr, wd, n_picks)
    model.load_state_dict(state)
    res, nll, top1 = evaluate(model, loaders["test"], n_picks)
    res.update(target=target, sets=sets, per_set=per_set, with_target=with_target, text=text, cfg=cfg, params=n_par,
               best_val_nll=best, history=hist, seconds=time.time() - t0)
    out = ROOT / "results" / f"{target}.PremierDraft" / "picknet"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    torch.save({"cfg": cfg, "state": state, "sets": sets, "feat_mu": mu, "feat_sd": sd}, out / "models" / f"{name}.pt")
    print(f"{target} {name} TEST nll {res['nll']:.4f} top1 {res['top1']:.4f} "
          f"(P1 {res['top1_p1']:.3f} P2 {res['top1_p2']:.3f} P3 {res['top1_p3']:.3f}) [{time.time() - t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
