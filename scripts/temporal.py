"""Does the wheel model go stale as a format matures? python scripts/temporal.py MSH

Walk forward over the set's weeks (weeks with < MIN_WEEK drafts are merged into the one before).
For each target week w >= 1, every arm is scored on all of week w's drafts:

  (WheelNet = the wheelnet_time config: v2 plus days-into-format as draft context)
  frozen      WheelNet trained on week 0 only, never updated
  retrain     WheelNet trained from scratch on weeks 0..w-1
  finetune    last week's finetune model (frozen at w=1), trained 2 more epochs on week w-1 only
  replay      as finetune, but week w-1 plus an equal-sized random sample of older drafts
  lookup_*    per-card x pick empirical wheel rate, fit on week 0 (frozen) or weeks 0..w-1

Writes results/<SET>.<EVENT>/temporal.json. With --retrain-only, runs just retrain + the lookups
and writes temporal_retrain.json instead.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import ROOT, load_tensors  # noqa: E402
from openlane.evaluate import metrics  # noqa: E402
from openlane.features import wheel_tensors  # noqa: E402
from openlane.nn import DraftDataset, WheelNet, card_features, collate, draft_context  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MIN_WEEK = 3000
FT_EPOCHS, FT_LR = 2, 3e-4
CTX = {"rank": False, "skill": False, "time": True}  # same inputs as wheelnet_time


def epochs_for(n: int) -> int:
    """Same budget rule as run_all_sets.sh: bigger training sets get fewer passes."""
    return min(8, max(3, round(8 * 48000 / n)))


def loader(ds, shuffle):
    return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=shuffle, collate_fn=collate)


def fit(model, ds, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dl = loader(ds, True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl),
                                                pct_start=0.05)
    model.train()
    for _ in range(epochs):
        for batch in dl:
            *x, label, mask, ctx = (v.to(DEV) for v in batch)
            loss = F.binary_cross_entropy_with_logits(model(*x, mask, ctx)[mask], label[mask].float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model


def score(model, ds):
    model.eval()
    out = []
    with torch.no_grad():
        for *x, label, mask, ctx in loader(ds, False):
            out.append(torch.sigmoid(model(*(v.to(DEV) for v in x), mask.to(DEV), ctx.to(DEV))).cpu().numpy())
    p = np.concatenate(out)
    return metrics(ds.label[ds.mask], p[ds.mask])


def lookup_rates(passed, back, drafts, prior_n=20.0):
    """[C, n_wheel] smoothed empirical wheel rate per card and pick."""
    n = (passed[drafts] > 0).sum((0, 1)).T.astype(np.float64)
    w = (back[drafts] > 0).sum((0, 1)).T.astype(np.float64)
    g = w.sum(0) / n.sum(0)
    return (w + prior_n * g) / (n + prior_n)


def lookup_score(rates, ds):
    p = rates[ds.idx.clip(max=len(rates) - 1), np.arange(ds.idx.shape[2])[None, None, :, None]]
    return metrics(ds.label[ds.mask], p[ds.mask])


def main(set_code: str, event: str = "PremierDraft", retrain_only: bool = False):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t = load_tensors(set_code, event)
    passed, back = wheel_tensors(t)
    feats = card_features(t.meta, types=False)
    ctx = draft_context(t, **CTX)
    C = len(t.names)

    ts = np.array(t.draft_time, dtype="datetime64[s]")  # 17Lands times are "YYYY-MM-DD HH:MM:SS"
    days = (ts - ts.min()).astype(np.int64) / 86400
    week = (days // 7).astype(int)
    for w in range(week.max(), 0, -1):  # fold thin trailing weeks into the previous one
        if (week == w).sum() < MIN_WEEK:
            week[week == w] = w - 1
    weeks = sorted(set(week))
    week = np.searchsorted(weeks, week)
    W = len(weeks)
    print(f"{set_code}: {W} weeks, drafts per week {np.bincount(week).tolist()}", flush=True)

    new = lambda: WheelNet(C, n_picks=t.n_picks, card_feats=feats, n_ctx=ctx.shape[1]).to(DEV)
    drafts = lambda sel: np.flatnonzero(sel)
    res = {"weeks": np.bincount(week).tolist(), "arms": {}}
    rec = lambda arm, w, m: res["arms"].setdefault(arm, {}).__setitem__(w, m)
    lookup0 = lookup_rates(passed, back, drafts(week == 0))
    frozen = finetune = replay = None

    for w in range(1, W):
        t0 = time.time()
        test = DraftDataset(t, passed, back, drafts(week == w), ctx)
        rec("lookup_frozen", w, lookup_score(lookup0, test))
        rec("lookup_refit", w, lookup_score(lookup_rates(passed, back, drafts(week < w)), test))

        tr = drafts(week < w)
        model = fit(new(), DraftDataset(t, passed, back, tr, ctx), epochs_for(len(tr)), 1e-3)
        rec("retrain", w, score(model, test))
        if retrain_only:
            pass
        elif w == 1:
            frozen = model
            finetune = new(); finetune.load_state_dict(model.state_dict())
            replay = new(); replay.load_state_dict(model.state_dict())
        else:
            last, old = drafts(week == w - 1), drafts(week < w - 1)
            finetune = fit(finetune, DraftDataset(t, passed, back, last, ctx), FT_EPOCHS, FT_LR)
            mix = np.sort(np.concatenate([last, rng.choice(old, min(len(old), len(last)), replace=False)]))
            replay = fit(replay, DraftDataset(t, passed, back, mix, ctx), FT_EPOCHS, FT_LR)
        if not retrain_only:
            rec("frozen", w, score(frozen, test))
            rec("finetune", w, score(finetune, test))
            rec("replay", w, score(replay, test))
        del model
        print(f"week {w}: " + "  ".join(f"{a}={v[w]['logloss']:.4f}" for a, v in res["arms"].items())
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "results" / f"{set_code}.{event}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("temporal_retrain.json" if retrain_only else "temporal.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--retrain-only"]
    main(*args, retrain_only="--retrain-only" in sys.argv)
