"""TableNet, step 2: unfreeze the all-sets PickNet and train it together with the wheel head.

    python scripts/train_tablenet_joint.py [key=value ...]

Starts from step 1 (scripts/train_tablenet.py): the all-sets PickNet `frahob_freeze` and the trained L=1 wheel
head. Everything trains, on PickNet's own objective plus the wheel loss:

    loss = pick NLL + deck_w * deck BCE + wheel_w * wheel BCE      (each a mean over its own items)

PickNet gets a low learning rate (it is already trained), the head a higher one. Same Premier training drafts
as step 1 (up to max_per_set per set, all 19 sets), one-cycle schedule, batch 32. Scores every set's Premier test
drafts for picks (NLL, top-1, top-1 on non-forced picks) and wheels (log loss, AUC) in one pass.
wheel_w=0 is the pick-only control: the same fine-tune without the wheel loss.

Keys: init, head (step 1 L=1 head), wheel_w (0.3), deck_w (0.3), lr_trunk (5e-5), lr_head (3e-4), epochs (1),
batch (32), max_per_set (15000), val_per_set (1000), val_every (2000), name ("tablenet_joint"), test_sets, bench.
Writes results/tablenet/{name}.json and models/{name}.pt (PickNet state + head state).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tablenet import (DEV, OUT, PICKNET, SHARDS, PickNet, Shard, WheelHead, gather, head_inputs,  # noqa: E402
                            metrics, pack_size, to_torch, train_batches, trunk, value)
from picknet.model import pick_loss  # noqa: E402

CFG = dict(init=str(PICKNET / "results" / "picknet_all" / "models" / "frahob_freeze.pt"),
           head=str(OUT / "models" / "tablenet_L1.pt"), wheel_w=0.3, deck_w=0.3, lr_trunk=5e-5, lr_head=3e-4,
           wd=0.01, epochs=1, batch=32, max_per_set=15000, val_per_set=1000, val_every=2000,
           name="tablenet_joint", test_sets="", bench=0, seed=0)


def deck_loss(m, h, tab, pick, valid, rate):
    """PickNet's deck-inclusion loss, as in train_picknet_all.losses()."""
    dl = m.deck_logits(h, tab, pick)
    T = rate.shape[1]
    known = rate < 255
    y = rate.float() / 254
    tt = torch.arange(T, device=h.device)
    msk = (tt[None, :, None] > tt[None, None, :]) & valid[:, :, None] & known[:, None, :]
    w = msk * ((tt[None, :, None] + 1) / T)
    bce = F.binary_cross_entropy_with_logits(dl.float(), y[:, None, :].expand_as(dl), reduction="none")
    return (bce * w).sum() / w.sum().clamp(min=1)


def forward(pn, head, b):
    """One batch -> pick NLL, log-probs, wheel logits/mask/label, and the trunk outputs."""
    x7 = [v.to(DEV, non_blocking=True) for v in b[:7]]
    target, valid = b[7].to(DEV), b[8].to(DEV)
    out = trunk(pn, *x7)
    nll, lp = pick_loss(out[3].float(), x7[1], target, valid)
    f, mask, label = head_inputs(pn, x7, pack_size(x7[1]), trunk_out=out)
    return x7, out, nll, lp, head(f, mask), mask, label


def main(*kv):
    cfg = dict(CFG)
    cfg.update({k: value(v) for k, v in (a.split("=", 1) for a in kv)})
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])  # same draws as step 1, so the same training drafts
    name = cfg["name"]
    t0 = time.time()

    cards = np.load(SHARDS / "cards.npz")
    pad = len(cards["name_id"])
    shards = [Shard(p, pad) for p in sorted(SHARDS.iterdir())
              if p.is_dir() and (p / "info.json").exists() and p.name.endswith(".PremierDraft")]
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

    ck = torch.load(cfg["init"], map_location="cpu", weights_only=False)
    pc = ck["cfg"]
    feats = (cards["feats"] - ck["feat_mu"]) / ck["feat_sd"]
    pn = PickNet(pad, 15, d=pc["d"], heads=pc["heads"], hist_layers=pc["hist_layers"], pack_layers=pc["pack_layers"],
                 dropout=pc["dropout"], feats=feats, id_map=cards["name_id"], n_ctx=ck["n_ctx"],
                 n_sets=len(cards["sets"]), deck_head=True)
    pn.load_state_dict(ck["state"])
    pn = pn.to(DEV)
    hk = torch.load(cfg["head"], map_location="cpu", weights_only=False)
    head = WheelHead(pc["d"], hk["cfg"]["d"], hk["layers"]).to(DEV)
    head.load_state_dict(hk["state"])
    print(f"{name}: sets {names}, train {len(train):,} drafts, val {len(val):,}; PickNet "
          f"{sum(p.numel() for p in pn.parameters()):,} + head {sum(p.numel() for p in head.parameters()):,} params; "
          f"cfg {cfg} [{time.time() - t0:.0f}s]", flush=True)

    @torch.no_grad()
    def score(pr, batch=128):
        """Pick NLL / top-1 / non-forced top-1 and wheel log loss / AUC over the drafts in pr."""
        pn.eval()
        head.eval()
        nll_sum, hit, n, hit_nf, n_nf, wp, wy = 0.0, 0, 0, 0, 0, [], []
        for i in range(0, len(pr), batch):
            b = to_torch(gather(shards, pr[i:i + batch]))
            x7, out, _, lp, wl, mask, label = forward(pn, head, b)
            target, valid = b[7].to(DEV), b[8].to(DEV)
            lt = lp.gather(-1, target[..., None]).squeeze(-1)
            h_ = lp.argmax(-1) == target
            nf = valid & ((x7[1] > 0).sum(-1) > 1)
            nll_sum += float(-lt[valid].sum())
            hit, n = hit + int(h_[valid].sum()), n + int(valid.sum())
            hit_nf, n_nf = hit_nf + int(h_[nf].sum()), n_nf + int(nf.sum())
            wp.append(torch.sigmoid(wl)[mask].cpu().numpy())
            wy.append(label[mask].cpu().numpy())
        pn.train()
        head.train()
        wm = metrics(np.concatenate(wy), np.concatenate(wp))
        return {"pick_nll": nll_sum / n, "pick_top1": hit / n, "pick_top1_nonforced": hit_nf / n_nf, "picks": n,
                "wheel_logloss": wm["logloss"], "wheel_auc": wm["auc"], "wheel_ece": wm["ece"],
                "wheel_n": int(sum(len(y) for y in wy))}

    # check: the scorer reproduces the published pick numbers of the starting model (MSH test, all_d256_ep3.json)
    msh = names.index("MSH")
    rows = np.flatnonzero(shards[msh].split == 2)
    s0 = score(np.stack([np.full(len(rows), msh), rows], 1))
    ref = json.loads((PICKNET / "results" / "picknet_all" / "all_d256_ep3.json").read_text())["test"]["MSH"]
    print(f"start, MSH test: pick top-1 {s0['pick_top1']:.4f} (published {ref['top1']:.4f}), nll {s0['pick_nll']:.4f} "
          f"(published {ref['nll']:.4f}); wheel {s0['wheel_logloss']:.4f} (step 1: 0.4267) [{time.time() - t0:.0f}s]",
          flush=True)
    assert abs(s0["pick_top1"] - ref["top1"]) < 1e-3, "scorer does not reproduce PickNet's published MSH numbers"

    per_epoch = len(train) // cfg["batch"]
    steps = 20 if cfg["bench"] else cfg["epochs"] * per_epoch
    groups = [{"params": list(pn.parameters()), "lr": cfg["lr_trunk"]},
              {"params": list(head.parameters()), "lr": cfg["lr_head"]}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg["lr_trunk"], cfg["lr_head"]],
                                                total_steps=cfg["epochs"] * per_epoch, pct_start=0.05)
    pn.train()
    head.train()
    gen = train_batches(shards, train, cfg["batch"], cfg["seed"], 0)
    hist, tot, tt = [], {"nll": 0.0, "deck": 0.0, "wheel": 0.0}, time.time()
    for step in range(1, steps + 1):
        b = next(gen)
        x7, (x, h, tab, logit), nll, lp, wl, mask, label = forward(pn, head, b)
        dk = deck_loss(pn, h, tab, x7[2], b[8].to(DEV), b[9].to(DEV)) if cfg["deck_w"] else torch.zeros((), device=DEV)
        wb = F.binary_cross_entropy_with_logits(wl[mask], label[mask].float())
        loss = nll + cfg["deck_w"] * dk + cfg["wheel_w"] * wb
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(pn.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        sched.step()
        for k, v in (("nll", nll), ("deck", dk), ("wheel", wb)):
            tot[k] += v.item()
        if cfg["bench"] and step == steps:
            dt = time.time() - tt
            print(f"bench: {dt / steps:.2f}s/step, {steps * cfg['batch'] / dt:.0f} drafts/s; "
                  f"epoch ~{per_epoch * dt / steps / 60:.0f} min", flush=True)
            return
        if step % 500 == 0:
            print(f"  step {step}/{steps} pick nll {tot['nll'] / 500:.4f} deck {tot['deck'] / 500:.4f} "
                  f"wheel {tot['wheel'] / 500:.4f} [{(time.time() - tt) / 60:.1f} min]", flush=True)
            tot = {k: 0.0 for k in tot}
        if step % cfg["val_every"] == 0 or step == steps:
            v = score(val)
            hist.append({"step": step, **v})
            print(f"VAL step {step}: pick nll {v['pick_nll']:.4f} top-1 {v['pick_top1']:.4f}  "
                  f"wheel {v['wheel_logloss']:.4f} auc {v['wheel_auc']:.4f}", flush=True)
    torch.save({"cfg": cfg, "picknet": pn.state_dict(), "head": head.state_dict(), "head_layers": hk["layers"],
                "head_d": hk["cfg"]["d"], "feat_mu": ck["feat_mu"], "feat_sd": ck["feat_sd"], "n_ctx": ck["n_ctx"]},
               OUT / "models" / f"{name}.pt")

    res = {"cfg": cfg, "start_msh": s0, "history": hist, "test": {}}
    for i, s in enumerate(shards):
        if names[i] not in test_sets:
            continue
        rows = np.flatnonzero(s.split == 2)
        r = score(np.stack([np.full(len(rows), i), rows], 1))
        res["test"][names[i]] = r
        print(f"TEST {names[i]}: pick top-1 {r['pick_top1']:.4f} nll {r['pick_nll']:.4f}  wheel {r['wheel_logloss']:.4f} "
              f"auc {r['wheel_auc']:.4f} n={r['wheel_n']:,} [{(time.time() - t0) / 60:.0f} min]", flush=True)
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"done [{(time.time() - t0) / 3600:.2f} h]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
