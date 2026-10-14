"""TableNet diagnostic D1: do the pick, deck and wheel losses pull PickNet's shared weights the same way?

    python scripts/grad_cosine.py [key=value ...]

For each checkpoint, runs `batches` training batches (train mode, same draws as train_tablenet_joint.py) and takes
the gradient of each loss separately with respect to PickNet's parameters (the wheel head is not shared, so it is
left out). Reports, per parameter group and overall:
  cos_batch   mean (and sd) over batches of the per-batch cosine between two losses' gradients
  cos_mean    cosine between the two losses' gradients averaged over all batches (the direction training
              follows on average; per-batch cosines are diluted toward 0 by minibatch noise)
  norm        mean per-batch gradient norm of each loss, unweighted (multiply by the loss weight to compare
              with the training step: pick 1, deck 0.3, wheel wheel_w)
  mnorm       norm of the batch-averaged gradient; mnorm/norm near 0 means the loss is at a stationary point
              there and its cos_mean is mostly the residual of a compromise between the losses
See paper/combined/gap-diagnosis.md (D1) for how to read them.

Keys: ckpts (comma list of name=path; "step1" means init + head), batches (300), batch (32), seed (1),
name ("grad_cosine"). Writes results/tablenet/{name}.json.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tablenet import DEV, OUT, SHARDS, PickNet, Shard, WheelHead, train_batches, value  # noqa: E402
from train_tablenet_joint import CFG as JCFG, deck_loss, forward  # noqa: E402

CFG = dict(ckpts="step1=step1,w0.3=" + str(OUT / "models" / "tablenet_joint.pt") + ",w1=" +
           str(OUT / "models" / "tablenet_joint_w1.pt"), batches=300, batch=32, seed=1, name="grad_cosine")
PAIRS = (("pick", "wheel"), ("deck", "wheel"), ("pick", "deck"))


def group(name):
    """Parameter name -> report group."""
    if name.startswith("cards"):
        return "card table"
    if name.startswith(("deck_", "pool_bag")):
        return "deck head"
    if name.startswith("head"):
        return "pick head"
    m = re.match(r"(blocks\.\d+)", name)
    if m:
        return "decoder " + m.group(1)
    m = re.match(r"hist\.layers\.(\d+)", name)
    if m:
        return f"history layer {m.group(1)}"
    if name.startswith("hist"):
        return "history (norm)"
    return "input embeddings"


def load(spec, pad, cards):
    """'step1' or a joint checkpoint path -> (PickNet, WheelHead), on DEV."""
    ck = torch.load(JCFG["init"], map_location="cpu", weights_only=False)
    pc = ck["cfg"]
    feats = (cards["feats"] - ck["feat_mu"]) / ck["feat_sd"]
    pn = PickNet(pad, 15, d=pc["d"], heads=pc["heads"], hist_layers=pc["hist_layers"], pack_layers=pc["pack_layers"],
                 dropout=pc["dropout"], feats=feats, id_map=cards["name_id"], n_ctx=ck["n_ctx"],
                 n_sets=len(cards["sets"]), deck_head=True)
    hk = torch.load(JCFG["head"], map_location="cpu", weights_only=False)
    head = WheelHead(pc["d"], hk["cfg"]["d"], hk["layers"])
    if spec == "step1":
        pn.load_state_dict(ck["state"])
        head.load_state_dict(hk["state"])
    else:
        jk = torch.load(spec, map_location="cpu", weights_only=False)
        pn.load_state_dict(jk["picknet"])
        head.load_state_dict(jk["head"])
    return pn.to(DEV).train(), head.to(DEV).train()


def main(*kv):
    cfg = dict(CFG)
    cfg.update({k: value(v) for k, v in (a.split("=", 1) for a in kv)})
    t0 = time.time()
    cards = np.load(SHARDS / "cards.npz")
    pad = len(cards["name_id"])
    shards = [Shard(p, pad) for p in sorted(SHARDS.iterdir())
              if p.is_dir() and (p / "info.json").exists() and p.name.endswith(".PremierDraft")]
    rng = np.random.default_rng(0)  # the joint run's draws, so these are its training drafts
    train = []
    for i, s in enumerate(shards):
        rows = np.flatnonzero(s.split == 0)
        if len(rows) > JCFG["max_per_set"]:
            rows = np.sort(rng.choice(rows, JCFG["max_per_set"], replace=False))
        train.append(np.stack([np.full(len(rows), i), rows], 1))
    train = np.concatenate(train).astype(np.int64)

    res = {"cfg": cfg, "ckpts": {}}
    for ck in cfg["ckpts"].split(","):
        tag, spec = ck.split("=", 1)
        torch.manual_seed(cfg["seed"])
        pn, head = load(spec, pad, cards)
        names = [n for n, p in pn.named_parameters() if p.requires_grad]
        params = [p for n, p in pn.named_parameters() if p.requires_grad]
        groups = sorted({group(n) for n in names})
        gid = [groups.index(group(n)) for n in names]
        G = len(groups)
        cos = {pr: [] for pr in PAIRS}          # per batch: [G+1] (groups, then all)
        nrm = {k: [] for k in ("pick", "deck", "wheel")}
        acc = {k: [torch.zeros_like(p) for p in params] for k in ("pick", "deck", "wheel")}
        gen = train_batches(shards, train, cfg["batch"], cfg["seed"], 0)
        for step in range(cfg["batches"]):
            b = next(gen)
            x7, (x, h, tab, logit), nll, lp, wl, mask, label = forward(pn, head, b)
            dk = deck_loss(pn, h, tab, x7[2], b[8].to(DEV), b[9].to(DEV))
            wb = F.binary_cross_entropy_with_logits(wl[mask], label[mask].float())
            g = {}
            for k, loss, last in (("pick", nll, False), ("deck", dk, False), ("wheel", wb, True)):
                gr = torch.autograd.grad(loss, params, retain_graph=not last, allow_unused=True)
                g[k] = [torch.zeros_like(p) if q is None else q.detach() for p, q in zip(params, gr)]
                for a, q in zip(acc[k], g[k]):
                    a += q
            # per-group dot products and squared norms
            sq = {k: torch.zeros(G + 1, device=DEV) for k in g}
            dot = {pr: torch.zeros(G + 1, device=DEV) for pr in PAIRS}
            for j, gi in enumerate(gid):
                for k in g:
                    v = (g[k][j].float() ** 2).sum()
                    sq[k][gi] += v
                    sq[k][G] += v
                for a, c in PAIRS:
                    v = (g[a][j].float() * g[c][j].float()).sum()
                    dot[(a, c)][gi] += v
                    dot[(a, c)][G] += v
            for a, c in PAIRS:
                cos[(a, c)].append((dot[(a, c)] / (sq[a].sqrt() * sq[c].sqrt()).clamp(min=1e-20)).cpu().numpy())
            for k in g:
                nrm[k].append(sq[k].sqrt().cpu().numpy())
            if (step + 1) % 50 == 0:
                print(f"  {tag} batch {step + 1}/{cfg['batches']}: cos(pick,wheel) all "
                      f"{np.mean([c[G] for c in cos[('pick', 'wheel')]]):+.4f} [{(time.time() - t0) / 60:.1f} min]",
                      flush=True)
        # cosine of the batch-averaged gradients
        msq = {k: np.zeros(G + 1) for k in acc}
        mdot = {pr: np.zeros(G + 1) for pr in PAIRS}
        for j, gi in enumerate(gid):
            for k in acc:
                v = float((acc[k][j].cpu().double() ** 2).sum())
                msq[k][gi] += v
                msq[k][G] += v
            for a, c in PAIRS:
                v = float((acc[a][j].cpu().double() * acc[c][j].cpu().double()).sum())
                mdot[(a, c)][gi] += v
                mdot[(a, c)][G] += v
        out = {}
        for i, gname in enumerate(groups + ["ALL"]):
            r = {"params": int(sum(params[j].numel() for j in range(len(params)) if i == G or gid[j] == i))}
            for a, c in PAIRS:
                cb = np.array([x[i] for x in cos[(a, c)]])
                r[f"{a}-{c}"] = {"cos_batch": float(cb.mean()), "cos_batch_sd": float(cb.std()),
                                 "frac_neg": float((cb < 0).mean()),
                                 "cos_mean": float(mdot[(a, c)][i] / max(np.sqrt(msq[a][i] * msq[c][i]), 1e-30))}
            for k in nrm:
                r[f"norm_{k}"] = float(np.mean([x[i] for x in nrm[k]]))
                r[f"mnorm_{k}"] = float(np.sqrt(msq[k][i]) / cfg["batches"])
            out[gname] = r
        res["ckpts"][tag] = out
        print(f"\n{tag} ({spec}): {cfg['batches']} batches of {cfg['batch']}")
        print(f"{'group':20}{'params':>10} | {'pick-wheel':>21} | {'deck-wheel':>21} | {'pick-deck':>21} | "
              f"{'|g| pick':>9}{'deck':>9}{'wheel':>9}{'|mean g| pick':>14}{'deck':>9}{'wheel':>9}")
        print(f"{'':31}| {'batch (sd)   mean':>21} | {'batch (sd)   mean':>21} | {'batch (sd)   mean':>21} |")
        for gname, r in out.items():
            cells = " | ".join(f"{r[f'{a}-{c}']['cos_batch']:+.3f}({r[f'{a}-{c}']['cos_batch_sd']:.2f}) "
                               f"{r[f'{a}-{c}']['cos_mean']:+.3f}" for a, c in PAIRS)
            print(f"{gname:20}{r['params']:>10,} | {cells} | {r['norm_pick']:9.4f}{r['norm_deck']:9.4f}"
                  f"{r['norm_wheel']:9.4f}{r['mnorm_pick']:14.4f}{r['mnorm_deck']:9.4f}{r['mnorm_wheel']:9.4f}", flush=True)
        (OUT / f"{cfg['name']}.json").write_text(json.dumps(res, indent=1))
        del pn, head, acc
        if DEV == "mps":
            torch.mps.empty_cache()
    print(f"done [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main(*sys.argv[1:])
