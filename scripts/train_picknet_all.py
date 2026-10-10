"""Train one PickNet on every set at once (puder-style scale), from scripts/build_pick_shards.py's shards.

    python scripts/train_picknet_all.py name=all_d256 steps=60000 [key=value ...]
    python scripts/train_picknet_all.py bench=1 d=256 ...        # time 30 steps and exit

Cards are printings with shared name ids (a reprint is one learned vector) plus card features, so
every set trains the same card table; a learned set embedding and per-draft context (rank, skill, days
since launch, calendar time, Traditional flag, pack size) go into every history token. Trains on the
train split of every set and event (Premier + Traditional), early-stops on Premier val drafts, and
scores the Premier test split of every set: the same chronological test drafts as train_picknet.py.

Switches (defaults in CFG): d heads hist_layers pack_layers dropout | batch lr wd steps warmup amp |
pack_drop (share of drafts whose history hides pack contents) deck_w (weight of the deck-inclusion
loss, 0 = no head) sets=A,B events=PremierDraft,TradDraft max_train (subsample train drafts) |
val_every val_drafts ckpt_every resume bench name test_sets=A,B (default: every set).
Writes results/picknet_all/{name}.json, models/{name}.pt (best val state), test_{name}_{SET}.npz.
"""
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import ROOT  # noqa: E402
from openlane.picknet import PickNet, pick_loss  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SHARDS = ROOT / "data" / "features" / "picks_all"
OUT = ROOT / "results" / "picknet_all"
N_PICKS = 15  # layout: 3 packs x 15 slots
CFG = dict(d=256, heads=8, hist_layers=4, pack_layers=3, dropout=0.05,
           batch=128, lr=7e-4, wd=0.01, steps=40000, warmup=0.03, amp=0,
           pack_drop=0.5, deck_w=0.3, sets="", events="PremierDraft,TradDraft", max_train=0,
           val_every=2000, val_drafts=4000, ckpt_every=1000, resume=1, bench=0, name="all", seed=0, test_sets="")
FIELDS = ("idx", "cnt", "pick", "oidx", "ocnt", "ctx", "sid", "target", "valid", "rate")


def value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


class Shard:
    def __init__(self, path: Path):
        self.info = json.loads((path / "info.json").read_text())
        self.name = f"{self.info['set']}.{self.info['event']}"
        self.a = {k: np.load(path / f"{k}.npy", mmap_mode="r") for k in FIELDS}
        self.split = np.load(path / "split.npy")
        self.meta = {k: np.load(path / f"{k}.npy") for k in ("rank", "wr", "ng")}

    def get(self, rows: np.ndarray) -> dict:
        return {k: np.asarray(v[rows]) for k, v in self.a.items()}


def to_torch(b: dict) -> list:
    """Batch arrays -> tensors in PickNet.forward order (+ target, valid, rate)."""
    t = lambda x, dt: torch.from_numpy(np.ascontiguousarray(x)).to(dt)  # noqa: E731
    return [t(b["idx"], torch.int32), t(b["cnt"], torch.int8), t(b["pick"], torch.int64),
            t(b["oidx"], torch.int32), t(b["ocnt"], torch.int8), t(b["ctx"], torch.float32),
            t(b["sid"], torch.int64), t(b["target"], torch.int64), t(b["valid"], torch.bool),
            t(b["rate"], torch.uint8)]


def gather(shards, pairs: np.ndarray) -> dict:
    """pairs [n, 2] (shard, row) -> one batch, read shard by shard with sorted rows."""
    parts = []
    for s in np.unique(pairs[:, 0]):
        rows = np.sort(pairs[pairs[:, 0] == s, 1])
        parts.append(shards[s].get(rows))
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


def train_batches(shards, pairs, batch, seed, start_step):
    """Endless shuffled batches over `pairs`, reproducible from (seed, step) so a resumed run continues
    the same order. A background thread reads ahead from the memmaps."""
    per_epoch = len(pairs) // batch
    q = queue.Queue(maxsize=8)

    def work():
        step = start_step
        while True:
            ep, i = divmod(step, per_epoch)
            perm = np.random.default_rng(seed + ep).permutation(len(pairs))
            for j in range(i, per_epoch):
                q.put(to_torch(gather(shards, pairs[perm[j * batch:(j + 1) * batch]])))
            step = (ep + 1) * per_epoch

    threading.Thread(target=work, daemon=True).start()
    while True:
        yield q.get()


def losses(model, b, deck_w, amp):
    x = [v.to(DEV, non_blocking=True) for v in b[:7]]
    target, valid, rate = (v.to(DEV) for v in b[7:])
    with torch.autocast(DEV, dtype=torch.bfloat16, enabled=bool(amp)):
        logit, h, tab = model(*x, return_state=True)
        dl = model.deck_logits(h, tab, x[2]) if deck_w else None
    nll, lp = pick_loss(logit.float(), x[1], target, valid)
    out = {"nll": nll, "lp": lp, "target": target, "valid": valid, "cnt": x[1]}
    if deck_w:
        T = rate.shape[1]
        known = rate < 255
        y = rate.float() / 254
        tt = torch.arange(T, device=DEV)
        m = (tt[None, :, None] > tt[None, None, :]) & valid[:, :, None] & known[:, None, :]
        w = m * ((tt[None, :, None] + 1) / T)
        bce = F.binary_cross_entropy_with_logits(dl.float(), y[:, None, :].expand_as(dl), reduction="none")
        out["deck"] = (bce * w).sum() / w.sum().clamp(min=1)
    return out


@torch.no_grad()
def evaluate(model, shards, pairs, batch=256, keep=False):
    """NLL / top-1 over valid picks (and non-forced picks); with keep, per-draft arrays in pair order."""
    model.eval()
    tot = {"nll": 0.0, "hit": 0, "n": 0, "hit_nf": 0, "n_nf": 0}
    keep_hit, keep_nll, keep_valid, keep_forced = [], [], [], []
    for i in range(0, len(pairs), batch):
        b = to_torch(gather(shards, pairs[i:i + batch]))
        x = [v.to(DEV) for v in b[:7]]
        target, valid = b[7].to(DEV), b[8].to(DEV)
        lp = pick_loss(model(*x).float(), x[1], target, valid)[1]
        lt = lp.gather(-1, target[..., None]).squeeze(-1)
        hit = lp.argmax(-1) == target
        forced = (x[1] > 0).sum(-1) == 1
        nf = valid & ~forced
        tot["nll"] += float(-lt[valid].sum())
        tot["hit"] += int(hit[valid].sum())
        tot["n"] += int(valid.sum())
        tot["hit_nf"] += int(hit[nf].sum())
        tot["n_nf"] += int(nf.sum())
        if keep:
            keep_hit.append(hit.cpu().numpy())
            keep_nll.append((-lt).float().cpu().numpy())
            keep_valid.append(valid.cpu().numpy())
            keep_forced.append(forced.cpu().numpy())
    model.train()
    res = {"nll": tot["nll"] / tot["n"], "top1": tot["hit"] / tot["n"], "top1_nonforced": tot["hit_nf"] / tot["n_nf"],
           "picks": tot["n"]}
    if keep:
        res["arrays"] = {k: np.concatenate(v) for k, v in
                         dict(top1=keep_hit, nll=keep_nll, valid=keep_valid, forced=keep_forced).items()}
    return res


def main(*kv):
    cfg = dict(CFG)
    cfg.update({k: value(v) for k, v in (a.split("=", 1) for a in kv)})
    torch.manual_seed(cfg["seed"])
    name = cfg["name"]
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cards = np.load(SHARDS / "cards.npz")
    all_sets = list(cards["sets"])
    want_sets = set(cfg["sets"].split(",")) if cfg["sets"] else set(all_sets)
    events = cfg["events"].split(",")
    shards = [Shard(p) for p in sorted(SHARDS.iterdir())
              if p.is_dir() and (p / "info.json").exists()
              and p.name.split(".")[0] in want_sets and p.name.split(".")[1] in events]
    rng = np.random.default_rng(cfg["seed"])

    def pairs_for(split, premier_only=False):
        out = [np.stack([np.full(int((s.split == split).sum()), i), np.flatnonzero(s.split == split)], 1)
               for i, s in enumerate(shards) if not premier_only or s.info["event"] == "PremierDraft"]
        return np.concatenate(out).astype(np.int64)

    train = pairs_for(0)
    if cfg["max_train"] and len(train) > cfg["max_train"]:
        train = train[np.sort(rng.choice(len(train), cfg["max_train"], replace=False))]
    val = pairs_for(1, premier_only=True)
    val = val[np.sort(rng.choice(len(val), min(cfg["val_drafts"], len(val)), replace=False))]
    print(f"{name}: {len(shards)} shards, train {len(train):,} drafts, val {len(val):,} [{time.time() - t0:.0f}s]",
          flush=True)

    feats = cards["feats"]
    mu, sd = feats.mean(0), feats.std(0) + 1e-3
    n_ctx = shards[0].a["ctx"].shape[1]
    model = PickNet(len(cards["name_id"]), N_PICKS, d=cfg["d"], heads=cfg["heads"], hist_layers=cfg["hist_layers"],
                    pack_layers=cfg["pack_layers"], dropout=cfg["dropout"], feats=(feats - mu) / sd,
                    id_map=cards["name_id"], n_ctx=n_ctx, n_sets=len(all_sets), pack_drop=cfg["pack_drop"],
                    deck_head=cfg["deck_w"] > 0).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par:,} cfg {cfg} device {DEV}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"],
                                                pct_start=cfg["warmup"])
    ck_path, best_path = OUT / "models" / f"{name}.ckpt", OUT / "models" / f"{name}.pt"
    step, best, hist = 0, float("inf"), []
    if cfg["resume"] and ck_path.exists() and not cfg["bench"]:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best, hist = ck["step"], ck["best"], ck["hist"]
        print(f"resumed at step {step} (best val nll {best:.4f})", flush=True)

    def save_best():
        torch.save({"cfg": cfg, "state": model.state_dict(), "feat_mu": mu, "feat_sd": sd, "n_ctx": n_ctx,
                    "sets": all_sets, "params": n_par}, best_path)

    batches = train_batches(shards, train, cfg["batch"], cfg["seed"], step)
    model.train()
    run = {"nll": 0.0, "deck": 0.0, "n": 0}
    t1, step0 = time.time(), step
    while step < cfg["steps"]:
        b = next(batches)
        out = losses(model, b, cfg["deck_w"], cfg["amp"])
        loss = out["nll"] + (cfg["deck_w"] * out["deck"] if cfg["deck_w"] else 0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        run["nll"] += out["nll"].item()
        run["deck"] += out["deck"].item() if cfg["deck_w"] else 0
        run["n"] += 1
        if cfg["bench"] and step == 5:
            t1 = time.time()  # skip warm-up steps
        if cfg["bench"] and step == 35:
            sps = 30 / (time.time() - t1)
            print(f"BENCH {sps:.2f} steps/s = {sps * cfg['batch']:.0f} drafts/s; "
                  f"{cfg['steps']} steps = {cfg['steps'] / sps / 3600:.2f} h; one epoch of {len(train):,} drafts "
                  f"= {len(train) / cfg['batch'] / sps / 3600:.2f} h", flush=True)
            return
        if step % 200 == 0:
            el = time.time() - t1
            print(f"  step {step}/{cfg['steps']} nll {run['nll'] / run['n']:.4f} deck {run['deck'] / run['n']:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {(step - step0) * cfg['batch'] / max(el, 1):.0f} drafts/s "
                  f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
            run = {"nll": 0.0, "deck": 0.0, "n": 0}
        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            v = evaluate(model, shards, val)
            hist.append({"step": step, "val_nll": v["nll"], "val_top1": v["top1"], "minutes": (time.time() - t0) / 60})
            tag = ""
            if v["nll"] < best:
                best, tag = v["nll"], " *"
                save_best()
            print(f"  VAL step {step}: nll {v['nll']:.4f} top1 {v['top1']:.4f}{tag}", flush=True)
        if step % cfg["ckpt_every"] == 0 or step == cfg["steps"]:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "hist": hist}, ck_path)

    # test: Premier test drafts of every set, best val state
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["state"])
    res = {"cfg": cfg, "params": n_par, "train_drafts": len(train), "best_val_nll": best, "history": hist,
           "test": {}}
    for i, s in enumerate(shards):
        if s.info["event"] != "PremierDraft" or (cfg["test_sets"] and s.info["set"] not in cfg["test_sets"].split(",")):
            continue
        rows = np.flatnonzero(s.split == 2)
        r = evaluate(model, shards, np.stack([np.full(len(rows), i), rows], 1), keep=True)
        arr = r.pop("arrays")
        np.savez_compressed(OUT / f"test_{name}_{s.info['set']}.npz", **arr,
                            **{k: v[rows] for k, v in s.meta.items()})
        res["test"][s.info["set"]] = r
        print(f"TEST {s.info['set']}: nll {r['nll']:.4f} top1 {r['top1']:.4f} "
              f"non-forced {r['top1_nonforced']:.4f}", flush=True)
    res["seconds"] = time.time() - t0
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"done [{(time.time() - t0) / 3600:.2f} h]")


if __name__ == "__main__":
    main(*sys.argv[1:])
