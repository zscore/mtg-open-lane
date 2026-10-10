"""Pack every set's drafts into one fixed layout on disk, for the all-sets pick model.

    python scripts/build_pick_shards.py [SET ...]      (default: every set with a PremierDraft file)

Each (set, event) becomes data/features/picks_all/{SET}.{EVENT}/*.npy memmaps with the same layout
for every set: 3 packs x 15 pick slots (a 14-pick set leaves slot 14 of each pack empty and invalid),
up to 15 distinct cards per pack, cards as global *printing* ids (one row per set x card, shared by a
set's Premier and Traditional drafts). Then data/features/picks_all/cards.npz holds the printing table:
card features (colour/rarity/types, Scryfall rules features, rules-text embedding), a name id shared
across sets (a reprint or basic land is one learned embedding), and the set of each printing.

Per draft: idx/cnt [45, 15] pack tokens, pick [45], target [45] (slot of the pick in idx), valid [45],
oidx/ocnt [45, 8] what the other seven took from a pack that came back, rate [45] maindeck rate x 254
(255 = unknown), ctx [F] (rank, skill, days since launch, calendar time, Traditional flag, pack size),
sid (set index), split (0 train / 1 val / 2 test: train_picknet's chronological 70/15/15 for Premier,
the same proportions by draft time for Traditional), and rank / wr / ng for slicing results.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlane.data import N_PACKS, N_SEATS, PARQUET, ROOT, card_metadata, load_tensors  # noqa: E402
from openlane.nn import draft_context, sparse  # noqa: E402
from train_picknet import card_inputs, splits_for  # noqa: E402

OUT = ROOT / "data" / "features" / "picks_all"
SLOTS, P, K_OTH = 15, 15, 8          # pick slots per pack, pack tokens, others-taken tokens
T = N_PACKS * SLOTS
EPOCH = np.datetime64("2023-01-01T00:00:00")
CHUNK = 20000


def all_sets():
    return sorted(p.name.split(".")[0] for p in PARQUET.glob("*.PremierDraft.parquet"))


def others_taken(pack, pick):
    """[n, K, N, C] copies the other seven took from the pack that comes back (as picknet.others_taken)."""
    N = pack.shape[2]
    out = np.zeros_like(pack)
    prev = pack[:, :, :N - N_SEATS].copy()
    pk = pick[:, :, :N - N_SEATS].astype(np.int64)
    np.put_along_axis(prev, pk[..., None], np.take_along_axis(prev, pk[..., None], -1) - 1, -1)
    out[:, :, N_SEATS:] = np.clip(prev - pack[:, :, N_SEATS:], 0, None)
    return out


def context(t, trad: bool) -> np.ndarray:
    """draft_context (rank one-hot, skill, days since launch) + years since 2023, Traditional flag,
    pack size one-hot (13 / 14 / 15 picks)."""
    ts = np.array(t.draft_time, dtype="datetime64[s]")
    years = ((ts - EPOCH).astype(np.float64) / 86400 / 365.25).astype(np.float32)
    D = len(ts)
    size = np.zeros((D, 3), np.float32)
    size[:, t.n_picks - 13] = 1
    return np.hstack([draft_context(t), years[:, None] - 2, np.full((D, 1), float(trad), np.float32), size])


def write(dirpath: Path, arrays: dict):
    dirpath.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(dirpath / f"{k}.npy", v)


def build_shard(set_code, event, sid, glob_of_name, pad, trad, out):
    t0 = time.time()
    t = load_tensors(set_code, event)
    D, n = len(t.draft_id), t.n_picks
    missing = [nm for nm in t.names if nm not in glob_of_name]
    if missing:  # a card column only the Traditional file has: treat it as padding
        print(f"    {set_code}.{event}: {len(missing)} cards not in the Premier file: {missing[:5]}")
    local2glob = np.array([glob_of_name.get(nm, pad) for nm in t.names] + [pad], np.int64)
    d = out / f"{set_code}.{event}"
    d.mkdir(parents=True, exist_ok=True)
    mm = {k: np.lib.format.open_memmap(d / f"{k}.npy", "w+", dt, shp) for k, dt, shp in [
        ("idx", np.int16, (D, T, P)), ("cnt", np.int8, (D, T, P)), ("pick", np.int16, (D, T)),
        ("target", np.int8, (D, T)), ("valid", np.bool_, (D, T)), ("oidx", np.int16, (D, T, K_OTH)),
        ("ocnt", np.int8, (D, T, K_OTH)), ("rate", np.uint8, (D, T))]}
    slot = (np.arange(N_PACKS)[:, None] * SLOTS + np.arange(n)[None]).ravel()  # real slots in the layout
    for a in range(0, D, CHUNK):
        b = min(a + CHUNK, D)
        pack, pick = t.pack[a:b], t.pick[a:b]
        m = b - a
        i, c = sparse(pack, P)                                  # [m, 3, n, P], pad = C
        oi, oc = sparse(others_taken(pack, pick), K_OTH)
        g = local2glob[i.astype(np.int64)].reshape(m, -1, P)
        gp = local2glob[pick.astype(np.int64)].reshape(m, -1)
        idx = np.full((m, T, P), pad, np.int16)
        cnt = np.zeros((m, T, P), np.int8)
        pk = np.full((m, T), pad, np.int16)
        oidx = np.full((m, T, K_OTH), pad, np.int16)
        ocnt = np.zeros((m, T, K_OTH), np.int8)
        idx[:, slot], cnt[:, slot], pk[:, slot] = g, c.reshape(m, -1, P), gp
        oidx[:, slot] = local2glob[oi.astype(np.int64)].reshape(m, -1, K_OTH)
        ocnt[:, slot] = oc.reshape(m, -1, K_OTH)
        tgt = (idx == pk[..., None]).argmax(-1).astype(np.int8)
        valid = np.zeros((m, T), bool)
        valid[:, slot] = True
        if t.p1p1_missing:
            valid[:, 0] = False
        r = t.maindeck[a:b].reshape(m, -1)
        rate = np.full((m, T), 255, np.uint8)
        rate[:, slot] = np.where(np.isnan(r), 255, np.round(np.clip(r, 0, 1) * 254)).astype(np.uint8)
        for k, v in dict(idx=idx, cnt=cnt, pick=pk, target=tgt, valid=valid, oidx=oidx, ocnt=ocnt,
                         rate=rate).items():
            mm[k][a:b] = v
    for v in mm.values():
        v.flush()
    if event == "PremierDraft":
        sp = splits_for(t, set_code, event)
    else:  # chronological 70/15/15
        x, y = int(0.7 * D), int(0.85 * D)
        sp = {"train": np.arange(x), "val": np.arange(x, y), "test": np.arange(y, D)}
    split = np.zeros(D, np.int8)
    split[sp["val"]], split[sp["test"]] = 1, 2
    write(d, dict(ctx=context(t, trad), sid=np.full(D, sid, np.int16), split=split,
                  rank=t.rank.astype(np.int8), wr=t.user_wr, ng=t.user_ngames))
    (d / "info.json").write_text(json.dumps(dict(set=set_code, event=event, drafts=D, n_picks=n,
                                                 p1p1_missing=bool(t.p1p1_missing))))
    print(f"  {set_code}.{event}: {D} drafts, {n} picks [{time.time() - t0:.0f}s]", flush=True)


def main(*sets):
    sets = list(sets) or all_sets()
    OUT.mkdir(parents=True, exist_ok=True)
    # printing table from each set's Premier file (card columns + features)
    feats, name_of, set_of, glob = [], [], [], {}
    for s, code in enumerate(sets):
        cols = pl.scan_parquet(PARQUET / f"{code}.PremierDraft.parquet").collect_schema().names()
        nms = [c[len("pack_card_"):] for c in cols if c.startswith("pack_card_")]
        t = SimpleNamespace(names=nms, set_code=code, meta=card_metadata(nms, code))
        f = card_inputs(t, True, True, True)
        glob[code] = {nm: len(name_of) + i for i, nm in enumerate(nms)}
        feats.append(f)
        name_of += nms
        set_of += [s] * len(nms)
        print(f"{code}: {len(nms)} cards, feature dim {f.shape[1]}", flush=True)
    pad = len(name_of)
    names = sorted(set(name_of))
    name_id = np.array([names.index(nm) for nm in name_of], np.int32)
    np.savez(OUT / "cards.npz", feats=np.vstack(feats).astype(np.float32), name_id=name_id,
             set_of=np.array(set_of, np.int16), printing=np.array(name_of), names=np.array(names),
             sets=np.array(sets))
    print(f"{pad} printings, {len(names)} distinct names", flush=True)
    for s, code in enumerate(sets):
        for event in ("PremierDraft", "TradDraft"):
            if not (PARQUET / f"{code}.{event}.parquet").exists():
                continue
            if (OUT / f"{code}.{event}" / "info.json").exists():
                print(f"  {code}.{event}: already built")
                continue
            build_shard(code, event, s, glob[code], pad, event == "TradDraft", OUT)
    print("done")


if __name__ == "__main__":
    main(*sys.argv[1:])
