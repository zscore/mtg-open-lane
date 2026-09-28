"""Wheel labels and features.

A pack seen at pick p (p <= 5) returns at pick p + 8 after the other 7 drafters each take one
card. One example = (draft, pack k, pick p, card c) for every card c the user passes; the label is
whether at least one copy of c is still in the pack when it comes back.

Features come in groups so we can ablate them:
  card    - static identity stats fit on train drafts only (ATA, ALSA, empirical wheel rate, ...)
  pack    - what else is in the pack being passed (how many better cards shield c)
  signals - evidence about the other drafters: how late colors flow to the user this round,
            what wheeled in earlier rounds, and the user's own pool
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .data import COLORS, N_PACKS, N_PICKS, N_SEATS, RARITIES, DraftTensors

WHEEL_PICKS = N_PICKS - N_SEATS  # picks 0..5 can wheel


def wheel_tensors(t: DraftTensors) -> tuple[np.ndarray, np.ndarray]:
    """passed[d,k,p,c] = copies passed on at pick p; back[d,k,p,c] = copies that came back."""
    passed = t.pack[:, :, :WHEEL_PICKS].copy()
    D = passed.shape[0]
    d, k, p = np.meshgrid(np.arange(D), np.arange(N_PACKS), np.arange(WHEEL_PICKS), indexing="ij")
    passed[d, k, p, t.pick[:, :, :WHEEL_PICKS]] -= 1
    back = t.pack[:, :, N_SEATS:]
    bad = (back > passed).any(-1)
    if bad.any():  # a handful of malformed 17Lands records; the returning pack must be a subset
        print(f"warning: {bad.sum()} returning packs are not subsets of the passed pack; clipping")
        back = np.minimum(back, passed)
    return passed, back


@dataclass
class CardStats:
    ata: np.ndarray        # average pick index when taken by the user
    alsa: np.ndarray       # average last pick index at which the card was seen
    take_rate: np.ndarray  # P(user takes c | c in pack)
    wheel_cp: np.ndarray   # [C, 6] empirical wheel rate by pick (smoothed)
    wheel_c: np.ndarray    # overall empirical wheel rate
    seen: np.ndarray       # number of passed instances in train


def fit_card_stats(t: DraftTensors, train: np.ndarray, passed, back, prior_n: float = 20.0) -> CardStats:
    C = len(t.names)
    pick = t.pick[train].reshape(-1)
    pnum = np.broadcast_to(np.arange(N_PICKS), t.pick[train].shape).reshape(-1)
    n_taken = np.bincount(pick, minlength=C)
    ata = np.bincount(pick, weights=pnum, minlength=C) / np.maximum(n_taken, 1)

    present = t.pack[train] > 0  # [D,3,14,C]
    last = np.where(present, np.arange(N_PICKS)[None, None, :, None], -1).max(axis=2)  # [D,3,C]
    seen_mask = last >= 0
    alsa = np.where(seen_mask, last, 0).sum((0, 1)) / np.maximum(seen_mask.sum((0, 1)), 1)
    n_seen_pick = present.sum((0, 1, 2))
    take_rate = n_taken / np.maximum(n_seen_pick, 1)

    pa, bk = passed[train] > 0, back[train] > 0
    n_cp = pa.sum((0, 1)).T.astype(np.float64)       # [C, 6]
    w_cp = bk.sum((0, 1)).T.astype(np.float64)
    global_p = w_cp.sum(0) / n_cp.sum(0)             # [6]
    wheel_cp = (w_cp + prior_n * global_p) / (n_cp + prior_n)
    wheel_c = (w_cp.sum(1) + prior_n * global_p.mean()) / (n_cp.sum(1) + prior_n)
    # unseen cards: fall back to rarity-agnostic late-ish defaults
    ata = np.where(n_taken > 0, ata, N_PICKS - 1)
    alsa = np.where(seen_mask.sum((0, 1)) > 0, alsa, N_PICKS - 1)
    f = lambda a: a.astype(np.float32)
    return CardStats(f(ata), f(alsa), f(take_rate), f(wheel_cp), f(wheel_c), n_cp.sum(1))


def _dot(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """x[..., C] @ m[C, F] in float32 without materialising huge intermediates."""
    s = x.shape
    return (x.reshape(-1, s[-1]).astype(np.float32) @ m).reshape(*s[:-1], m.shape[-1])


FEATURE_GROUPS: dict[str, list[str]] = {}


def build_examples(t: DraftTensors, stats: CardStats, passed, back, drafts: np.ndarray) -> pl.DataFrame:
    """Long-format examples for the given draft indices."""
    col = t.color_matrix                       # [C,5]
    ncol = col.sum(1)
    col_norm = col / np.maximum(ncol, 1)[:, None]
    meta = t.meta
    rarity = meta["rarity"].replace_strict({r: i for i, r in enumerate(RARITIES)}, default=5).to_numpy()

    pack = t.pack[drafts]
    pool = t.pool[drafts]
    pk = t.pick[drafts]
    P = passed[drafts]
    B = back[drafts]
    D = len(drafts)

    # ---------------- pack context ----------------
    # better[c', c] = 1 if c' is usually taken before c (lower empirical wheel rate at that pick)
    per_pick_better = []
    per_pick_better_col = []
    for p in range(WHEEL_PICKS):
        w = stats.wheel_cp[:, p]
        better = (w[:, None] < w[None, :]).astype(np.float32)          # [C', C]
        per_pick_better.append(_dot(P[:, :, p], better))                 # [D,3,C]
        per_pick_better_col.append(
            np.stack([_dot(P[:, :, p], better * col[:, j:j + 1]) for j in range(5)], -1)
        )                                                                 # [D,3,C,5]
    n_better = np.stack(per_pick_better, 2)                               # [D,3,6,C]
    n_better_col = np.stack(per_pick_better_col, 2)                       # [D,3,6,C,5]
    # expected number of 7 picks absorbed by *other* cards ~ sum of their take propensity
    take_prop = 1.0 - stats.wheel_cp                                      # [C,6]
    pack_take_mass = np.stack([_dot(P[:, :, p], take_prop[:, p:p + 1])[..., 0]
                               for p in range(WHEEL_PICKS)], 2)          # [D,3,6]
    pack_size = P.sum(-1)                                                 # [D,3,6]
    pack_col_mass = _dot(P, col)                                          # [D,3,6,5]

    # ---------------- signals ----------------
    # lateness: a card seen at pick q that is usually gone by then is evidence that the upstream
    # drafters are not in its colors.
    late_w = np.maximum(0, np.arange(N_PICKS)[None, :] - stats.alsa[:, None])   # [C,14]
    late = np.stack([_dot(pack[:, :, q], col * late_w[:, q:q + 1]) for q in range(N_PICKS)], 2)
    late[:, :, 0] = 0                                                      # fresh pack: no signal
    late_cum = np.cumsum(late, 2)                                          # [D,3,14,5]
    late_round_total = late.sum(2)                                         # [D,3,5]

    # wheel residual per color per round: what came back beyond card-level expectation
    resid = np.zeros((D, N_PACKS, 5), np.float32)
    for p in range(WHEEL_PICKS):
        r = ((B[:, :, p] > 0) - stats.wheel_cp[None, None, :, p]) * (P[:, :, p] > 0)
        resid += _dot(r, col)
    # prior-round aggregates split by passing direction relative to the current round
    prior_same = np.zeros((D, N_PACKS, 5), np.float32)   # same direction: same neighbours upstream
    prior_opp = np.zeros((D, N_PACKS, 5), np.float32)    # opposite: today's downstream neighbours
    prior_resid = np.zeros((D, N_PACKS, 5), np.float32)
    for k in range(N_PACKS):
        for r in range(k):
            tgt = prior_same if (r % 2 == k % 2) else prior_opp
            tgt[:, k] += late_round_total[:, r]
            prior_resid[:, k] += resid[:, r]

    pool_col = _dot(pool, col)                                             # [D,3,14,5]
    pool_n = pool.sum(-1)
    pool_share = pool_col / np.maximum(pool_n, 1)[..., None]

    # ---------------- assemble rows ----------------
    d, k, p, c = np.nonzero(P > 0)
    cc = col_norm[c]                                                       # [N,5]

    def proj(a):  # project a [N,5] color vector onto the card's colors
        return (a * cc).sum(1)

    my = pk[d, k, p]
    cols = {
        "draft_idx": drafts[d],
        "pack_number": k.astype(np.int8),
        "pick_number": p.astype(np.int8),
        "card": c.astype(np.int16),
        # card
        "ata": stats.ata[c],
        "alsa": stats.alsa[c],
        "take_rate": stats.take_rate[c],
        "wheel_cp": stats.wheel_cp[c, p],
        "wheel_c": stats.wheel_c[c],
        "rarity": rarity[c].astype(np.int8),
        "mana_value": meta["mana_value"].to_numpy()[c],
        "is_creature": meta["is_creature"].to_numpy()[c].astype(np.int8),
        "is_bonus": meta["is_bonus"].to_numpy()[c].astype(np.int8),
        "n_colors": ncol[c],
        **{f"is_{x}": col[c, j] for j, x in enumerate(COLORS)},
        # pack
        "copies": P[d, k, p, c],
        "pack_size": pack_size[d, k, p],
        "n_better": n_better[d, k, p, c],
        "n_better_minus_7": n_better[d, k, p, c] - (N_SEATS - 1),
        "n_better_samecol": proj(n_better_col[d, k, p, c]),
        "pack_take_mass": pack_take_mass[d, k, p] - take_prop[c, p],
        "pack_samecol_mass": proj(pack_col_mass[d, k, p]) - (ncol[c] > 0),
        "my_pick_wheel_cp": stats.wheel_cp[my, p],
        "my_pick_samecol": (col_norm[my] * cc).sum(1),
        # signals
        "late_cur_card": proj(late_cum[d, k, p]),
        "late_cur_total": late_cum[d, k, p].sum(1),
        "late_prior_same_card": proj(prior_same[d, k]),
        "late_prior_opp_card": proj(prior_opp[d, k]),
        "late_prior_total": (prior_same[d, k] + prior_opp[d, k]).sum(1),
        "resid_prior_card": proj(prior_resid[d, k]),
        "resid_prior_total": prior_resid[d, k].sum(1),
        "pool_share_card": proj(pool_share[d, k, p]),
        "pool_n": pool_n[d, k, p],
        "rank": t.rank[drafts][d].astype(np.int8),
        "user_wr": t.user_wr[drafts][d],
        "user_ngames": t.user_ngames[drafts][d],
        **{f"late_cur_{x}": late_cum[d, k, p, j] for j, x in enumerate(COLORS)},
        **{f"late_prior_opp_{x}": prior_opp[d, k, j] for j, x in enumerate(COLORS)},
        **{f"resid_prior_{x}": prior_resid[d, k, j] for j, x in enumerate(COLORS)},
        **{f"pool_share_{x}": pool_share[d, k, p, j] for j, x in enumerate(COLORS)},
        # label
        "wheeled": (B[d, k, p, c] > 0).astype(np.int8),
    }
    return pl.DataFrame(cols)


CARD_FEATURES = ["pick_number", "pack_number", "card", "ata", "alsa", "take_rate", "wheel_cp",
                 "wheel_c", "rarity", "mana_value", "is_creature", "is_bonus", "n_colors",
                 *[f"is_{x}" for x in COLORS]]
PACK_FEATURES = ["copies", "pack_size", "n_better", "n_better_minus_7", "n_better_samecol",
                 "pack_take_mass", "pack_samecol_mass", "my_pick_wheel_cp", "my_pick_samecol"]
SIGNAL_FEATURES = ["late_cur_card", "late_cur_total", "late_prior_same_card", "late_prior_opp_card",
                   "late_prior_total", "resid_prior_card", "resid_prior_total", "pool_share_card",
                   "pool_n", "rank", "user_wr", "user_ngames",
                   *[f"late_cur_{x}" for x in COLORS], *[f"late_prior_opp_{x}" for x in COLORS],
                   *[f"resid_prior_{x}" for x in COLORS], *[f"pool_share_{x}" for x in COLORS]]
FEATURE_GROUPS.update(card=CARD_FEATURES, pack=PACK_FEATURES, signals=SIGNAL_FEATURES)
