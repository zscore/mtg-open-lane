"""WheelNet: a transformer that reads the raw draft history and scores every passed card.

History encoder: one token per pick seen so far (bag-of-cards of the pack shown + the card taken +
the cards the other seven drafters took out of a pack that came back + pack/pick position), causal
self-attention, so the state at pick t only sees picks <= t. This contains every signal the
hand-built features try to summarise (colours flowing late, what wheeled in earlier rounds, the
user's pool).

Pack decoder: the cards being passed at a wheel-eligible pick form a set; each card token gets the
history state added, then alternates set self-attention (cards shield each other) with
cross-attention to the history up to that pick, and gets a logit per card.

Optional per-draft context (drafter rank, skill buckets, days into the format) goes through a small
MLP and is added to every history token.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import COLORS, N_PACKS, N_SEATS, RANK_ORDER, RARITIES, DraftTensors

CARD_TYPES = ("Creature", "Artifact", "Enchantment", "Instant", "Sorcery", "Land", "Planeswalker",
              "Battle", "Kindred")
SUPERTYPES = ("Legendary", "Basic", "Snow")


def sparse(counts: np.ndarray, k: int):
    """[..., C] card counts -> ([..., k] distinct card indices padded with C, [..., k] copies).

    Keeps card-index order. Stores a pack in k entries instead of C, ~10x less RAM.
    """
    C = counts.shape[-1]
    flat = counts.reshape(-1, C)
    r, c = np.nonzero(flat)
    j = np.arange(len(r)) - np.searchsorted(r, r)  # rank within row (nonzero is row-major)
    assert j.max(initial=0) < k, f"more than {k} distinct cards in a row"
    idx = np.full((len(flat), k), C, np.int16)
    cnt = np.zeros((len(flat), k), np.int8)
    idx[r, j], cnt[r, j] = c, flat[r, c]
    shp = counts.shape[:-1] + (k,)
    return idx.reshape(shp), cnt.reshape(shp)


def pack_tokens(passed: np.ndarray, back: np.ndarray, C: int):
    """Distinct passed cards per (d,k,p), padded with index C. Returns idx, copies, label, mask."""
    idx, copies = sparse(passed, passed.shape[2] + N_SEATS - 1)  # at most n_picks - 1 distinct
    mask = copies > 0
    b = np.concatenate([back, np.zeros(back.shape[:-1] + (1,), back.dtype)], -1)
    label = np.take_along_axis(b, idx.astype(np.int64), -1) > 0
    return idx.astype(np.int64), copies.astype(np.int64), label, mask


def taken_by_others(passed: np.ndarray, back: np.ndarray):
    """Sparse copies the other seven drafters took from the pack that returns at each pick.

    Empty for picks 0..7, whose pack has not been round the table yet.
    """
    D, _, W, C = passed.shape
    k = W + N_SEATS - 1  # 7 normally; malformed records can have more
    idx, cnt = sparse(passed - back, k)
    out_i = np.full((D, N_PACKS, W + N_SEATS, k), C, np.int16)
    out_c = np.zeros(out_i.shape, np.int8)
    out_i[:, :, N_SEATS:], out_c[:, :, N_SEATS:] = idx, cnt
    return out_i, out_c


def type_features(types: list[str]) -> np.ndarray:
    """[C, F] multi-hot of card types and supertypes over all faces; subtypes are ignored.

    "Legendary Artifact Creature - Thopter // Land" -> Legendary, Artifact, Creature, Land.
    """
    main = [{w.replace("Tribal", "Kindred") for f in t.split(" // ") for w in f.split(" - ")[0].split()}
            for t in types]
    cols = [*CARD_TYPES, *SUPERTYPES]
    return np.array([[c in m for c in cols] for m in main], np.float32).reshape(len(types), len(cols))


def card_features(meta, types: bool = True) -> np.ndarray:
    """[C, F] colours, rarity one-hot, mana value, bonus-sheet flag, and either the card types and
    supertypes or just a creature flag."""
    x = [meta.select(list(COLORS)).to_numpy(),
         np.stack([(meta["rarity"] == r).to_numpy() for r in RARITIES], 1),
         meta["mana_value"].clip(0, 8).to_numpy()[:, None] / 8,
         meta.select("is_bonus").to_numpy()]
    if types and "types" in meta.columns:
        x.append(type_features(meta["types"].to_list()))
    else:
        x.append(meta.select("is_creature").to_numpy())
    return np.concatenate([np.asarray(a, np.float32) for a in x], 1)


def draft_context(t: DraftTensors, rank: bool = True, skill: bool = True,
                  time: bool = True) -> np.ndarray:
    """[D, F] per-draft facts the history tokens all get: the drafter's rank, their 17Lands skill
    buckets, and how far into the format the draft happened.

    Time is continuous (days since the set's first draft) rather than a week id: splits are
    chronological, so test drafts fall in weeks a learned per-week embedding never trained on.
    """
    D, x = len(t.draft_id), []
    if rank:  # bronze..mythic, plus unknown (-1)
        x.append(np.eye(len(RANK_ORDER) + 1, dtype=np.float32)[t.rank.astype(np.int64) + 1])
    if skill:  # buckets are -1 when missing
        wr, ng = t.user_wr, t.user_ngames
        x.append(np.stack([np.where(wr >= 0, wr - 0.54, 0) / 0.09, wr < 0,
                           np.log10(ng.clip(min=1)) / 3, ng < 0], 1).astype(np.float32))
    if time:
        ts = np.array(t.draft_time, dtype="datetime64[s]")
        days = (ts - ts.min()).astype(np.float64) / 86400
        x.append(np.stack([days / 30, np.log1p(days) / np.log1p(30)], 1).astype(np.float32))
    return np.concatenate(x, 1) if x else np.zeros((D, 0), np.float32)


class DraftDataset(torch.utils.data.Dataset):
    def __init__(self, t: DraftTensors, passed, back, drafts: np.ndarray,
                 ctx: np.ndarray | None = None):
        """ctx: [all drafts, F] from draft_context, or None for no per-draft context."""
        C = len(t.names)
        n, T = len(drafts), N_PACKS * t.n_picks
        # splits are contiguous ranges: index with a slice so the big arrays are views, not copies
        sel = slice(drafts[0], drafts[-1] + 1) if (np.diff(drafts) == 1).all() else drafts
        self.pack, self.pack_n = (a.reshape(n, T, -1) for a in sparse(t.pack[sel], t.n_picks))
        self.oth, self.oth_n = (a.reshape(n, T, -1)
                                for a in taken_by_others(passed[sel], back[sel]))
        self.pick = t.pick[sel].reshape(len(drafts), -1).astype(np.int64)
        self.idx, self.copies, self.label, self.mask = pack_tokens(passed[sel], back[sel], C)
        self.ctx = np.zeros((n, 0), np.float32) if ctx is None else ctx[sel]
        self.drafts = drafts

    def __len__(self):
        return len(self.drafts)

    def __getitem__(self, i):
        return (self.pack[i], self.pack_n[i], self.oth[i], self.oth_n[i], self.pick[i],
                self.idx[i], self.copies[i], self.label[i], self.mask[i], self.ctx[i])


def collate(batch):
    cols = list(zip(*batch))
    return [torch.from_numpy(np.stack(c)) for c in cols]


class WheelNet(nn.Module):
    def __init__(self, n_cards: int, n_picks: int = 14, d: int = 192, heads: int = 6, hist_layers: int = 4,
                 pack_layers: int = 3, dropout: float = 0.2, cross: bool = True,
                 others: bool = True, tied_bag: bool = True, card_feats: np.ndarray | None = None,
                 n_ctx: int = 0, ids: bool = True):
        """ids=False: a card is only an MLP of `card_feats` (no learned per-card vector), so one model
        can be trained on several sets and score cards it never saw."""
        super().__init__()
        self.C, self.cross, self.use_others, self.tied_bag = n_cards, cross, others, tied_bag
        self.ids = ids
        W = self.W = n_picks - N_SEATS  # wheel-eligible picks per pack
        self.max_pass = n_picks - 1
        T = N_PACKS * n_picks
        if ids:
            self.card_ = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        if card_feats is not None:  # known attributes seed each card's learned embedding
            f = torch.as_tensor(card_feats, dtype=torch.float32)
            self.register_buffer("feats", torch.cat([f, f.new_zeros(1, f.shape[1])]))
            self.feat_proj = (nn.Linear(f.shape[1], d, bias=False) if ids else
                              nn.Sequential(nn.Linear(f.shape[1], 2 * d), nn.GELU(), nn.Linear(2 * d, d)))
        else:
            assert ids, "a features-only card table needs card_feats"
            self.feats = None
        if tied_bag:  # bags are count-weighted means of the shared card embeddings
            self.bag = nn.Linear(d, d)
            self.others = nn.Linear(d, d)
        else:
            self.bag_emb = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
            self.others_emb = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        self.taken = nn.Linear(d, d)
        if n_ctx:  # per-draft context (rank, skill, time), added to every history token
            self.draft_ctx = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d))
        self.n_ctx = n_ctx
        self.pos = nn.Embedding(T, d)
        self.copies = nn.Embedding(4, d)
        self.pick_pos = nn.Embedding(N_PACKS * W, d)
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                                   norm_first=True)
        self.hist = nn.TransformerEncoder(layer(), hist_layers, enable_nested_tensor=False)
        if cross:
            dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            self.packenc = nn.TransformerDecoder(dec, pack_layers)
        else:
            self.packenc = nn.TransformerEncoder(layer(), pack_layers, enable_nested_tensor=False)
        self.hist_norm = nn.LayerNorm(d)
        self.ctx = nn.Linear(d, d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("causal", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))
        # history position of every wheel-eligible pick
        wp = torch.tensor([k * n_picks + p for k in range(N_PACKS) for p in range(W)])
        self.register_buffer("wheel_pos", wp)
        # cross-attention may only see history up to and including the pick being scored
        future = torch.arange(T)[None] > wp[:, None]                                   # [18, 42]
        self.register_buffer("future_tok", future.repeat_interleave(self.max_pass, 0))  # [234, 42]
        pos = torch.arange(len(wp)).repeat_interleave(self.max_pass)
        self.register_buffer("block", pos[:, None] != pos[None])                   # [234, 234]

    def card_table(self):
        if not self.ids:  # padding row stays zero
            w = self.feat_proj(self.feats)
            return w * (torch.arange(len(w), device=w.device) < len(w) - 1)[:, None]
        w = self.card_.weight
        return w if self.feats is None else w + self.feat_proj(self.feats)

    def card(self, i):
        return nn.functional.embedding(i, self.card_table())

    def _bag(self, idx, cnt, which: str):
        """Count-weighted mean embedding of a sparse bag of cards."""
        w = cnt.float()
        w = w / w.sum(-1, keepdim=True).clamp(min=1)
        table = self.card_table() if self.tied_bag else getattr(self, which + "_emb").weight
        x = (w[..., None] * nn.functional.embedding(idx.long(), table)).sum(-2)
        return getattr(self, which)(x) if self.tied_bag else x

    def forward(self, pack, pack_n, oth, oth_n, pick, idx, copies, mask, ctx=None):
        B = pack.shape[0]
        h = self._bag(pack, pack_n, "bag") + self.pos.weight[None]
        if self.n_ctx:
            h = h + self.draft_ctx(ctx)[:, None]
        h = h + (self.taken(self.card(pick)) if self.tied_bag else self.card(pick))
        if self.use_others:
            h = h + self._bag(oth, oth_n, "others")
        h = self.hist_norm(self.hist(h, mask=self.causal, is_causal=True))  # [B, 42, d]
        state = h[:, self.wheel_pos]                                         # [B, 18, d]

        n_pos, M = N_PACKS * self.W, self.max_pass
        idx = idx.reshape(B * n_pos, M)
        m = mask.reshape(B * n_pos, M)
        x = (self.card(idx) + self.copies(copies.reshape(B * n_pos, M).clamp(max=3))
             + (self.ctx(state) + self.pick_pos.weight[None]).reshape(B * n_pos, 1, -1))
        # a pack with no passed cards (P1P1 when the set's files lack it) would attend to nothing
        # and give NaNs; let it see its first padding slot. Those outputs are never scored.
        pad = ~m
        pad[:, 0] &= m.any(-1)
        if self.cross:
            # all 18 packs of a draft in one sequence so the history is projected once per draft;
            # block-diagonal self-attention keeps the packs separate
            x = self.packenc(x.reshape(B, n_pos * M, -1), h, tgt_mask=self.block,
                             memory_mask=self.future_tok, tgt_key_padding_mask=pad.reshape(B, -1))
        else:
            x = self.packenc(x, src_key_padding_mask=pad)
        return self.head(x).squeeze(-1).reshape(B, N_PACKS, self.W, M)
