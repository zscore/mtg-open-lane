"""WheelNet: a transformer that reads the raw draft history and scores every passed card.

History encoder: one token per pick seen so far (bag-of-cards of the pack shown + the card taken +
the cards the other seven drafters took out of a pack that came back + pack/pick position), causal
self-attention, so the state at pick t only sees picks <= t. This contains every signal the
hand-built features try to summarise (colours flowing late, what wheeled in earlier rounds, the
user's pool).

Pack decoder: the cards being passed at a wheel-eligible pick form a set; each card token gets the
history state added, then alternates set self-attention (cards shield each other) with
cross-attention to the history up to that pick, and gets a logit per card.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import COLORS, N_PACKS, N_PICKS, N_SEATS, RARITIES, DraftTensors

WHEEL_PICKS = N_PICKS - N_SEATS
MAX_PASS = N_PICKS - 1  # at most 13 distinct cards passed
T = N_PACKS * N_PICKS


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
    idx, copies = sparse(passed, MAX_PASS)
    mask = copies > 0
    b = np.concatenate([back, np.zeros(back.shape[:-1] + (1,), back.dtype)], -1)
    label = np.take_along_axis(b, idx.astype(np.int64), -1) > 0
    return idx.astype(np.int64), copies.astype(np.int64), label, mask


def taken_by_others(passed: np.ndarray, back: np.ndarray):
    """Sparse copies the other seven drafters took from the pack that returns at each pick.

    Empty for picks 0..7, whose pack has not been round the table yet.
    """
    D, _, _, C = passed.shape
    idx, cnt = sparse(passed - back, MAX_PASS)  # 7 normally; malformed records can have more
    out_i = np.full((D, N_PACKS, N_PICKS, MAX_PASS), C, np.int16)
    out_c = np.zeros(out_i.shape, np.int8)
    out_i[:, :, N_SEATS:], out_c[:, :, N_SEATS:] = idx, cnt
    return out_i, out_c


def card_features(meta) -> np.ndarray:
    """[C, F] colours, rarity one-hot, mana value, creature, bonus-sheet flag per card."""
    x = [meta.select(list(COLORS)).to_numpy(),
         np.stack([(meta["rarity"] == r).to_numpy() for r in RARITIES], 1),
         meta["mana_value"].clip(0, 8).to_numpy()[:, None] / 8,
         meta.select("is_creature", "is_bonus").to_numpy()]
    return np.concatenate([np.asarray(a, np.float32) for a in x], 1)


class DraftDataset(torch.utils.data.Dataset):
    def __init__(self, t: DraftTensors, passed, back, drafts: np.ndarray):
        C = len(t.names)
        n = len(drafts)
        self.pack, self.pack_n = (a.reshape(n, T, -1) for a in sparse(t.pack[drafts], N_PICKS))
        self.oth, self.oth_n = (a.reshape(n, T, -1)
                                for a in taken_by_others(passed[drafts], back[drafts]))
        self.pick = t.pick[drafts].reshape(len(drafts), -1).astype(np.int64)
        self.idx, self.copies, self.label, self.mask = pack_tokens(passed[drafts], back[drafts], C)
        self.drafts = drafts

    def __len__(self):
        return len(self.drafts)

    def __getitem__(self, i):
        return (self.pack[i], self.pack_n[i], self.oth[i], self.oth_n[i], self.pick[i],
                self.idx[i], self.copies[i], self.label[i], self.mask[i])


def collate(batch):
    cols = list(zip(*batch))
    return [torch.from_numpy(np.stack(c)) for c in cols]


class WheelNet(nn.Module):
    def __init__(self, n_cards: int, d: int = 192, heads: int = 6, hist_layers: int = 4,
                 pack_layers: int = 3, dropout: float = 0.2, cross: bool = True,
                 others: bool = True, tied_bag: bool = True, card_feats: np.ndarray | None = None):
        super().__init__()
        self.C, self.cross, self.use_others, self.tied_bag = n_cards, cross, others, tied_bag
        self.card_ = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        if card_feats is not None:  # known attributes seed each card's learned embedding
            f = torch.as_tensor(card_feats, dtype=torch.float32)
            self.register_buffer("feats", torch.cat([f, f.new_zeros(1, f.shape[1])]))
            self.feat_proj = nn.Linear(f.shape[1], d, bias=False)
        else:
            self.feats = None
        if tied_bag:  # bags are count-weighted means of the shared card embeddings
            self.bag = nn.Linear(d, d)
            self.others = nn.Linear(d, d)
        else:
            self.bag_emb = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
            self.others_emb = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        self.taken = nn.Linear(d, d)
        self.pos = nn.Embedding(T, d)
        self.copies = nn.Embedding(4, d)
        self.pick_pos = nn.Embedding(N_PACKS * WHEEL_PICKS, d)
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
        wp = torch.tensor([k * N_PICKS + p for k in range(N_PACKS) for p in range(WHEEL_PICKS)])
        self.register_buffer("wheel_pos", wp)
        # cross-attention may only see history up to and including the pick being scored
        future = torch.arange(T)[None] > wp[:, None]                              # [18, 42]
        self.register_buffer("future_tok", future.repeat_interleave(MAX_PASS, 0))  # [234, 42]
        pos = torch.arange(len(wp)).repeat_interleave(MAX_PASS)
        self.register_buffer("block", pos[:, None] != pos[None])                   # [234, 234]

    def card_table(self):
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

    def forward(self, pack, pack_n, oth, oth_n, pick, idx, copies, mask):
        B = pack.shape[0]
        h = self._bag(pack, pack_n, "bag") + self.pos.weight[None]
        h = h + (self.taken(self.card(pick)) if self.tied_bag else self.card(pick))
        if self.use_others:
            h = h + self._bag(oth, oth_n, "others")
        h = self.hist_norm(self.hist(h, mask=self.causal, is_causal=True))  # [B, 42, d]
        state = h[:, self.wheel_pos]                                         # [B, 18, d]

        n_pos = N_PACKS * WHEEL_PICKS
        idx = idx.reshape(B * n_pos, MAX_PASS)
        m = mask.reshape(B * n_pos, MAX_PASS)
        x = (self.card(idx) + self.copies(copies.reshape(B * n_pos, MAX_PASS).clamp(max=3))
             + (self.ctx(state) + self.pick_pos.weight[None]).reshape(B * n_pos, 1, -1))
        # rows with no passed cards cannot happen (>=8 cards passed at picks 0..5)
        if self.cross:
            # all 18 packs of a draft in one sequence so the history is projected once per draft;
            # block-diagonal self-attention keeps the packs separate
            x = self.packenc(x.reshape(B, n_pos * MAX_PASS, -1), h, tgt_mask=self.block,
                             memory_mask=self.future_tok, tgt_key_padding_mask=~m.reshape(B, -1))
        else:
            x = self.packenc(x, src_key_padding_mask=~m)
        return self.head(x).squeeze(-1).reshape(B, N_PACKS, WHEEL_PICKS, MAX_PASS)
