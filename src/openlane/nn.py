"""WheelNet: a transformer that reads the raw draft history and scores every passed card.

History encoder: one token per pick seen so far (bag-of-cards of the pack shown + the card taken +
pack/pick position), causal self-attention, so the state at pick t only sees picks <= t. This
contains every signal the hand-built features try to summarise (colours flowing late, what
wheeled in earlier rounds, the user's pool).

Pack decoder: the cards being passed at a wheel-eligible pick form a set; each card token gets the
history state added, then set self-attention (cards shield each other) and a logit per card.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import N_PACKS, N_PICKS, N_SEATS, DraftTensors

WHEEL_PICKS = N_PICKS - N_SEATS
MAX_PASS = N_PICKS - 1  # at most 13 distinct cards passed


def pack_tokens(passed: np.ndarray, back: np.ndarray, C: int):
    """Distinct passed cards per (d,k,p), padded with index C. Returns idx, copies, label, mask."""
    D = passed.shape[0]
    flat_p = passed.reshape(-1, C)
    flat_b = back.reshape(-1, C)
    present = flat_p > 0
    # stable sort puts present cards first, in card-index order
    order = np.argsort(~present, axis=1, kind="stable")[:, :MAX_PASS]
    mask = np.take_along_axis(present, order, 1)
    idx = np.where(mask, order, C).astype(np.int64)
    copies = np.take_along_axis(flat_p, order, 1) * mask
    label = (np.take_along_axis(flat_b, order, 1) > 0) & mask
    shp = (D, N_PACKS, WHEEL_PICKS, MAX_PASS)
    return idx.reshape(shp), copies.reshape(shp).astype(np.int64), label.reshape(shp), mask.reshape(shp)


class DraftDataset(torch.utils.data.Dataset):
    def __init__(self, t: DraftTensors, passed, back, drafts: np.ndarray):
        C = len(t.names)
        self.pack = t.pack[drafts].reshape(len(drafts), N_PACKS * N_PICKS, C)
        self.pick = t.pick[drafts].reshape(len(drafts), -1).astype(np.int64)
        self.idx, self.copies, self.label, self.mask = pack_tokens(passed[drafts], back[drafts], C)
        self.drafts = drafts

    def __len__(self):
        return len(self.drafts)

    def __getitem__(self, i):
        return (self.pack[i], self.pick[i], self.idx[i], self.copies[i], self.label[i], self.mask[i])


def collate(batch):
    cols = list(zip(*batch))
    return [torch.from_numpy(np.stack(c)) for c in cols]


class WheelNet(nn.Module):
    def __init__(self, n_cards: int, d: int = 128, heads: int = 4, hist_layers: int = 3,
                 pack_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.C = n_cards
        self.card = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        self.bag = nn.Linear(n_cards, d, bias=False)  # pack contents -> token
        self.pos = nn.Embedding(N_PACKS * N_PICKS, d)
        self.copies = nn.Embedding(4, d)
        self.pick_pos = nn.Embedding(N_PACKS * WHEEL_PICKS, d)
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                                   norm_first=True)
        self.hist = nn.TransformerEncoder(layer(), hist_layers, enable_nested_tensor=False)
        self.packenc = nn.TransformerEncoder(layer(), pack_layers, enable_nested_tensor=False)
        self.ctx = nn.Linear(d, d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        T = N_PACKS * N_PICKS
        self.register_buffer("causal", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))
        # history position of every wheel-eligible pick
        wp = [k * N_PICKS + p for k in range(N_PACKS) for p in range(WHEEL_PICKS)]
        self.register_buffer("wheel_pos", torch.tensor(wp))

    def forward(self, pack, pick, idx, copies, mask):
        B = pack.shape[0]
        # history: pack bag normalised by pack size, plus taken card, plus position
        bag = pack.float()
        bag = bag / bag.sum(-1, keepdim=True).clamp(min=1)
        h = self.bag(bag) + self.card(pick) + self.pos.weight[None]
        h = self.hist(h, mask=self.causal, is_causal=True)          # [B, 42, d]
        state = h[:, self.wheel_pos]                                 # [B, 18, d]

        n_pos = N_PACKS * WHEEL_PICKS
        idx = idx.reshape(B * n_pos, MAX_PASS)
        m = mask.reshape(B * n_pos, MAX_PASS)
        x = (self.card(idx) + self.copies(copies.reshape(B * n_pos, MAX_PASS).clamp(max=3))
             + (self.ctx(state) + self.pick_pos.weight[None]).reshape(B * n_pos, 1, -1))
        # rows with no passed cards cannot happen (>=8 cards passed at picks 0..5)
        x = self.packenc(x, src_key_padding_mask=~m)
        return self.head(x).squeeze(-1).reshape(B, N_PACKS, WHEEL_PICKS, MAX_PASS)
