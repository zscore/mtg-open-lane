"""PickNet: predict which card a human takes from the pack, given everything they have seen so far.

A pick is a discrete choice among the cards in the pack (a softmax over exactly those cards, as in
DraftFM and McFadden's conditional logit). Models, from simplest to richest:

  rating      logit = u_c: one learned number per card (a "pick order" / ratings bot)
  pool        logit = u_c + <e_c, mean of pool card embeddings>: a bilinear card x pool model, the
              learned version of "stay in your colours"
  picknet     a causal transformer over the draft so far, then the candidate cards attend to each
              other and to that history

PickNet history token t (one per pick): count-weighted bag of the pack shown at t, the card taken
at t-1, the cards the other seven took from the pack that returned at t (picks 8+ of a round),
position, and per-draft context (rank, skill, time). Causal self-attention, so state t sees packs
<= t and picks < t: exactly what the drafter knew when choosing. Candidate tokens (card embedding +
copies + projected state t) alternate self-attention within the pack (which other cards compete)
with cross-attention to the history up to t, then a per-card logit.

`history=False` drops the transformer over the history: the state at t is then only the bag of the
pool so far (+ position, context), which tests whether signals (what was passed, what wheeled)
matter beyond the pool, the puder question.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_PACKS, N_SEATS, DraftTensors
from .nn import sparse


# ---------------------------------------------------------------- data

def others_taken(t: DraftTensors) -> np.ndarray:
    """[D, K, N, C] copies the other seven took from the pack that comes back at each pick (zero for
    the first 8 picks of a round): pack shown at p-8, minus the pick made then, minus pack at p."""
    D, K, N, C = t.pack.shape
    out = np.zeros_like(t.pack)
    prev = t.pack[:, :, :N - N_SEATS].copy()
    pk = t.pick[:, :, :N - N_SEATS].astype(np.int64)
    np.put_along_axis(prev, pk[..., None], np.take_along_axis(prev, pk[..., None], -1) - 1, -1)
    out[:, :, N_SEATS:] = np.clip(prev - t.pack[:, :, N_SEATS:], 0, None)
    return out


class PickDataset(torch.utils.data.Dataset):
    """Sparse per-draft arrays: pack tokens [T, P], copies, target slot [T], picked card [T],
    others-taken tokens [T, 10], context [F], loss mask [T]."""

    def __init__(self, t: DraftTensors, drafts: np.ndarray, ctx: np.ndarray | None = None,
                 oth: np.ndarray | None = None):
        n, T, C = len(drafts), N_PACKS * t.n_picks, len(t.names)
        sel = slice(drafts[0], drafts[-1] + 1) if (np.diff(drafts) == 1).all() else drafts
        idx, cnt = sparse(t.pack[sel], t.n_picks + 1)
        self.idx, self.cnt = idx.reshape(n, T, -1), cnt.reshape(n, T, -1)
        self.pick = t.pick[sel].reshape(n, T).astype(np.int64)
        self.target = (self.idx == self.pick[..., None]).argmax(-1)
        self.valid = np.ones((n, T), bool)
        if t.p1p1_missing:
            self.valid[:, 0] = False
        oth = others_taken(t) if oth is None else oth
        oi, oc = sparse(oth[sel], N_SEATS + 2)
        self.oidx, self.ocnt = oi.reshape(n, T, -1), oc.reshape(n, T, -1)
        self.ctx = np.zeros((n, 0), np.float32) if ctx is None else ctx[sel].astype(np.float32)
        self.drafts = drafts
        self.C = C

    def __len__(self):
        return len(self.drafts)

    def __getitem__(self, i):
        return (self.idx[i], self.cnt[i], self.pick[i], self.oidx[i], self.ocnt[i], self.ctx[i],
                self.target[i], self.valid[i])


def collate(batch):
    return [torch.from_numpy(np.stack(c)) for c in zip(*batch)]


# ---------------------------------------------------------------- model

class CardTable(nn.Module):
    """Card embeddings: learned per-card vectors, plus a projection of fixed card features. With
    `ids=False` a card is only its features (an MLP of them), so the model can score unseen cards.
    `id_map` [n_cards] maps each card (a printing) to a name id: printings that share a name (reprints,
    basic lands) then share one learned vector, while their features stay per printing."""

    def __init__(self, n_cards: int, d: int, feats: np.ndarray | None = None, ids: bool = True,
                 id_map: np.ndarray | None = None):
        super().__init__()
        self.ids = ids
        self.register_buffer("id_map", None)
        if ids and id_map is not None:
            m = torch.as_tensor(np.asarray(id_map), dtype=torch.long)
            n_names = int(m.max()) + 1
            self.emb = nn.Embedding(n_names + 1, d, padding_idx=n_names)
            self.id_map = torch.cat([m, m.new_full((1,), n_names)])
        elif ids:
            self.emb = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        if feats is not None:
            f = torch.as_tensor(feats, dtype=torch.float32)
            self.register_buffer("feats", torch.cat([f, f.new_zeros(1, f.shape[1])]))
            self.proj = (nn.Linear(f.shape[1], d, bias=False) if ids else
                         nn.Sequential(nn.Linear(f.shape[1], 2 * d), nn.GELU(), nn.Linear(2 * d, d)))
        else:
            assert ids, "a features-only card table needs features"
            self.feats = None

    def forward(self) -> torch.Tensor:
        w = (self.emb.weight if self.id_map is None else self.emb(self.id_map)) if self.ids else 0
        if self.feats is not None:
            w = w + self.proj(self.feats)
        if not self.ids:  # padding row stays zero
            w = w * (torch.arange(len(w), device=w.device) < len(w) - 1)[:, None]
        return w


def bag(table: torch.Tensor, idx: torch.Tensor, cnt: torch.Tensor, mean: bool = True) -> torch.Tensor:
    w = cnt.float()
    if mean:
        w = w / w.sum(-1, keepdim=True).clamp(min=1)
    return (w[..., None] * F.embedding(idx.long(), table)).sum(-2)


class RatingModel(nn.Module):
    """logit = u_c (+ optional pick-position slope)."""

    def __init__(self, n_cards: int, **_):
        super().__init__()
        self.u = nn.Embedding(n_cards + 1, 1)

    def forward(self, idx, cnt, pick, oidx, ocnt, ctx):
        return self.u(idx.long()).squeeze(-1)


class PoolModel(nn.Module):
    """logit = u_c + <a_c, mean pool embedding> + <b_c, position embedding>, rank-d bilinear."""

    def __init__(self, n_cards: int, n_picks: int, d: int = 32, feats=None, **_):
        super().__init__()
        self.u = nn.Embedding(n_cards + 1, 1)
        self.cards = CardTable(n_cards, d, feats)
        self.a = nn.Embedding(n_cards + 1, d)
        self.pos = nn.Embedding(N_PACKS * n_picks, d)
        for e in (self.a, self.pos):
            nn.init.normal_(e.weight, std=0.02)
        self.C = n_cards

    def forward(self, idx, cnt, pick, oidx, ocnt, ctx):
        B, T, P = idx.shape
        tab = self.cards()
        # pool before pick t: cumulative sum of picked-card embeddings, excluding t
        e = F.embedding(pick, tab)
        pool = (torch.cumsum(e, 1) - e) / torch.arange(T, device=e.device).clamp(min=1)[None, :, None]
        state = pool + self.pos.weight[None, :T]
        return self.u(idx.long()).squeeze(-1) + (self.a(idx.long()) * state[:, :, None]).sum(-1)


class Block(nn.Module):
    """Pre-norm candidate block: self-attention within the pack, cross-attention to history, MLP."""

    def __init__(self, d: int, heads: int, dropout: float, cross: bool = True):
        super().__init__()
        self.h, self.p, self.cross = heads, dropout, cross
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o1 = nn.Linear(d, d)
        if cross:
            self.q2, self.kv2, self.o2 = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def _heads(self, x):  # [..., L, d] -> [..., h, L, dh]
        return x.unflatten(-1, (self.h, -1)).transpose(-3, -2)

    def forward(self, x, pad, mem, mem_mask):
        """x [B, T, P, d] candidates, pad [B, T, P] True=real, mem [B, T, d], mem_mask [T, T] True=see."""
        B, T, P, d = x.shape
        dp = self.p if self.training else 0.0
        q, k, v = self._heads(self.qkv(self.ln1(x)).reshape(B * T, P, 3 * d)).chunk(3, -1)  # [BT, h, P, dh]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=pad.reshape(B * T, 1, 1, P), dropout_p=dp)
        x = x + self.drop(self.o1(a.transpose(1, 2).reshape(B, T, P, d)))
        if self.cross:
            q = self._heads(self.q2(self.ln2(x)).reshape(B, T * P, d))             # [B, h, TP, dh]
            k, v = self._heads(self.kv2(mem)).chunk(2, -1)                          # [B, h, T, dh]
            m = mem_mask.repeat_interleave(P, 0)                                     # [TP, T]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=dp)
            x = x + self.drop(self.o2(a.transpose(1, 2).reshape(B, T, P, d)))
        return x + self.drop(self.mlp(self.ln3(x)))


class PickNet(nn.Module):
    def __init__(self, n_cards: int, n_picks: int, d: int = 128, heads: int = 4, hist_layers: int = 3,
                 pack_layers: int = 2, dropout: float = 0.1, feats: np.ndarray | None = None,
                 ids: bool = True, n_ctx: int = 0, history: bool = True, others: bool = True,
                 cross: bool = True, id_map: np.ndarray | None = None, n_sets: int = 0,
                 pack_drop: float = 0.0, deck_head: bool = False):
        super().__init__()
        T = N_PACKS * n_picks
        self.history, self.use_others, self.n_ctx = history, others, n_ctx
        self.n_sets, self.pack_drop = n_sets, pack_drop
        self.cards = CardTable(n_cards, d, feats, ids, id_map)
        if n_sets:
            self.set_emb = nn.Embedding(n_sets, d)
            nn.init.normal_(self.set_emb.weight, std=0.02)
        if deck_head:
            self.deck_q, self.deck_k = nn.Linear(d, d), nn.Linear(d, d)
        self.pack_bag, self.pick_in, self.oth_bag, self.pool_bag = (nn.Linear(d, d) for _ in range(4))
        self.pos = nn.Embedding(T, d)
        self.copies = nn.Embedding(4, d)
        if n_ctx:
            self.ctx_mlp = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d))
        if history:
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            self.hist = nn.TransformerEncoder(layer, hist_layers, enable_nested_tensor=False)
        self.hist_norm = nn.LayerNorm(d)
        self.state_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList(Block(d, heads, dropout, cross and history) for _ in range(pack_layers))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("causal", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))
        self.C = n_cards

    def forward(self, idx, cnt, pick, oidx, ocnt, ctx, sid=None, return_state=False):
        """sid [B]: set index (with n_sets). return_state also returns the history states and card
        table, for deck_logits. In training, `pack_drop` hides the shown pack and what the others took
        from the history tokens of that share of drafts (puder's pack dropout); the candidates are
        always the pack."""
        B, T, P = idx.shape
        tab = self.cards()
        e_pick = F.embedding(pick, tab)
        prev = torch.cat([torch.zeros_like(e_pick[:, :1]), e_pick[:, :-1]], 1)  # card taken at t-1
        keep = 1.0
        if self.training and self.pack_drop > 0:
            keep = (torch.rand(B, 1, 1, device=idx.device) >= self.pack_drop).to(tab.dtype)
        h = self.pos.weight[None, :T] + keep * self.pack_bag(bag(tab, idx, cnt))
        if self.n_ctx:
            h = h + self.ctx_mlp(ctx)[:, None]
        if self.n_sets:
            h = h + self.set_emb(sid.long())[:, None]
        if self.history:
            h = h + self.pick_in(prev)
            if self.use_others:
                h = h + keep * self.oth_bag(bag(tab, oidx, ocnt))
            h = self.hist(h, mask=self.causal[:T, :T], is_causal=True)
        else:  # pool so far as a bag, no history attention
            pool = (torch.cumsum(e_pick, 1) - e_pick) / torch.arange(T, device=h.device).clamp(min=1)[None, :, None]
            h = h + self.pool_bag(pool)
        h = self.hist_norm(h)
        pad = cnt > 0
        pad[..., 0] |= ~pad.any(-1)  # an empty pack (missing P1P1) attends to its first slot; never scored
        x = F.embedding(idx.long(), tab) + self.copies(cnt.long().clamp(max=3)) + self.state_proj(h)[:, :, None]
        see = ~self.causal[:T, :T]
        for blk in self.blocks:
            x = blk(x, pad, h, see)
        logit = self.head(x).squeeze(-1)
        return (logit, h, tab) if return_state else logit

    def deck_logits(self, h, tab, pick):
        """[B, T, T] logit that pick j ends up in the main deck, from history state t (puder's
        deck-inclusion head). State t has seen picks < t, so only j < t is meaningful."""
        q, k = self.deck_q(h), self.deck_k(F.embedding(pick, tab))
        return q @ k.transpose(1, 2) / q.shape[-1] ** 0.5


MODELS = {"rating": RatingModel, "pool": PoolModel, "picknet": PickNet}


def pick_loss(logit, cnt, target, valid):
    """Mean NLL of the chosen card over valid picks; logits [B, T, P]."""
    logit = logit.masked_fill(cnt == 0, float("-inf"))
    lp = logit.log_softmax(-1)
    nll = -lp.gather(-1, target[..., None]).squeeze(-1)
    return nll[valid].mean(), lp


def load_model(path, device="cpu"):
    """Rebuild a model saved by scripts/train_picknet.py."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_ctx = next((v.shape[1] for k, v in ck["state"].items() if k.endswith("ctx_mlp.0.weight")), 0)
    m = MODELS[ck["model"]](ck["n_cards"], n_picks=ck["n_picks"], feats=ck["feats"], n_ctx=n_ctx, **ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck
