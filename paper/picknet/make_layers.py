"""Build layers.html: detailed diagrams of the history encoder, deck head and pack decoder.

  python paper/picknet/make_layers.py
"""
from pathlib import Path

OUT = Path(__file__).parent / "layers.html"
_n = [0]


def svg(w, h, body, label):
    _n[0] += 1
    mid = f"ah{_n[0]}"
    body = body.replace("ARROW", mid)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" style="max-width:{w}px">'
            f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>'
            f'{body}</svg>')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls="lbl", anchor="middle", extra=""):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'


def box(x, y, w, h, cls, label, sub=None):
    cx, cy = x + w / 2, y + h / 2
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="{cls}"/>'
    if sub:
        out += text(cx, cy - 3, label) + text(cx, cy + 13, sub, "sub")
    else:
        out += text(cx, cy + 4.5, label)
    return out


def arrow(*pts, cls="edge"):
    p = " ".join(f"{x},{y}" for x, y in pts)
    return f'<polyline points="{p}" class="{cls}" marker-end="url(#ARROW)"/>'


def line(*pts, cls="edge"):
    p = " ".join(f"{x},{y}" for x, y in pts)
    return f'<polyline points="{p}" class="{cls}"/>'


def plus(cx, cy, r=13):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" class="box-op"/>'
            f'<line x1="{cx-6}" y1="{cy}" x2="{cx+6}" y2="{cy}" class="edge"/>'
            f'<line x1="{cx}" y1="{cy-6}" x2="{cx}" y2="{cy+6}" class="edge"/>')


def dot(x, y):
    return f'<circle cx="{x}" cy="{y}" r="3.2" fill="currentColor"/>'


def vlabel(x, y, s, cls="grp-lbl"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="middle" transform="rotate(-90 {x} {y})">{esc(s)}</text>'


# ------------------------------------------------------------------ overview
def overview():
    b = ""
    b += box(20, 30, 170, 50, "box-d", "45 history tokens", "[B, 45, 256]")
    b += box(240, 30, 180, 50, "box-w", "History encoder", "4 layers · 3.16M")
    b += box(470, 30, 150, 50, "box-d", "states h", "[B, 45, 256]")
    b += box(670, 30, 150, 50, "box-k", "Deck head", "training only")
    b += box(240, 160, 180, 50, "box-d", "current pack", "[B, 45, 15] cards")
    b += box(470, 160, 150, 50, "box-w", "Pack decoder", "3 blocks · 3.16M")
    b += box(670, 160, 150, 50, "box-o", "p(card)", "[B, 45, 15]")
    b += arrow((190, 55), (238, 55))
    b += arrow((420, 55), (468, 55))
    b += text(445, 47, "norm", "tiny")
    b += arrow((620, 55), (668, 55))
    b += arrow((545, 80), (545, 158))
    b += text(553, 124, "state + cross-attn", "tiny", "start")
    b += arrow((420, 185), (468, 185))
    b += arrow((620, 185), (668, 185))
    return svg(840, 230, b, "Overview: history tokens go through the history encoder to states h; h feeds the deck head and the pack decoder, which scores the current pack.")


# ------------------------------------------------------------------ encoder layer
def encoder_layer():
    b = ""
    X0, W, CX = 200, 260, 330
    SX, PX = 490, 670
    b += text(SX, 22, "tensor shape", "colhead", "start") + text(PX, 22, "weights", "colhead", "start")

    def row(y, cls, label, shape=None, params=None, x=X0, w=W):
        out = box(x, y, w, 36, cls, label)
        if shape:
            out += text(SX, y + 22, shape, "sub", "start")
        if params:
            out += text(PX, y + 22, params, "pc", "start")
        return out

    # groups
    b += f'<rect x="178" y="100" width="304" height="556" rx="10" class="grp"/>'
    b += f'<rect x="178" y="714" width="304" height="292" rx="10" class="grp"/>'
    b += vlabel(56, 380, "ATTENTION SUBLAYER") + vlabel(76, 380, "tokens read each other", "grp-sub")
    b += vlabel(56, 860, "MLP SUBLAYER") + vlabel(76, 860, "each token on its own", "grp-sub")

    b += row(40, "box-d", "x  (residual stream in)", "[B, 45, 256]")
    b += row(108, "box-w", "LayerNorm 1", "[B, 45, 256]", "512")
    b += row(168, "box-w", "Linear W_qkv  256 → 768", "[B, 45, 768]", "197,376")
    b += box(200, 234, 80, 36, "box-d", "Q") + box(290, 234, 80, 36, "box-d", "K") + box(380, 234, 80, 36, "box-d", "V")
    b += text(SX, 256, "[B, 8, 45, 32] each", "sub", "start")
    b += row(300, "box-op", "Q · Kᵀ / √32", "[B, 8, 45, 45]", w=170)
    b += row(360, "box-m", "causal mask: future → −∞", "[B, 8, 45, 45]", w=170)
    b += row(420, "box-op", "softmax each row", "rows sum to 1", w=170)
    b += row(486, "box-op", "A · V", "[B, 8, 45, 32]")
    b += row(552, "box-w", "concat 8 heads → Linear W_o", "[B, 45, 256]", "65,792")
    b += row(612, "box-op", "Dropout 0.05")
    b += plus(CX, 680)
    b += row(722, "box-w", "LayerNorm 2", "[B, 45, 256]", "512")
    b += row(782, "box-w", "Linear  256 → 1024", "[B, 45, 1024]", "263,168")
    b += row(842, "box-op", "ReLU → Dropout", "[B, 45, 1024]")
    b += row(902, "box-w", "Linear  1024 → 256", "[B, 45, 256]", "262,400")
    b += row(962, "box-op", "Dropout 0.05")
    b += plus(CX, 1030)
    b += row(1060, "box-d", "x  (to next layer)", "[B, 45, 256]")

    for y0, y1 in [(76, 106), (144, 166), (588, 610), (648, 665), (693, 720), (758, 780), (818, 840),
                   (878, 900), (938, 960), (998, 1015), (1043, 1058), (522, 550)]:
        b += arrow((CX, y0), (CX, y1))
    for x in (240, 330, 420):
        b += arrow((x, 204), (x, 232))
    b += arrow((240, 270), (240, 298)) + arrow((330, 270), (330, 298))
    b += arrow((420, 270), (420, 484))
    for y0, y1 in [(336, 358), (396, 418), (456, 484)]:
        b += arrow((285, y0), (285, y1))
    # residual skips
    b += dot(CX, 92) + arrow((CX, 92), (120, 92), (120, 680), (315, 680), cls="skip")
    b += dot(CX, 706) + arrow((CX, 706), (120, 706), (120, 1030), (315, 1030), cls="skip")
    b += text(128, 670, "+ x", "tiny", "start") + text(128, 1020, "+ x", "tiny", "start")
    return svg(780, 1110, b, "One pre-LN history encoder layer: LayerNorm, causal multi-head attention, residual add, LayerNorm, ReLU MLP, residual add.")


def grid(ox, oy, n, cell, state, labels_top, labels_left, title_top=None, title_left=None, txt=None):
    """state(r, c) -> css class or None."""
    b = ""
    for r in range(n):
        for c in range(n):
            cls = state(r, c)
            if cls:
                b += f'<rect x="{ox + c*cell}" y="{oy + r*cell}" width="{cell}" height="{cell}" class="{cls}"/>'
            if txt:
                s = txt(r, c)
                if s:
                    b += text(ox + c*cell + cell/2, oy + r*cell + cell/2 + 4, s, "cell-t")
    for i, s in enumerate(labels_top):
        b += text(ox + i*cell + cell/2, oy - 8, s, "sub")
    for i, s in enumerate(labels_left):
        b += text(ox - 8, oy + i*cell + cell/2 + 4, s, "sub", "end")
    if title_top:
        b += text(ox + n*cell/2, oy - 30, title_top, "axis")
    if title_left:
        b += vlabel(ox - 52, oy + n*cell/2, title_left, "axis")
    return b


def causal_grid():
    n, cell, ox, oy = 10, 26, 100, 70
    b = grid(ox, oy, n, cell, lambda r, c: "cell-on" if c <= r else "cell-off",
             [str(j) for j in range(n)], [str(t) for t in range(n)],
             "j: token being read", "t: token doing the reading",
             txt=lambda r, c: "−∞" if c == r + 1 else None)
    # highlight row 6
    b += f'<rect x="{ox}" y="{oy + 6*cell}" width="{n*cell}" height="{cell}" class="hl"/>'
    lx = 400
    b += f'<rect x="{lx}" y="76" width="18" height="18" class="cell-on"/>' + text(lx + 28, 90, "can read (j ≤ t)", "lgd", "start")
    b += f'<rect x="{lx}" y="106" width="18" height="18" class="cell-off"/>' + text(lx + 28, 120, "masked: score set to −∞", "lgd", "start")
    b += arrow((lx - 10, oy + 6.5*cell), (ox + n*cell + 6, oy + 6.5*cell))
    b += text(lx, 220, "Row 6: token 6 holds pack 6 and", "lgd", "start")
    b += text(lx, 238, "pick 5. It reads tokens 0–6, so it", "lgd", "start")
    b += text(lx, 256, "sees picks 0–5 and never pick 6,", "lgd", "start")
    b += text(lx, 274, "the card it must predict.", "lgd", "start")
    return svg(680, 345, b, "Causal mask: token t may attend to tokens 0 through t; everything above the diagonal is set to minus infinity.")


# ------------------------------------------------------------------ deck head
def deck_head():
    b = ""
    b += box(20, 30, 180, 46, "box-d", "h  history states", "[B, 45, 256]")
    b += box(290, 30, 170, 46, "box-k", "Linear deck_q", "65,792 weights")
    b += box(540, 30, 110, 46, "box-d", "Q", "[B, 45, 256]")
    b += box(20, 120, 130, 46, "box-d", "picks", "card ids [B, 45]")
    b += box(170, 120, 100, 46, "box-w", "card table", "lookup")
    b += box(290, 120, 170, 46, "box-k", "Linear deck_k", "65,792 weights")
    b += box(540, 120, 110, 46, "box-d", "K", "[B, 45, 256]")
    b += box(690, 75, 130, 46, "box-op", "Q · Kᵀ / 16", "√256 = 16")
    b += box(690, 180, 130, 46, "box-d", "logits", "[B, 45 t, 45 j]")
    b += box(290, 270, 250, 46, "box-d", "target y_j: maindeck rate", "17Lands, 0–1, same for all t")
    b += box(290, 340, 250, 46, "box-m", "keep t > j, valid t, known y_j", "weight (t+1)/45")
    b += box(690, 270, 130, 116, "box-op", "weighted BCE", "→ deck loss")
    b += box(690, 420, 130, 46, "box-k", "NLL + 0.3 · deck", "total loss")
    b += arrow((200, 53), (288, 53)) + arrow((460, 53), (538, 53))
    b += arrow((150, 143), (168, 143)) + arrow((270, 143), (288, 143)) + arrow((460, 143), (538, 143))
    b += arrow((650, 53), (670, 53), (670, 90), (688, 90))
    b += arrow((650, 143), (670, 143), (670, 106), (688, 106))
    b += arrow((755, 121), (755, 178)) + arrow((755, 226), (755, 268))
    b += arrow((540, 293), (688, 293)) + arrow((540, 363), (688, 363))
    b += arrow((755, 386), (755, 418))
    b += arrow((690, 443), (8, 443), (8, 53), (18, 53), cls="grad")
    b += text(20, 435, "gradients flow back into the encoder and the card table", "tiny", "start")
    b += text(680, 198, "row t: a history state", "tiny", "end")
    b += text(680, 214, "column j: a pick", "tiny", "end")
    return svg(840, 480, b, "Deck head: a bilinear score between history state t and the embedding of pick j, trained with weighted binary cross-entropy against each pick's maindeck rate.")


def deck_grid():
    n, cell, ox, oy = 10, 26, 100, 96
    ys = ["1", "1", "0", "1", ".5", "0", "1", "1", "0", "1"]

    def st(r, c):
        if c < r:
            return f"cell-k k{min(4, r * 5 // n)}"
        return "cell-blank"
    b = grid(ox, oy, n, cell, st, [str(j) for j in range(n)], [str(t) for t in range(n)],
             None, "t: history state")
    b += text(ox + n*cell/2, 26, "j: pick being judged", "axis")
    for j, y in enumerate(ys):
        b += text(ox + j*cell + cell/2, oy - 30, y, "sub-k")
    b += text(ox - 8, oy - 26, "y_j", "sub-k", "end")
    b += f'<rect x="{ox}" y="{oy + 6*cell}" width="{n*cell}" height="{cell}" class="hl"/>'
    lx = 400
    b += f'<rect x="{lx}" y="100" width="18" height="18" class="cell-k k1"/>' + text(lx + 28, 114, "trained: state t has seen pick j", "lgd", "start")
    b += f'<rect x="{lx}" y="130" width="18" height="18" class="cell-blank"/>' + text(lx + 28, 144, "ignored: pick j not seen yet", "lgd", "start")
    b += text(lx, 172, "darker = later state = more weight", "lgd", "start")
    b += arrow((lx - 10, oy + 6.5*cell), (ox + n*cell + 6, oy + 6.5*cell))
    b += text(lx, 250, "Row 6: after picks 0–5, guess which", "lgd", "start")
    b += text(lx, 268, "of them end up in the main deck.", "lgd", "start")
    b += text(lx, 286, "Targets: 1, 1, 0, 1, 0.5, 0.", "lgd", "start")
    return svg(680, 370, b, "Deck head targets: cell (t, j) is trained only when t is greater than j; each column's target is that pick's maindeck rate.")


# ------------------------------------------------------------------ pack decoder
def cand_tokens():
    b = ""
    rows = [("pack at slot t", "idx [B, 45, 15]", "card table", "lookup", "card vectors", "[B, 45, 15, 256]"),
            ("copies of each", "cnt [B, 45, 15], 1–4", "copies emb", "4 × 256 = 1,024", "copy vectors", "[B, 45, 15, 256]"),
            ("history state h_t", "[B, 45, 256]", "state_proj", "65,792 weights", "same for all 15", "[B, 45, 1, 256]")]
    for i, (a, asub, m, msub, o, osub) in enumerate(rows):
        y = 30 + i * 74
        b += box(20, y, 170, 46, "box-d", a, asub)
        b += box(230, y, 130, 46, "box-w", m, msub)
        b += box(400, y, 170, 46, "box-d", o, osub)
        b += arrow((190, y + 23), (228, y + 23)) + arrow((360, y + 23), (398, y + 23))
    b += arrow((570, 53), (632, 53), (632, 111)) + arrow((570, 127), (616, 127)) + arrow((570, 201), (632, 201), (632, 143))
    b += plus(632, 127, 14)
    b += arrow((646, 127), (678, 127))
    b += box(680, 104, 150, 46, "box-d", "x  candidates", "[B, 45, 15, 256]")
    return svg(850, 230, b, "Candidate tokens: each card in the pack gets its card vector plus a copies vector plus the projected history state for that slot.")


def decoder_block():
    b = ""
    X0, W, CX, H = 200, 260, 330, 44
    RX, RW = 530, 250
    y = 30
    ys = {}

    def put(key, cls, label, sub=None, x=X0, w=W):
        nonlocal b
        b += box(x, ys[key], w, H, cls, label, sub)

    order = ["in", "s1", "ln1", "qkv", "sa", "o1", "d1", "p1", "s2", "ln2", "q2", "ca", "o2", "d2", "p2",
             "s3", "ln3", "l1", "act", "l2", "d3", "p3", "out"]
    for k in order:
        ys[k] = y
        y += 28 if k in ("s1", "s2", "s3", "p1", "p2", "p3") else 62
    put("in", "box-d", "x  candidates", "[B, 45, 15, 256]")
    put("ln1", "box-w", "LayerNorm 1", "512 weights")
    put("qkv", "box-w", "Linear W_qkv 256 → 768", "197,376 · → [B·45, 8, 15, 32]")
    put("sa", "box-op", "softmax(Q·Kᵀ/√32) · V", "each pack on its own")
    put("o1", "box-w", "Linear o1", "65,792 · [B, 45, 15, 256]")
    put("d1", "box-op", "Dropout 0.05")
    put("ln2", "box-w", "LayerNorm 2", "512 weights")
    put("q2", "box-w", "Linear q2  (queries)", "65,792 · [B, 8, 45·15, 32]")
    put("ca", "box-op", "softmax(Q·Kᵀ/√32) · V", "cards read the history")
    put("o2", "box-w", "Linear o2", "65,792 · [B, 45, 15, 256]")
    put("d2", "box-op", "Dropout 0.05")
    put("ln3", "box-w", "LayerNorm 3", "512 weights")
    put("l1", "box-w", "Linear 256 → 1024", "263,168 · [B, 45, 15, 1024]")
    put("act", "box-op", "GELU → Dropout")
    put("l2", "box-w", "Linear 1024 → 256", "262,400 · [B, 45, 15, 256]")
    put("d3", "box-op", "Dropout 0.05")
    put("out", "box-d", "x  (next block, ×3)", "[B, 45, 15, 256]")
    # right lane
    b += box(RX, ys["sa"], RW, H, "box-m", "pad mask: real cards only", "keys only · no order among cards")
    b += arrow((RX, ys["sa"] + 22), (X0 + W + 2, ys["sa"] + 22))
    b += box(RX, ys["ln2"], RW, H, "box-d", "h  history states", "[B, 45, 256] · already normed")
    b += box(RX, ys["q2"], RW, H, "box-w", "Linear kv2  256 → 512", "131,584 · K, V [B, 8, 45, 32]")
    b += arrow((RX + RW/2, ys["ln2"] + H), (RX + RW/2, ys["q2"] - 2))
    kvy = ys["q2"] + H + 9
    b += arrow((RX + RW/2, ys["q2"] + H), (RX + RW/2, kvy), (420, kvy), (420, ys["ca"] - 2))
    b += text(428, ys["ca"] - 3, "K, V", "tiny", "start")
    b += box(RX, ys["ca"], RW, H, "box-m", "slot t reads h_0 … h_t", "same causal rule as the encoder")
    b += arrow((RX, ys["ca"] + 22), (X0 + W + 2, ys["ca"] + 22))
    # main arrows
    seq = ["in", "ln1", "qkv", "sa", "o1", "d1", "p1", "ln2", "q2", "ca", "o2", "d2", "p2", "ln3", "l1", "act", "l2", "d3", "p3", "out"]
    for a, c in zip(seq, seq[1:]):
        y0 = ys[a] + 13 + 14 if a[0] == "p" else ys[a] + H
        y1 = ys[c] + 14 - 13 - 2 if c[0] == "p" else ys[c] - 2
        if a[0] == "p":
            y0 = ys[a] + 14 + 13
        if c[0] == "p":
            y1 = ys[c] + 14 - 13 - 2
        b += arrow((CX, y0), (CX, y1))
    groups = [("s1", "p1", "SELF-ATTENTION", "cards in one pack compare"),
              ("s2", "p2", "CROSS-ATTENTION", "cards read the draft so far"),
              ("s3", "p3", "MLP", "each card on its own")]
    for s, p, lab, sub in groups:
        cy = ys[p] + 14
        b += plus(CX, cy)
        sy = ys[s] + 14
        b += dot(CX, sy) + arrow((CX, sy), (120, sy), (120, cy), (315, cy), cls="skip")
        b += text(128, cy - 8, "+ x", "tiny", "start")
        top, bot = ys[s] + 22, ys[p]
        b += f'<rect x="178" y="{top}" width="304" height="{bot - top - 4}" rx="10" class="grp"/>'
        mid = (top + bot) / 2
        b += vlabel(56, mid, lab) + vlabel(76, mid, sub, "grp-sub")
    return svg(800, ys["out"] + H + 20, b, "One pack decoder block: self-attention among the cards of a pack, cross-attention from the cards to the history states, and a GELU MLP, each pre-normed with a residual add.")


def attn_patterns():
    b = ""
    # left: self-attention within pack
    n, cell, ox, oy = 8, 24, 90, 90
    real = 6
    b += text(ox + n*cell/2, 26, "self-attention within one pack", "axis")
    b += text(ox + n*cell/2, 44, "pick 10: 6 cards left (8 of 15 slots shown)", "tiny")
    b += grid(ox, oy, n, cell, lambda r, c: ("cell-off" if c >= real else ("cell-faint" if r >= real else "cell-on")),
              ["c" + str(j) for j in range(n)], ["c" + str(j) for j in range(n)])
    b += text(ox + n*cell/2, oy + n*cell + 26, "empty-slot columns masked;", "tiny")
    b += text(ox + n*cell/2, oy + n*cell + 42, "empty-slot rows computed, never scored", "tiny")
    # right: cross attention
    cell2, ox2, oy2, T, per = 22, 540, 90, 6, 2
    b += text(ox2 + T*cell2/2, 26, "cross-attention to the history", "axis")
    b += text(ox2 + T*cell2/2, 44, "2 of the 15 cards per slot shown", "tiny")
    for r in range(T * per):
        t = r // per
        for c in range(T):
            cls = "cell-on" if c <= t else "cell-off"
            b += f'<rect x="{ox2 + c*cell2}" y="{oy2 + r*cell2}" width="{cell2}" height="{cell2}" class="{cls}"/>'
    for c in range(T):
        b += text(ox2 + c*cell2 + cell2/2, oy2 - 8, f"h{c}", "sub")
    for t in range(T):
        b += text(ox2 - 10, oy2 + (t*per + 1)*cell2 + 4, f"slot {t}", "sub", "end")
        if t:
            b += f'<line x1="{ox2 - 4}" y1="{oy2 + t*per*cell2}" x2="{ox2 + T*cell2}" y2="{oy2 + t*per*cell2}" class="sep"/>'
    b += text(ox2 + T*cell2/2, oy2 + T*per*cell2 + 26, "queries: cards · keys: states", "tiny")
    return svg(840, 400, b, "Attention patterns: within a pack every real card sees every real card; in cross-attention a card at slot t sees history states 0 through t.")


def score_head():
    b = ""
    b += box(20, 30, 150, 46, "box-d", "x  after 3 blocks", "[B, 45, 15, 256]")
    b += box(195, 30, 100, 46, "box-w", "LayerNorm", "512")
    b += box(320, 30, 150, 46, "box-w", "Linear 256 → 256", "65,792")
    b += box(495, 30, 90, 46, "box-op", "GELU")
    b += box(610, 30, 150, 46, "box-w", "Linear 256 → 1", "257")
    for x0, x1 in [(170, 193), (295, 318), (470, 493), (585, 608)]:
        b += arrow((x0, 53), (x1, 53))
    b += box(20, 130, 150, 46, "box-d", "logit per card", "[B, 45, 15]")
    b += box(195, 130, 200, 46, "box-m", "empty slots → −∞", "cnt == 0")
    b += box(420, 130, 170, 46, "box-op", "log-softmax over 15", "a choice among the pack")
    b += box(615, 130, 205, 46, "box-o", "−log p(card taken)", "mean over valid picks")
    b += arrow((685, 76), (685, 103), (95, 103), (95, 128))
    for x0, x1 in [(170, 193), (395, 418), (590, 613)]:
        b += arrow((x0, 153), (x1, 153))
    return svg(840, 200, b, "Score head: LayerNorm, Linear, GELU, Linear to one logit per card; empty slots masked; log-softmax over the pack; negative log-likelihood of the card the human took.")


# ------------------------------------------------------------------ MLP
def mlp_neurons():
    b = ""
    xs = (150, 420, 690)
    ins = [110, 170, 230, 290]
    hid = [80, 132, 184, 236, 288, 340]
    outs = [110, 170, 230, 290]
    H = 2  # highlighted hidden unit
    for i, yi in enumerate(ins):
        for j, yh in enumerate(hid):
            b += f'<line x1="{xs[0]+16}" y1="{yi}" x2="{xs[1]-16}" y2="{yh}" class="{"nn-hi" if j == H else "nn-edge"}"/>'
    for j, yh in enumerate(hid):
        for k, yo in enumerate(outs):
            b += f'<line x1="{xs[1]+16}" y1="{yh}" x2="{xs[2]-16}" y2="{yo}" class="{"nn-hi2" if j == H else "nn-edge"}"/>'
    for i, y in enumerate(ins):
        b += f'<circle cx="{xs[0]}" cy="{y}" r="16" class="box-d"/>' + text(xs[0], y + 4.5, f"z{i+1}", "lbl")
    for j, y in enumerate(hid):
        b += f'<circle cx="{xs[1]}" cy="{y}" r="16" class="{"box-w" if j == H else "box-op"}"/>' + text(xs[1], y + 4.5, f"a{j+1}", "lbl")
    for k, y in enumerate(outs):
        b += f'<circle cx="{xs[2]}" cy="{y}" r="16" class="box-d"/>' + text(xs[2], y + 4.5, f"y{k+1}", "lbl")
    b += text(xs[0], 24, "input z", "axis") + text(xs[1], 24, "hidden a = ReLU(W1 z + b1)", "axis") + text(xs[2], 24, "output = W2 a + b2", "axis")
    for x, n in zip(xs, (256, 1024, 256)):
        b += text(x, 44, f"{n} in PickNet", "sub")
    b += text(285, 384, "solid: row 3 of W1, what unit 3 looks for", "tiny")
    b += text(555, 384, "dashed: column 3 of W2, what unit 3 writes", "tiny")
    return svg(840, 400, b, "A small MLP with 4 inputs, 6 hidden units and 4 outputs. Every input connects to every hidden unit and every hidden unit to every output. Unit 3's incoming and outgoing weights are highlighted.")


def mlp_matrix():
    b = ""
    cy = 135

    def mat(x, w, h, cls, name, shape, note=None):
        out = f'<rect x="{x}" y="{cy - h/2}" width="{w}" height="{h}" rx="4" class="{cls}"/>'
        out += text(x + w/2, cy + 5, name, "lbl")
        out += text(x + w/2, 258, shape, "sub")
        if note:
            out += text(x + w/2, 274, note, "tiny")
        return out
    b += mat(20, 50, 100, "box-d", "Z", "[N, 256]", "N tokens")
    b += text(86, cy + 6, "×", "op-sym")
    b += mat(102, 200, 50, "box-w", "W1", "[256, 1024]", "+ b1")
    b += text(318, cy + 6, "→", "op-sym")
    b += mat(336, 200, 100, "box-op", "ReLU( · )", "[N, 1024]", "hidden")
    b += text(552, cy + 6, "×", "op-sym")
    b += mat(568, 50, 200, "box-w", "W2", "[1024, 256]", "+ b2")
    b += text(636, cy + 6, "=", "op-sym")
    b += mat(654, 50, 100, "box-d", "out", "[N, 256]", "N tokens")
    b += text(361, 22, "one row of Z = one token · same W1, W2 for every row", "tiny")
    return svg(724, 290, b, "The MLP as two matrix multiplies: N tokens by 256, times a 256 by 1024 matrix, ReLU, times a 1024 by 256 matrix, giving N by 256.")


def act_chart():
    import math
    W, Hh, X0, X1, Y0, Y1 = 680, 320, 60, 640, 280, 40
    lo, hi = -0.5, 3.0
    sx = lambda x: X0 + (x + 3) / 6 * (X1 - X0)
    sy = lambda y: Y0 - (y - lo) / (hi - lo) * (Y0 - Y1)
    gelu = lambda x: 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    relu = lambda x: max(0.0, x)
    b = ""
    for y in (0, 1, 2, 3):
        b += f'<line x1="{X0}" y1="{sy(y):.1f}" x2="{X1}" y2="{sy(y):.1f}" class="{"axis0" if y == 0 else "gridl"}"/>'
        b += text(X0 - 10, sy(y) + 4, str(y), "sub", "end")
    for x in range(-3, 4):
        b += f'<line x1="{sx(x):.1f}" y1="{Y1}" x2="{sx(x):.1f}" y2="{Y0}" class="{"axis0" if x == 0 else "gridl"}"/>'
        b += text(sx(x), Y0 + 18, str(x), "sub")
    b += text((X0 + X1) / 2, Y0 + 38, "input to the unit (pre-activation)", "tiny")
    pts = [i / 20 for i in range(-60, 61)]
    for f, cls in ((relu, "s-relu"), (gelu, "s-gelu")):
        p = " ".join(f"{sx(x):.1f},{sy(f(x)):.1f}" for x in pts)
        b += f'<polyline points="{p}" class="{cls}"/>'
    # legend
    b += f'<line x1="{X0+14}" y1="58" x2="{X0+40}" y2="58" class="s-relu"/>' + text(X0 + 48, 62, "ReLU: max(0, x)", "lgd", "start")
    b += f'<line x1="{X0+14}" y1="82" x2="{X0+40}" y2="82" class="s-gelu"/>' + text(X0 + 48, 86, "GELU: x · Φ(x)", "lgd", "start")
    # direct notes
    b += text(sx(-1), sy(gelu(-1)) + 22, "GELU dips to −0.17", "tiny")
    b += text(sx(-2.2), sy(0) - 10, "ReLU: flat 0, zero gradient", "tiny")
    # hover layer
    b += (f'<line id="act-x" x1="0" y1="{Y1}" x2="0" y2="{Y0}" class="xhair" visibility="hidden"/>'
          f'<circle id="act-d1" r="4.5" class="dot-relu" visibility="hidden"/>'
          f'<circle id="act-d2" r="4.5" class="dot-gelu" visibility="hidden"/>'
          f'<rect id="act-hit" x="{X0}" y="{Y1}" width="{X1-X0}" height="{Y0-Y1}" fill="transparent" '
          f'data-x0="{X0}" data-x1="{X1}" data-y0="{Y0}" data-y1="{Y1}" data-lo="{lo}" data-hi="{hi}"/>')
    return svg(W, Hh, b, "ReLU and GELU from x = −3 to 3. ReLU is exactly zero for negative inputs; GELU is smooth and dips slightly below zero, reaching about −0.17 near x = −0.75.")


def act_table():
    import math
    gelu = lambda x: 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    xs = [-2, -1, -0.5, 0, 0.5, 1, 2]
    head = "".join(f"<th>{x:g}</th>" for x in xs)
    r = "".join(f'<td class="n">{max(0, x):.2f}</td>' for x in xs)
    g = "".join(f'<td class="n">{gelu(x):.3f}</td>' for x in xs)
    return (f'<table class="act"><thead><tr><th>x</th>{head}</tr></thead><tbody>'
            f'<tr><td>ReLU</td>{r}</tr><tr><td>GELU</td>{g}</tr></tbody></table>')


HTML = Path(__file__).with_name("layers_template.html").read_text()
for k, f in dict(OVERVIEW=overview, ENC=encoder_layer, CAUSAL=causal_grid, DECK=deck_head, DECKGRID=deck_grid,
                 CAND=cand_tokens, DEC=decoder_block, PATTERNS=attn_patterns, HEAD=score_head,
                 NEURONS=mlp_neurons, MATRIX=mlp_matrix, ACT=act_chart, ACTTABLE=act_table).items():
    HTML = HTML.replace("{{" + k + "}}", f())
OUT.write_text("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" + HTML.replace("<main>", "</head>\n<body>\n<main>", 1) + "\n</body>\n</html>\n")
print(OUT, len(HTML))
