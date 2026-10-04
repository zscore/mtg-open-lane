"""Card facts from Scryfall oracle data (mana costs, mana produced, keywords, rules text), and deck
features built from them.

`card_info` gives one row per card column. `card_matrix` turns it into per-card model inputs (for the
transformer's token metadata). `deck_features` gives per-deck numbers that are *not* sums of per-card
values (a linear model over card counts can already learn any sum over cards): land-count buckets, the
squares of curve counts, number of colours, and how well the lands cast the spells.
"""
from __future__ import annotations

import gzip
import json
import re

import numpy as np
import polars as pl

from .data import COLORS, RAW

SKIP_LAYOUTS = {"art_series", "token", "double_faced_token", "emblem", "vanguard", "scheme", "planar"}
KEYWORDS = ("Flying", "Vigilance", "Flash", "Reach", "Trample", "Lifelink", "Haste", "Deathtouch", "Ward",
            "Prowess", "First strike", "Double strike", "Menace", "Indestructible", "Defender", "Hexproof",
            "Scry", "Surveil", "Cycling", "Landcycling", "Treasure", "Food", "Equip", "Enchant", "Crew",
            "Transform", "Fight", "Mill", "Investigate", "Landfall")
# rough rules-text roles; matched case-insensitively against the oracle text of all faces
TEXT_FLAGS = {
    "removal": r"(destroy|exile) target (creature|nonland|permanent|artifact|enchantment|planeswalker)"
               r"|deals? (\d+|x|that much) damage to (target|any target|each creature)|fights? |gets? -\d+/-\d+"
               r"|target creature .{0,40}(can't attack|can't block|doesn't untap)",
    "draw": r"draws? (a|two|three|x|that many) cards?",
    "counter": r"counter target",
    "tokens": r"create[s]? .{0,40}token",
    "pump": r"gets? \+\d+/\+\d+|\+1/\+1 counter",
    "lifegain": r"gain \d+ life|lifelink",
    "fixing": r"add (one mana of any color|\{[wubrg]\} or \{[wubrg]\})|treasure|search your library for a basic land",
    "evasion": r"can't be blocked|flying|menace|trample",
}
SOURCE_NEED = {1: 7.0, 2: 10.0, 3: 12.0}  # coloured sources a 40-card deck wants for 1/2/3 pips of a colour


def _scryfall(names: list[str], set_code: str) -> dict[str, dict]:
    want, best = set(names), {}
    with gzip.open(RAW / "scryfall_oracle_cards.jsonl.gz", "rt") as f:
        for line in f:
            c = json.loads(line)
            if c.get("layout") in SKIP_LAYOUTS:
                continue
            keys = {c["name"]} | ({c["card_faces"][0]["name"]} if c.get("card_faces") else set())
            for k in keys & want:
                score = (c.get("set") == set_code.lower(), "arena" in c.get("games", []))
                if k not in best or score > best[k][0]:
                    best[k] = (score, c)
    return {k: c for k, (_, c) in best.items()}


def _pips(cost: str) -> dict[str, float]:
    out = dict.fromkeys(COLORS, 0.0)
    generic = 0.0
    for sym in re.findall(r"\{([^}]+)\}", cost or ""):
        parts = [p for p in sym.split("/") if p in COLORS]
        if parts:
            for p in parts:  # hybrid {W/U} counts half to each; phyrexian {W/P} counts as W
                out[p] += 1 / len(parts)
        elif sym.isdigit():
            generic += int(sym)
    return {**out, "generic": generic}


def card_info(names: list[str], set_code: str) -> pl.DataFrame:
    """One row per card: pips per colour, generic cost, mana value, type flags, colours produced,
    power/toughness, keywords and rules-text roles. The front face is used for costs and types."""
    sc = _scryfall(names, set_code)
    missing = [n for n in names if n not in sc]
    assert not missing, f"no Scryfall data for {missing[:5]}"
    rows = []
    for n in names:
        c = sc[n]
        front = c["card_faces"][0] if c.get("card_faces") else c
        text = " ".join(f.get("oracle_text", "") for f in c.get("card_faces", [c])).lower()
        types = front.get("type_line", "")
        is_land = "Land" in types
        pips = _pips(front.get("mana_cost", ""))
        produced = set(c.get("produced_mana") or []) & set(COLORS)
        pt = [front.get("power"), front.get("toughness")]
        num = lambda v: float(v) if v and v.replace(".", "").isdigit() else 0.0  # noqa: E731
        kws = set(c.get("keywords", []))
        rows.append({
            "name": n, **{f"pip_{k}": pips[k] for k in COLORS}, "generic": pips["generic"],
            "mv": float(c.get("cmc", 0.0)) if not c.get("card_faces") else float(sum(_pips(front.get("mana_cost", "")).values())),
            "is_land": is_land, "is_basic": "Basic" in types, "is_creature": "Creature" in types,
            **{f"makes_{k}": (k in produced) and is_land for k in COLORS},
            "nonland_source": bool(produced) and not is_land,
            "power": num(pt[0]), "toughness": num(pt[1]),
            **{f"kw_{k}": k in kws for k in KEYWORDS},
            **{f"text_{k}": bool(re.search(p, text)) for k, p in TEXT_FLAGS.items()},
        })
    return pl.DataFrame(rows)


def card_matrix(info: pl.DataFrame) -> np.ndarray:
    """[C, F] per-card numeric inputs (pips, cost, mana value, types, colours produced, P/T, keywords,
    text roles), scaled to roughly [0, 1]."""
    x = info.drop("name").with_columns(pl.col("generic", "mv") / 8, pl.col("power", "toughness").clip(0, 8) / 8,
                                       *[pl.col(f"pip_{k}") / 3 for k in COLORS])
    return x.cast(pl.Float32).to_numpy()


DECK_FEATURES = ("lands_le15", "lands_16", "lands_18", "lands_ge19", "n_colours_1", "n_colours_3plus", "splash",
                 "curve_sq", "mv_mean", "mv_mean_sq", "creatures_sq", "removal_sq", "two_drops_short",
                 "source_shortfall", "worst_colour_gap", "pip_source_mismatch", "unused_sources")


def deck_features(X: np.ndarray, info: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """[N, F] nonlinear deck features from [N, C] card counts. Linear functions of the counts (how many
    2-drops, how many creatures) are left out on purpose: the card weights already cover them."""
    X = X.astype(np.float32)
    land = info["is_land"].to_numpy()
    nonland = ~land
    mv = info["mv"].to_numpy()
    pips = info.select([f"pip_{k}" for k in COLORS]).to_numpy().astype(np.float32)
    makes = info.select([f"makes_{k}" for k in COLORS]).to_numpy().astype(np.float32)
    src = X @ makes + 0.5 * (X @ (info["nonland_source"].to_numpy()[:, None] * np.ones((1, 5), np.float32)))
    n_land = X @ land.astype(np.float32)
    n_spell = X @ nonland.astype(np.float32)
    # curve: counts of nonland cards by mana value 1, 2, ..., 6+
    buckets = np.stack([nonland & (np.clip(mv, 1, 6) == b) for b in range(1, 7)], 1).astype(np.float32)
    curve = X @ buckets
    curve_c = curve - curve.mean(0)
    mv_mean = (X @ (mv * nonland)) / np.maximum(n_spell, 1)
    creatures = X @ info["is_creature"].to_numpy().astype(np.float32)
    removal = X @ info["text_removal"].to_numpy().astype(np.float32)
    # colours: a colour is "main" with 4+ coloured spells, "splash" with 1-3
    uses = X @ (pips[:, :] > 0).astype(np.float32)
    main = uses >= 4
    splash = (uses > 0) & ~main
    # castability: for each spell and colour it needs, sources wanted (by pip count) minus sources held
    need = np.zeros_like(pips)
    for p, s in SOURCE_NEED.items():
        need[np.ceil(pips) >= p] = s
    gap = np.zeros(len(X), np.float32)
    worst = np.zeros(len(X), np.float32)
    for k in range(5):
        has = need[:, k] > 0
        if not has.any():
            continue
        g = np.maximum(0, need[None, has, k] - src[:, k:k + 1])  # [N, cards needing k]
        gap += ((X[:, has] > 0) * g * X[:, has]).sum(1)
        worst = np.maximum(worst, ((X[:, has] > 0) * g).max(1))
    pip_share = (X @ pips) / np.maximum((X @ pips).sum(1, keepdims=True), 1e-6)
    src_share = src / np.maximum(src.sum(1, keepdims=True), 1e-6)
    unused = (src * (uses == 0)).sum(1)
    F = np.stack([n_land <= 15, n_land == 16, n_land == 18, n_land >= 19,
                  main.sum(1) <= 1, main.sum(1) >= 3, splash.sum(1) > 0,
                  (curve_c ** 2).sum(1), mv_mean, (mv_mean - mv_mean.mean()) ** 2,
                  (creatures - creatures.mean()) ** 2, (removal - removal.mean()) ** 2,
                  np.maximum(0, 4 - curve[:, 1]),
                  gap / np.maximum(n_spell, 1), worst, np.abs(pip_share - src_share).sum(1), unused], 1)
    return F.astype(np.float32), list(DECK_FEATURES)
