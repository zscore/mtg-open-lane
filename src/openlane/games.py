"""17Lands game + replay data: per-game outcomes, decks, and the cards each side cast.

Game data has the user's full deck and the cards they drew, but only the opponent's colours.
Replay data has, turn by turn, every card either player cast (as Arena ids). Cards cast are
the only view of the two decks that is recorded the same way for both players.
"""
from __future__ import annotations

import gzip
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from .data import PARQUET, RAW, S3, download_cards

MAX_TURNS = 30
META = ["draft_id", "draft_time", "game_time", "build_index", "match_number", "game_number", "rank", "opp_rank",
        "main_colors", "splash_colors", "on_play", "num_mulligans", "opp_num_mulligans",
        "opp_colors", "num_turns", "won", "user_n_games_bucket", "user_game_win_rate_bucket"]
# (side, column) pairs whose values are "|"-joined ids of spells that side cast. The turn prefix
# is whose turn it is; `oppo_instants_sorceries_cast` on the user's turn is the opponent's.
USER_CAST = [("user", "creatures_cast"), ("user", "non_creatures_cast"),
             ("user", "user_instants_sorceries_cast"), ("oppo", "user_instants_sorceries_cast")]
OPPO_CAST = [("oppo", "creatures_cast"), ("oppo", "non_creatures_cast"),
             ("oppo", "oppo_instants_sorceries_cast"), ("user", "oppo_instants_sorceries_cast")]


def _fetch(kind: str, set_code: str, event: str) -> Path:
    gz = RAW / f"{kind}_data_public.{set_code}.{event}.csv.gz"
    if not gz.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{S3}/{kind}_data/{gz.name}", gz)
    return gz


def _slim_parquet(gz: Path, out: Path, keep) -> None:
    """Decompress to disk and stream only the columns `keep(name)` accepts into parquet."""
    tmp = out.with_suffix(".csv.tmp")
    with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 24)
    try:
        header = pl.read_csv(tmp, n_rows=0, infer_schema=False).columns
        cols = [c for c in header if keep(c)]
        # card counts are read as floats because some files (LCI) write them as "0.0"
        counts = {c for c in header
                  if c.startswith(("deck_", "sideboard_", "drawn_", "opening_hand_", "tutored_"))}
        schema = {c: pl.Float32 if c in counts else pl.Utf8 for c in header}
        pl.scan_csv(tmp, schema_overrides=schema).select(
            [pl.col(c).cast(pl.Int8) if c in counts else pl.col(c) for c in cols]).sink_parquet(out)
    finally:
        tmp.unlink()


def replay_path(set_code: str, event: str = "PremierDraft") -> Path:
    out = PARQUET / f"{set_code}.{event}.replay.parquet"
    if not out.exists():
        wanted = {f"{s}_turn_{{}}_{c}" for s, c in USER_CAST + OPPO_CAST}
        pat = re.compile(r"(user|oppo)_turn_(\d+)_(.+)")

        def keep(c):
            m = pat.fullmatch(c)
            return c in META or (m is not None and f"{m[1]}_turn_{{}}_{m[3]}" in wanted)
        _slim_parquet(_fetch("replay", set_code, event), out, keep)
    return out


def game_path(set_code: str, event: str = "PremierDraft") -> Path:
    out = PARQUET / f"{set_code}.{event}.game.parquet"
    if not out.exists():
        _slim_parquet(_fetch("game", set_code, event), out,
                      lambda c: c in META or c.startswith(("deck_", "drawn_", "opening_hand_")))
    return out


def id_to_index(names: list[str]) -> dict[int, int]:
    """Arena id -> index into `names` (every printing of a name maps to the same index)."""
    cards = pl.read_csv(download_cards(), columns=["id", "name"])
    idx = {n: i for i, n in enumerate(names)}
    return {int(i): idx[n] for i, n in cards.iter_rows() if n in idx}


def cast_matrix(replay: pl.DataFrame, names: list[str], cutoffs=(MAX_TURNS,)):
    """Counts of each card cast by the user (U) and opponent (O) in each game.

    A cutoff counts turns per player (3 = through each player's third turn).
    Returns ({cutoff: (U, O)}, unmapped_frac) with [games, cards] int8 arrays.
    """
    lookup = id_to_index(names)
    U = np.zeros((replay.height, len(names)), dtype=np.int8)
    O = np.zeros_like(U)
    out, unmapped, total = {}, 0, 0
    for t in range(1, max(cutoffs) + 1):
        for M, sides in ((U, USER_CAST), (O, OPPO_CAST)):
            for side, col in sides:
                name = f"{side}_turn_{t}_{col}"
                if name not in replay.columns:
                    continue
                s = replay[name]
                rows = np.flatnonzero(s.is_not_null().to_numpy())
                for r, v in zip(rows, s.gather(rows).to_list()):
                    for tok in v.split("|"):
                        total += 1
                        j = lookup.get(int(tok))
                        if j is None:
                            unmapped += 1
                        else:
                            M[r, j] += 1
        if t in cutoffs:
            out[t] = (U.copy(), O.copy())
    return out, unmapped / max(total, 1)
