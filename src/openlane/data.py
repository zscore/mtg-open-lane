"""Download 17Lands public draft data and load it as dense per-draft tensors."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PARQUET = ROOT / "data" / "parquet"
S3 = "https://17lands-public.s3.amazonaws.com/analysis_data"

N_PACKS, N_PICKS, N_SEATS = 3, 14, 8
COLORS = "WUBRG"
RARITIES = ["common", "uncommon", "rare", "mythic", "basic", "special"]


def download(set_code: str, event: str = "PremierDraft") -> Path:
    """Fetch a draft file (cached) and convert it to parquet."""
    RAW.mkdir(parents=True, exist_ok=True)
    PARQUET.mkdir(parents=True, exist_ok=True)
    out = PARQUET / f"{set_code}.{event}.parquet"
    if out.exists():
        return out
    gz = RAW / f"draft_data_public.{set_code}.{event}.csv.gz"
    if not gz.exists():
        urllib.request.urlretrieve(f"{S3}/draft_data/{gz.name}", gz)
    to_parquet(gz, out)
    return out


def to_parquet(gz: Path, out: Path) -> None:
    """Stream a 17Lands .csv.gz to parquet with Int8 card counts.

    Decompressed files run to 10GB+ and the ~700 card columns default to Int64, so decompress to
    a temp file on disk and let polars stream it instead of holding it in RAM.
    """
    import gzip
    import shutil

    tmp = out.with_suffix(".csv.tmp")
    with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 24)
    try:
        header = pl.read_csv(tmp, n_rows=0).columns
        small = {c: pl.Int8 for c in header
                 if c.startswith(("pack_card_", "pool_")) or c in ("pack_number", "pick_number")}
        pl.scan_csv(tmp, schema_overrides=small, infer_schema_length=10000).sink_parquet(out)
    finally:
        tmp.unlink()


def download_cards() -> Path:
    path = RAW / "cards.csv"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{S3}/cards/cards.csv", path)
    return path


def card_metadata(names: list[str], set_code: str) -> pl.DataFrame:
    """Colors / rarity / mana value for each card column, preferring the set's own printing.

    Bonus-sheet cards live under other expansion codes, so fall back to any printing by name.
    """
    cards = pl.read_csv(download_cards(), schema_overrides={"color_identity": pl.Utf8})
    cards = cards.with_columns((pl.col("expansion") == set_code).alias("_own"))
    cards = cards.sort("_own", descending=True).unique("name", keep="first")
    meta = pl.DataFrame({"name": names}).join(cards, on="name", how="left")
    ci = meta["color_identity"].fill_null("")
    out = meta.select(
        "name",
        pl.col("rarity").fill_null("special"),
        pl.col("mana_value").fill_null(0).cast(pl.Float32),
        pl.col("types").fill_null("").str.contains("Creature").alias("is_creature"),
        (pl.col("expansion") != set_code).fill_null(True).alias("is_bonus"),
    )
    return out.with_columns([ci.str.contains(c).alias(c) for c in COLORS])


@dataclass
class DraftTensors:
    """Dense arrays for complete drafts, sorted by draft_time.

    pack[d, k, p, c]: copies of card c in the pack shown at pack k, pick p.
    pool[d, k, p, c]: copies of card c already taken before that pick.
    pick[d, k, p]:    index of the card taken.
    """

    set_code: str
    names: list[str]
    meta: pl.DataFrame
    draft_id: np.ndarray
    draft_time: np.ndarray
    rank: np.ndarray
    user_wr: np.ndarray
    user_ngames: np.ndarray
    pack: np.ndarray
    pool: np.ndarray
    pick: np.ndarray

    @property
    def color_matrix(self) -> np.ndarray:
        return self.meta.select(list(COLORS)).to_numpy().astype(np.float32)


RANK_ORDER = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3, "diamond": 4, "mythic": 5}


def load_tensors(set_code: str, event: str = "PremierDraft") -> DraftTensors:
    df = pl.read_parquet(download(set_code, event))
    pack_cols = [c for c in df.columns if c.startswith("pack_card_")]
    names = [c[len("pack_card_"):] for c in pack_cols]
    pool_cols = [f"pool_{n}" for n in names]

    full = df.group_by("draft_id").len().filter(pl.col("len") == N_PACKS * N_PICKS)["draft_id"]
    df = df.filter(pl.col("draft_id").is_in(full.implode()))
    df = df.sort(["draft_time", "draft_id", "pack_number", "pick_number"])
    D, C = df.height // (N_PACKS * N_PICKS), len(names)

    shape = (D, N_PACKS, N_PICKS, C)
    pack = df.select(pack_cols).to_numpy().astype(np.int8).reshape(shape)
    pool = df.select(pool_cols).to_numpy().astype(np.int8).reshape(shape)
    idx = {n: i for i, n in enumerate(names)}
    pick = np.array([idx[n] for n in df["pick"].to_list()], dtype=np.int16).reshape(shape[:3])

    per_draft = df.gather_every(N_PACKS * N_PICKS)
    return DraftTensors(
        set_code=set_code,
        names=names,
        meta=card_metadata(names, set_code),
        draft_id=per_draft["draft_id"].to_numpy(),
        draft_time=per_draft["draft_time"].to_numpy(),
        rank=per_draft["rank"].replace_strict(RANK_ORDER, default=None).fill_null(-1).to_numpy(),
        user_wr=per_draft["user_game_win_rate_bucket"].fill_null(-1).cast(pl.Float32).to_numpy(),
        user_ngames=per_draft["user_n_games_bucket"].fill_null(-1).cast(pl.Float32).to_numpy(),
        pack=pack,
        pool=pool,
        pick=pick,
    )
