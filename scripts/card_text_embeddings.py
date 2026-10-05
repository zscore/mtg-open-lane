"""Frozen sentence embeddings of each card's type line and rules text (as in DraftFM), per set.

    python scripts/card_text_embeddings.py MSH FDN ...   (needs sentence-transformers; any env)

The card's own name is replaced by "CARDNAME" so the text can't act as an identifier. All faces are
included. Writes data/features/{SET}.PremierDraft/text_emb.npy, [C, 384] in the draft file's
card-column order, L2-normalised (BAAI/bge-small-en-v1.5).
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.cardinfo import _scryfall  # noqa: E402
from openlane.data import PARQUET, ROOT  # noqa: E402


def card_text(c: dict) -> str:
    faces = c.get("card_faces") or [c]
    out = []
    for f in faces:
        t = f"{f.get('type_line', '')}. {f.get('oracle_text', '')}"
        for n in {c["name"], *(g.get("name", "") for g in faces)} - {""}:
            t = t.replace(n, "CARDNAME")
        out.append(t)
    return " // ".join(out)


def main(*sets):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    for s in sets:
        cols = pl.scan_parquet(PARQUET / f"{s}.PremierDraft.parquet").collect_schema().names()
        names = [c[len("pack_card_"):] for c in cols if c.startswith("pack_card_")]
        sc = _scryfall(names, s)
        texts = [card_text(sc[n]) if n in sc else n for n in names]
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        d = ROOT / "data" / "features" / f"{s}.PremierDraft"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "text_emb.npy", emb.astype(np.float32))
        print(s, emb.shape, "missing", sum(n not in sc for n in names), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
