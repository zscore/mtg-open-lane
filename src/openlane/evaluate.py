"""Metrics shared by every model."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        # expected calibration error over 20 equal-width bins
        "ece": float(_ece(y, p)),
    }


def _ece(y, p, bins=20):
    b = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(b, minlength=bins)
    conf = np.bincount(b, weights=p, minlength=bins)
    acc = np.bincount(b, weights=y, minlength=bins)
    m = n > 0
    return np.sum(np.abs(acc[m] - conf[m])) / len(y)


def metrics_by(df: pl.DataFrame, pred: str, by: list[str]) -> pl.DataFrame:
    rows = []
    for key, g in df.group_by(by, maintain_order=True):
        rows.append({**dict(zip(by, key)), "n": g.height,
                     **metrics(g["wheeled"].to_numpy(), g[pred].to_numpy())})
    return pl.DataFrame(rows).sort(by)
