"""Train baselines + LightGBM ablations: python scripts/train.py MSH [event] [key=value ...]

max_train=N fits on the N most recent training drafts (default 60000, so the big sets fit in
16GB; card stats still use every training draft). quick=1 skips the three ablation GBMs.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane import ROOT  # noqa: E402
from mtgdata.data import FEATURES  # noqa: E402
from openlane.evaluate import metrics, metrics_by  # noqa: E402
from openlane.features import CARD_FEATURES, PACK_FEATURES, SIGNAL_FEATURES  # noqa: E402

LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=10)


def alsa_design(df: pl.DataFrame, n_wheel: int) -> np.ndarray:
    """ALSA/ATA-only features with a separate slope per pick (the plan's baseline)."""
    p = df["pick_number"].to_numpy()
    onehot = np.eye(n_wheel, dtype=np.float32)[p]
    a, t = df["alsa"].to_numpy()[:, None], df["ata"].to_numpy()[:, None]
    return np.hstack([onehot, onehot * a, onehot * t, onehot * a ** 2, onehot * t ** 2])


def main(set_code: str, event: str = "PremierDraft", *kv: str):
    opts = {"max_train": 60000, "quick": 0} | {k: int(v) for k, v in (a.split("=") for a in kv)}
    feat_dir = FEATURES / f"{set_code}.{event}"
    out = ROOT / "results" / f"{set_code}.{event}"
    (out / "models").mkdir(parents=True, exist_ok=True)
    train_drafts = pickle.load(open(feat_dir / "meta.pkl", "rb"))["splits"]["train"]
    first = int(train_drafts[max(0, len(train_drafts) - opts["max_train"])])
    train = pl.scan_parquet(feat_dir / "train.parquet").filter(pl.col("draft_idx") >= first).collect()
    print(f"GBM/baselines fit on {len(train_drafts) - (first - int(train_drafts[0])):,} of "
          f"{len(train_drafts):,} training drafts", flush=True)
    val = pl.read_parquet(feat_dir / "val.parquet")
    test = pl.read_parquet(feat_dir / "test.parquet")
    ytr, yva = train["wheeled"].to_numpy(), val["wheeled"].to_numpy()
    preds, results = {}, {}

    def record(name, p_test, extra=None):
        preds[name] = p_test.astype(np.float32)
        results[name] = {**metrics(test["wheeled"].to_numpy(), p_test), **(extra or {})}
        print(f"{name:28s} " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()
                                          if isinstance(v, float)), flush=True)

    # 1. prior by pick number only
    prior = train.group_by("pick_number").agg(pl.col("wheeled").mean().alias("r"))
    record("prior_by_pick", test.join(prior, on="pick_number", how="left")["r"].to_numpy())

    # 2. ALSA/ATA logistic regression (card-level aggregate, no identity)
    lr = LogisticRegression(max_iter=500)
    sub = np.random.default_rng(0).choice(train.height, min(train.height, 2_000_000), replace=False)
    n_wheel = int(train["pick_number"].max()) + 1
    lr.fit(alsa_design(train[sub], n_wheel), ytr[sub])
    record("alsa_ata_logreg", lr.predict_proba(alsa_design(test, n_wheel))[:, 1])

    # 3. per-card x pick empirical wheel rate (strongest pure card-identity baseline)
    record("card_x_pick_rate", test["wheel_cp"].to_numpy())

    # 4-6. LightGBM ablations
    groups = {
        "lgbm_card": CARD_FEATURES,
        "lgbm_card_pack": CARD_FEATURES + PACK_FEATURES,
        "lgbm_card_signals": CARD_FEATURES + SIGNAL_FEATURES,
        "lgbm_all": CARD_FEATURES + PACK_FEATURES + SIGNAL_FEATURES,
    }
    if opts["quick"]:
        groups = {"lgbm_all": groups["lgbm_all"]}
    importances = {}
    for name, feats in groups.items():
        t0 = time.time()
        dtr = lgb.Dataset(train.select(feats).to_numpy(), ytr, feature_name=feats,
                          categorical_feature=["card"], free_raw_data=True)
        dva = lgb.Dataset(val.select(feats).to_numpy(), yva, reference=dtr)
        model = lgb.train(LGB_PARAMS, dtr, 3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        model.save_model(str(out / "models" / f"{name}.txt"))
        record(name, model.predict(test.select(feats).to_numpy()),
               {"best_iter": model.best_iteration, "train_s": round(time.time() - t0)})
        imp = model.feature_importance("gain")
        importances[name] = dict(sorted(zip(feats, (imp / imp.sum()).round(4).tolist()),
                                        key=lambda kv: -kv[1]))

    # per-slice breakdowns for the headline comparison
    test = test.with_columns([pl.Series(k, v) for k, v in preds.items()])
    test.select(["draft_idx", "pack_number", "pick_number", "card", "wheeled", *preds]) \
        .write_parquet(out / "test_predictions.parquet")
    slices = {}
    for m in ["card_x_pick_rate", "lgbm_card_pack", "lgbm_all"] if not opts["quick"] else ["lgbm_all"]:
        slices[m] = metrics_by(test, m, ["pack_number", "pick_number"]).to_dicts()

    json.dump({"test": results, "importance": importances, "slices": slices},
              open(out / "results.json", "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
