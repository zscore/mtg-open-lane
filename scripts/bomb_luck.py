"""Do bomb-heavy sets reduce the impact of skill?

Per set, one row per draft (draft_id) with at least 3 games, from the Premier Draft game data:
    wr     = the draft's game win rate
    skill  = the drafter's 17Lands overall game win rate bucket (user_game_win_rate_bucket), a long-run skill measure
    bombs  = mean copies of bombs (results/bombs/bombs_impact.csv, >= 2.5 pp per copy) in the main deck over its games

Per set:
    sd true WR    : SD of draft win rates with the binomial noise of their few games removed
    skill share   : Var(fitted wr from skill, cubic) / true Var(wr). How much of the real spread in results is skill.
    bomb share    : Var(fitted wr from bomb count, 0 / 1 / 2+ dummies) / true Var(wr)
    skill slope   : pp of draft WR per +10 pp of skill bucket
    bomb effect   : pp of draft WR for a deck with >= 1 bomb, holding skill fixed (cubic)
    bomb x skill  : change in skill slope (per 10 pp) for drafts with >= 1 bomb. Negative = bombs level the field.
    P(deck bomb)  : share of drafts whose deck has >= 1 bomb

Only users with user_n_games_bucket >= 100 are kept. The skill bucket is computed from all the user's games, these
included, so for low-volume users it is mostly this draft's own result. Output: results/bombs/bomb_luck.csv.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from openlane.games import game_path

OUT = ROOT / "results" / "bombs"
MIN_USER_GAMES = 100  # user_n_games_bucket: with >= 100 lifetime games, this draft is a small part of the skill bucket


def r2_share(X, y, w, true_var):
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    fit = X @ beta
    m = np.average(fit, weights=w)
    return np.average((fit - m) ** 2, weights=w) / true_var, beta


def run_set(s, bombs):
    lf = pl.scan_parquet(game_path(s, "PremierDraft"))
    cols = set(lf.collect_schema().names())
    bcols = [f"deck_{n}" for n in bombs if f"deck_{n}" in cols]
    nb = pl.sum_horizontal([pl.col(c).cast(pl.Float32) for c in bcols]) if bcols else pl.lit(0.0)
    d = (lf.select("draft_id", (pl.col("won").cast(pl.String) == "True").cast(pl.Float64).alias("won"),
                   pl.col("user_game_win_rate_bucket").cast(pl.Float64).alias("skill"), nb.alias("nb"),
                   pl.col("user_n_games_bucket").cast(pl.Int64).alias("ng"))
         .group_by("draft_id")
         .agg(pl.len().alias("n"), pl.col("won").mean().alias("wr"), pl.col("skill").mean(), pl.col("nb").mean(),
              pl.col("ng").min())
         .filter((pl.col("n") >= 3) & pl.col("skill").is_not_null() & (pl.col("ng") >= MIN_USER_GAMES))
         .collect())
    n, wr, sk, b = (d[c].to_numpy().astype(np.float64) for c in ["n", "wr", "skill", "nb"])
    w = np.ones_like(n)  # one vote per draft, as in the bombs notebook's sd true WR
    obs_var = wr.var()
    noise = np.mean(wr * (1 - wr) / (n - 1))  # unbiased per-draft binomial variance
    true_var = obs_var - noise
    z = (sk - sk.mean()) * 10  # skill in units of 10 pp
    one = np.ones_like(z)
    S = np.column_stack([one, z, z ** 2, z ** 3])
    skill_share, _ = r2_share(S, wr, w, true_var)
    has = (b >= 0.5).astype(float)
    two = (b >= 1.5).astype(float)
    bomb_share, _ = r2_share(np.column_stack([one, has, two]), wr, w, true_var)
    _, beta_lin = r2_share(np.column_stack([one, z]), wr, w, true_var)
    _, beta_b = r2_share(np.column_stack([S, has]), wr, w, true_var)
    _, beta_x = r2_share(np.column_stack([one, z, has, z * has]), wr, w, true_var)
    return {"set": s, "drafts": len(wr), "sd true WR": 100 * np.sqrt(true_var),
            "skill share": skill_share, "bomb share": bomb_share,
            "skill slope": 100 * beta_lin[1], "bomb effect": 100 * beta_b[4], "bomb x skill": 100 * beta_x[3],
            "P(deck bomb)": np.average(has, weights=w)}


def plot(d):
    import matplotlib.pyplot as plt
    BLUE, ORANGE, MUTED, INK, GRID = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#e6e5e0"
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "legend.frameon": False,
    })
    nb = pl.read_csv(OUT / "bombs_impact.csv").group_by("set").agg(pl.len().alias("bombs"))
    d = d.join(nb, on="set", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(d["bombs"], d["skill share"], color=BLUE, s=45, zorder=3)
    for s, x, y in zip(d["set"], d["bombs"], d["skill share"]):
        ax.annotate(s, (x, y), xytext=(4, -10 if s == "TMT" else 3), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("bombs in the set (cards ≥ 2.5 pp of game win rate)")
    ax.set_ylabel("share of real spread in draft results explained by skill")
    ax.set_title("Bomb-heavy sets don't reward skill less", loc="left")
    ax = axes[1]
    x = np.arange(d.height)
    ax.bar(x - 0.2, d["skill share"], width=0.4, color=BLUE, label="drafter skill (17Lands win-rate bracket)")
    ax.bar(x + 0.2, d["bomb share"], width=0.4, color=ORANGE, label="bombs in the deck")
    ax.set_xticks(x, d["set"], rotation=60)
    ax.set_ylabel("share of real spread in draft results")
    ax.set_title("Skill explains far more than bombs, in every set", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "bomb_luck.png", dpi=110)
    fig.savefig(ROOT / "blog" / "bombs_img" / "skill_vs_bombs.png", dpi=110)


if __name__ == "__main__":
    bi = pl.read_csv(OUT / "bombs_impact.csv")
    sets = bi["set"].unique(maintain_order=True).to_list()
    order = pl.read_csv(OUT / "deck_score_summary_gih.csv")["set"].to_list()
    rows = []
    for s in order:
        rows.append(run_set(s, bi.filter(pl.col("set") == s)["name"].to_list()))
        r = rows[-1]
        print(f"{s}: skill share {r['skill share']:.2f}, bomb share {r['bomb share']:.3f}, "
              f"bomb effect {r['bomb effect']:+.1f}, bomb x skill {r['bomb x skill']:+.2f}", flush=True)
    pl.DataFrame(rows).write_csv(OUT / "bomb_luck.csv")
    plot(pl.DataFrame(rows))
