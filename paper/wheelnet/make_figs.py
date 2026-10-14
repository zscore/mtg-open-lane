"""Figures, tables and numbers for paper/wheelnet/wheelnet.tex, from saved results.

    .venv/bin/python paper/wheelnet/make_figs.py

Reads results/<SET>.PremierDraft/{summary,wheel_breakdown,open_lane,force_vs_follow,temporal}.json and, for the
calibration figure, MSH's test predictions plus the wheel feature cache in mtg-data.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "lines.linewidth": 1.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
load = lambda p: json.loads(p.read_text())  # noqa: E731
nums = {}

# ---------------------------------------------------------------- per-set wheel results
sets = sorted(p.parent.name.split(".")[0] for p in R.glob("*.PremierDraft/summary.json")
              if "wheelnet" in {r["model"] for r in load(p)})
rows = {}
for s in sets:
    d = R / f"{s}.PremierDraft"
    summ = {r["model"]: r for r in load(d / "summary.json")}
    bd = load(d / "wheel_breakdown.json")
    n_picks = int(re.search(r"'n_picks': (\d+)", (d / "run.log").read_text()).group(1)) if s != "MSH" else 14
    gain = lambda r, m: (r["alsa_ata_logreg"]["logloss"] - r[m]["logloss"]) / r["alsa_ata_logreg"]["logloss"]  # noqa: E731
    rows[s] = dict(
        drafts=load(d / "open_lane.json")["drafts"], pack=n_picks, n=bd["all"]["n"], base=bd["all"]["base_rate"],
        alsa=summ["alsa_ata_logreg"]["logloss"], gbm=summ["lgbm_all"]["logloss"], wn=summ["wheelnet"]["logloss"],
        auc_alsa=summ["alsa_ata_logreg"]["auc"], auc_wn=summ["wheelnet"]["auc"],
        g_gbm=summ["lgbm_all"]["vs_alsa"], g_wn=summ["wheelnet"]["vs_alsa"],
        rank_gain=[gain(bd[k], "wheelnet") for k in ("rank1", "rank2", "rank3", "rank4", "rank5", "rank6+")],
        ece_alsa=float(np.mean([bd[f"rank{k}"]["alsa_ata_logreg"]["ece"] for k in (1, 2, 3)])),
        ece_wn=float(np.mean([bd[f"rank{k}"]["wheelnet"]["ece"] for k in (1, 2, 3)])),
    )
nums["wheel_sets"] = len(sets)
nums["test_examples"] = int(sum(r["n"] for r in rows.values()))
nums["drafts_wheel"] = int(sum(r["drafts"] for r in rows.values()))
for k in ("g_gbm", "g_wn"):
    v = [rows[s][k] for s in sets]
    nums[f"{k}_mean"], nums[f"{k}_min"], nums[f"{k}_max"] = float(np.mean(v)), float(min(v)), float(max(v))
nums["wn_beats_gbm"] = int(sum(rows[s]["g_wn"] > rows[s]["g_gbm"] for s in sets))
rg = np.array([rows[s]["rank_gain"] for s in sets])
nums["rank_gain_mean"] = rg.mean(0).tolist()
nums["rank1_gain_range"] = [float(rg[:, 0].min()), float(rg[:, 0].max())]
nums["ece_top3"] = {s: (rows[s]["ece_alsa"], rows[s]["ece_wn"]) for s in sets}
nums["ece_ratio_range"] = [float(min(rows[s]["ece_alsa"] / rows[s]["ece_wn"] for s in sets)),
                           float(max(rows[s]["ece_alsa"] / rows[s]["ece_wn"] for s in sets))]

body = []
for s in sets:
    r = rows[s]
    body.append(f"{s} & {r['drafts'] / 1e3:.0f} & {r['pack']} & {r['n'] / 1e6:.2f} & {r['base']:.3f} & "
                f"{r['alsa']:.4f} & {r['gbm']:.4f} & \\textbf{{{r['wn']:.4f}}} & "
                f"+{100 * r['g_gbm']:.1f} & \\textbf{{+{100 * r['g_wn']:.1f}}} & "
                f"{r['auc_alsa']:.3f} & \\textbf{{{r['auc_wn']:.3f}}} & +{100 * r['rank_gain'][0]:.1f} & "
                f"{r['ece_alsa']:.3f} & \\textbf{{{r['ece_wn']:.3f}}} \\\\")
body.append("\\midrule")
mean = lambda k: float(np.mean([rows[s][k] for s in sets]))  # noqa: E731
body.append(f"mean & & & & {mean('base'):.3f} & {mean('alsa'):.4f} & {mean('gbm'):.4f} & \\textbf{{{mean('wn'):.4f}}} & "
            f"+{100 * mean('g_gbm'):.1f} & \\textbf{{+{100 * mean('g_wn'):.1f}}} & {mean('auc_alsa'):.3f} & "
            f"\\textbf{{{mean('auc_wn'):.3f}}} & +{100 * rg[:, 0].mean():.1f} & {mean('ece_alsa'):.3f} & "
            f"\\textbf{{{mean('ece_wn'):.3f}}} \\\\")
(OUT / "table_sets.tex").write_text(r"""\begin{tabular}{lrrrc ccc cc cc c cc}
\toprule
 & drafts & pack & test ex. & wheel & \multicolumn{3}{c}{log loss $\downarrow$} & \multicolumn{2}{c}{vs ALSA (\%) $\uparrow$} & \multicolumn{2}{c}{AUC $\uparrow$} & best card & \multicolumn{2}{c}{ECE, top 3 $\downarrow$} \\
\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-13}\cmidrule(lr){14-15}
set & (k) & size & (M) & rate & ALSA & GBM & WheelNet & GBM & WheelNet & ALSA & WheelNet & vs ALSA (\%) & ALSA & WheelNet \\
\midrule
""" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Fig: per-set gain over ALSA
fig, ax = plt.subplots(figsize=(3.4, 2.6))
order = sorted(sets, key=lambda s: rows[s]["g_wn"])
y = np.arange(len(order))
for i, s in enumerate(order):
    ax.plot([100 * rows[s]["g_gbm"], 100 * rows[s]["g_wn"]], [i, i], color="#d4d3cd", lw=1, zorder=1)
ax.scatter([100 * rows[s]["g_gbm"] for s in order], y, s=16, color=AQUA, label="LightGBM, hand-built features", zorder=2)
ax.scatter([100 * rows[s]["g_wn"] for s in order], y, s=18, color=BLUE, label="WheelNet, raw history", zorder=3)
ax.set_yticks(y, order)
ax.set_xlabel("log-loss reduction vs the ALSA/ATA baseline (%, test drafts)")
ax.set_xlim(4, 12.5)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "per_set.pdf")

# ---- Fig: gain by rank of the passed card within its pack
fig, ax = plt.subplots(figsize=(3.4, 2.1))
x = np.arange(1, 7)
for i, s in enumerate(sets):
    ax.plot(x, 100 * rg[i], color="#d4d3cd", lw=0.8, zorder=1)
ax.plot(x, 100 * rg.mean(0), color=BLUE, marker="o", ms=3, label=f"mean of {len(sets)} sets", zorder=3)
ax.plot(x, 100 * rg[sets.index("MSH")], color=ORANGE, lw=1.0, ls="--", label="MSH", zorder=2)
ax.set_xticks(x, ["1\n(best)", "2", "3", "4", "5", "6+"])
ax.set_xlabel("rank of the passed card in its pack, by training ATA")
ax.set_ylabel("WheelNet vs ALSA (%)")
ax.legend(fontsize=6.5, loc="upper right")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "by_rank.pdf")

# ---- Fig: calibration of the second-best passed card, MSH
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from wheel_breakdown import load as load_bd  # noqa: E402
bd, _ = load_bd("MSH", ["wheelnet"])
r2 = bd.filter(pl.col("rank_in_pack") == 2)
edges = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
fig, ax = plt.subplots(figsize=(3.0, 2.6))
ax.plot([0, 0.6], [0, 0.6], color="#c8c7c1", lw=0.8, ls=":")
cal = {}
for m, lab, c in [("alsa_ata_logreg", "ALSA/ATA baseline", GRAY), ("wheelnet", "WheelNet", BLUE)]:
    p, yv = r2[m].to_numpy(), r2["wheeled"].to_numpy()
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    pts = [(p[b == i].mean(), yv[b == i].mean(), (b == i).mean()) for i in range(len(edges) - 1) if (b == i).sum() >= 200]
    cal[m] = pts
    ax.plot([a for a, _, _ in pts], [o for _, o, _ in pts], color=c, marker="o", ms=3, label=lab)
ax.set_xlabel("predicted P(wheel)")
ax.set_ylabel("observed wheel rate")
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.6)
ax.legend(fontsize=6.5, loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "calib_rank2.pdf")
nums["calib_rank2"] = cal
p_wn = r2["wheelnet"].to_numpy()
y2 = r2["wheeled"].to_numpy()
nums["rank2_lt2pct"] = [float((p_wn < 0.02).mean()), float(y2[p_wn < 0.02].mean())]
nums["rank2_gt20pct"] = [float((p_wn > 0.2).mean()), float(y2[p_wn > 0.2].mean())]
top3 = bd.filter(pl.col("rank_in_pack") <= 3)["wheelnet"].to_numpy()
nums["top3_ge50"] = float((top3 >= 0.5).mean())
pa = r2["alsa_ata_logreg"].to_numpy()
sel = (pa > 0.3) & (pa <= 0.5)
nums["rank2_alsa_30_50"] = [float(sel.mean()), float(y2[sel].mean())]
sel = (p_wn > 0.3) & (p_wn <= 0.5)
nums["rank2_wn_30_50"] = [float(sel.mean()), float(y2[sel].mean())]

# ---------------------------------------------------------------- weekly walk-forward (MSH)
tj = load(R / "MSH.PremierDraft" / "temporal.json")
arms = tj["arms"]
weeks = sorted(arms["retrain"], key=int)
fig, ax = plt.subplots(figsize=(3.4, 2.1))
wk = [int(w) + 1 for w in weeks]
for a, lab, c, ls in [("frozen", "frozen after week 1", ORANGE, "-"), ("finetune", "weekly fine-tune", BLUE, "-"),
                      ("replay", "fine-tune + replay", AQUA, "--")]:
    ax.plot(wk, [1e3 * (arms[a][w]["logloss"] - arms["retrain"][w]["logloss"]) for w in weeks], color=c, ls=ls,
            marker="o", ms=3, label=lab)
ax.plot(wk, [1e3 * (arms["lookup_frozen"][w]["logloss"] - arms["lookup_refit"][w]["logloss"]) for w in weeks],
        color=GRAY, marker="s", ms=2.5, lw=1.0, label="per-card lookup, frozen vs refit")
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xticks(wk)
ax.set_xlabel("week of the format being scored (MSH)")
ax.set_ylabel("log loss vs weekly retrain ($\\times10^{-3}$)")
ax.legend(fontsize=6.3, loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "temporal.pdf")
nums["temporal"] = {a: {w: arms[a][w]["logloss"] for w in weeks} for a in arms}
nums["temporal_weeks"] = tj["weeks"]

# ---------------------------------------------------------------- open lane (15 sets)
lane_sets = sorted(p.parent.name.split(".")[0] for p in R.glob("*.PremierDraft/open_lane.json"))
L = {s: load(R / f"{s}.PremierDraft" / "open_lane.json") for s in lane_sets}
RANKS = ["bronze", "silver", "gold", "platinum", "diamond", "mythic"]
SKILL = ["<0.50", "0.50-0.56", "0.56-0.62", ">=0.62"]
lrows = []
for s in lane_sets:
    r = L[s]
    pay = r["payoff"]["P1 openness of main colours | +skill"]
    fr = r["follow_by_rank"]
    q = r["winrate_by_P1_openness_quintile"]
    lrows.append(dict(set=s, drafts=r["drafts"], p13=r["persistence"]["P1-P3 (same neighbour)"],
                      p12=r["persistence"]["P1-P2"], follow=r["follow_all"]["slope"],
                      f_lo=fr[RANKS[0]]["slope"] if RANKS[0] in fr else np.nan, f_hi=fr["mythic"]["slope"],
                      skill=[r["follow_by_skill"][k]["slope"] for k in SKILL],
                      pay=pay["coef"][0], pay_se=pay["se"][0], q1=q[0], q5=q[4],
                      sep=r["payoff"]["P1, P2, P3 separately | +skill"]["coef"],
                      other=r["payoff"]["P1 main vs P1 other colours | +skill"]["coef"][1],
                      in_best=r["share_in_most_open_P1_pair"], wr_in=r["winrate_in_vs_not_most_open_pair"]))
lm = lambda k: float(np.nanmean([r[k] for r in lrows]))  # noqa: E731
sgn = lambda v: (f"${'-' if round(v, 3) < 0 else '+'}$" if round(v, 3) else "") + f"{abs(v):.3f}"  # noqa: E731
nums["lane"] = {k: lm(k) for k in ("p13", "p12", "follow", "f_lo", "f_hi", "pay", "q1", "q5", "other", "in_best")}
nums["lane"]["sep"] = np.mean([r["sep"] for r in lrows], 0).tolist()
nums["lane"]["skill"] = np.mean([r["skill"] for r in lrows], 0).tolist()
nums["lane"]["other_neg"] = int(sum(r["other"] < 0 for r in lrows))
nums["lane"]["sets"] = len(lane_sets)
nums["lane"]["drafts"] = int(sum(r["drafts"] for r in lrows))
nums["lane"]["pay_se_range"] = [float(min(r["pay_se"] for r in lrows)), float(max(r["pay_se"] for r in lrows))]
nums["lane"]["wr_in_minus_not"] = [float(min(r["wr_in"][0] - r["wr_in"][1] for r in lrows)),
                                   float(max(r["wr_in"][0] - r["wr_in"][1] for r in lrows))]
skill_up = sum(all(np.diff(r["skill"]) > -1e-9) for r in lrows)
nums["lane"]["skill_monotone_sets"] = int(skill_up)
nums["lane"]["skill_hi_gt_lo"] = int(sum(r["skill"][-1] > r["skill"][0] for r in lrows))

lb = []
for r in lrows:
    lb.append(f"{r['set']} & {r['drafts'] / 1e3:.0f} & {r['p13']:.3f} & {r['p12']:.3f} & {r['follow']:.2f} & "
              f"{r['f_lo']:.2f} $\\to$ {r['f_hi']:.2f} & +{r['pay']:.3f} $\\pm$ {r['pay_se']:.3f} & "
              f"{100 * r['q1']:.1f} $\\to$ {100 * r['q5']:.1f} & {sgn(r['other'])} \\\\")
lb.append("\\midrule")
lb.append(f"mean & & {lm('p13'):.3f} & {lm('p12'):.3f} & {lm('follow'):.2f} & {lm('f_lo'):.2f} $\\to$ {lm('f_hi'):.2f} & "
          f"+{lm('pay'):.3f} & {100 * lm('q1'):.1f} $\\to$ {100 * lm('q5'):.1f} & {sgn(lm('other'))} \\\\")
(OUT / "table_lane.tex").write_text(r"""\begin{tabular}{lr cc c c c c c}
\toprule
 & drafts & \multicolumn{2}{c}{persistence (corr.)} & follow & follow slope & payoff per sd & game win \% by & payoff, \\
\cmidrule(lr){3-4}
set & (k) & P1--P3 & P1--P2 & slope & bronze $\to$ mythic & (+ skill) & P1 quintile 1 $\to$ 5 & other colours \\
\midrule
""" + "\n".join(lb) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Fig: open lane, two panels
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.4, 2.0), gridspec_kw={"width_ratios": [1, 1.15]})
q = np.array([L[s]["winrate_by_P1_openness_quintile"] for s in lane_sets])
for row in q:
    a1.plot(np.arange(1, 6), 100 * row, color="#d4d3cd", lw=0.7)
a1.plot(np.arange(1, 6), 100 * q.mean(0), color=BLUE, marker="o", ms=3)
a1.set_xticks(range(1, 6))
a1.set_xlabel("P1 openness quintile\nof your main colours")
a1.set_ylabel("game win rate (%)")
sep = np.array([r["sep"] for r in lrows])
oth = np.array([r["other"] for r in lrows])
cols = [sep[:, 0], sep[:, 1], sep[:, 2], oth]
labs = ["P1", "P2", "P3", "P1,\nother"]
for i, (v, lab) in enumerate(zip(cols, labs)):
    jit = np.linspace(-0.18, 0.18, len(v))
    a2.scatter(i + jit, v, s=5, color="#bdbcb6", zorder=1)
    a2.scatter([i], [v.mean()], s=26, color=BLUE if i < 3 else ORANGE, zorder=3)
a2.axhline(0, color=MUTED, lw=0.6)
a2.set_xticks(range(4), labs)
a2.set_ylabel("log-odds per game per sd")
a2.set_xlabel("openness of ... (fitted jointly)")
a2.grid(axis="x", visible=False)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(OUT / "open_lane.pdf")

# ---------------------------------------------------------------- force vs follow
fvf = {s: load(R / f"{s}.PremierDraft" / "force_vs_follow.json") for s in ("MSH", "FDN")}
nums["fvf"] = {s: {k: (v["seat0"], v.get("vs_follow"), v.get("vs_follow_se")) for k, v in f.items()
                   if isinstance(v, dict) and "seat0" in v} for s, f in fvf.items()}
nums["fvf_pairs"] = {s: f["force"]["by_pair"] for s, f in fvf.items()}

# ---------------------------------------------------------------- MSH headline
ms = {r["model"]: r for r in load(R / "MSH.PremierDraft" / "summary.json")}
for name in ("time", "multiset_plus_target", "multiset_zeroshot"):
    p = R / "MSH.PremierDraft" / f"results_wheelnet_{name}.json"
    ms[f"wheelnet_{name}"] = load(p)
nums["msh"] = {k: {m: v[m] for m in ("logloss", "brier", "auc", "ece") if m in v} for k, v in ms.items()}
(OUT / "numbers.json").write_text(json.dumps(nums, indent=1, default=float))
print(json.dumps({k: v for k, v in nums.items() if k not in ("ece_top3", "msh", "temporal")}, indent=1, default=float))
