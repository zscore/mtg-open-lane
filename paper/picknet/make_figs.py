"""Figures and numbers for paper/picknet/picknet.tex, from saved test predictions.

    .venv/bin/python paper/picknet/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figs"
RUN = "all_d256_ep2"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "lines.linewidth": 1.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
# puder v1 re-scored on its own Premier test drafts (real picks only): paper/picknet/puder/
PUDER_ALL = {k.lstrip("$"): v for k, v in
             json.loads((Path(__file__).resolve().parent / "puder" / "puder_per_set.json").read_text()).items()}
PUDER = {s: (PUDER_ALL[s]["top1"], PUDER_ALL[s]["drafts"]) for s in
         ("BLB", "DSK", "FDN", "LCI", "MKM", "MOM", "OTJ", "WOE")}


def per_set(s, m):
    p = R / f"{s}.PremierDraft" / "picknet" / f"{m}.json"
    return json.loads(p.read_text())["top1"] if p.exists() else None


res = {n: json.loads((R / "picknet_all" / f"{n}.json").read_text()) for n in ("all_d256", RUN)}
test = res[RUN]["test"]
sets = sorted(test)
nums = {}

# ---- headline numbers
w = np.array([test[s]["picks"] for s in sets])
nums["all17_top1"] = float(np.average([test[s]["top1"] for s in sets], weights=w))
nums["all17_nf"] = float(np.average([test[s]["top1_nonforced"] for s in sets], weights=w))
nums["all17_nll"] = float(np.average([test[s]["nll"] for s in sets], weights=w))
nums["test_picks"] = int(w.sum())
nums["ep1_all17_top1"] = float(np.average([res["all_d256"]["test"][s]["top1"] for s in sets], weights=w))
pw = np.array([PUDER[s][1] for s in PUDER])
nums["shared_ours"] = float(np.average([test[s]["top1"] for s in PUDER], weights=pw))
nums["shared_ours_ep1"] = float(np.average([res["all_d256"]["test"][s]["top1"] for s in PUDER], weights=pw))
nums["shared_puder"] = float(np.average([PUDER[s][0] for s in PUDER], weights=pw))

# per-draft spread -> standard error of puder's 478-draft average
drafts, sds, p1p1 = 0, [], {}
for s in sets:
    z = np.load(R / "picknet_all" / f"test_{RUN}_{s}.npz")
    v = z["valid"]
    acc = (z["top1"] & v).sum(1) / v.sum(1)
    drafts += len(acc)
    if s in PUDER:
        sds.append(acc.std())
    p1p1[s] = (z["top1"][:, 0][v[:, 0]].mean() if v[:, 0].any() else np.nan, v[:, 0].sum())
nums["test_drafts"] = drafts
nums["puder_se"] = float(np.mean(sds) / np.sqrt(pw.sum()))
nums["p1p1"] = float(np.nansum([a * n for a, n in p1p1.values()]) / sum(n for a, n in p1p1.values() if not np.isnan(a)))

# ---- Fig: per-set accuracy
fig, ax = plt.subplots(figsize=(3.4, 2.8))
order = sorted(sets, key=lambda s: test[s]["top1"])
y = np.arange(len(order))
for i, s in enumerate(order):
    pool, rat = per_set(s, "pool_d32"), per_set(s, "rating")
    if pool:
        ax.plot([pool, test[s]["top1"]], [i, i], color="#d4d3cd", lw=1, zorder=1)
ax.scatter([per_set(s, "rating") or np.nan for s in order], y, s=14, color=GRAY, marker="|", label="rating (per set)", zorder=2)
ax.scatter([per_set(s, "pool_d32") or np.nan for s in order], y, s=16, color=AQUA, label="pool bilinear (per set)", zorder=2)
ax.scatter([PUDER[s][0] if s in PUDER else np.nan for s in order], y, s=22, facecolor="white", edgecolor=ORANGE,
           lw=1.2, label="puder (own random split)", zorder=3)
ax.scatter([test[s]["top1"] for s in order], y, s=18, color=BLUE, label="PickNet, all sets", zorder=4)
ax.set_yticks(y, order)
ax.set_xlabel("top-1 agreement with the human pick (test drafts)")
ax.set_xlim(0.43, 0.74)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "per_set.pdf")

# ---- Fig: accuracy by pick within the pack, MSH (14-card packs)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
n = 14
z = np.load(R / "picknet_all" / f"test_{RUN}_MSH.npz")
slot = (np.arange(3)[:, None] * 15 + np.arange(n)[None]).ravel()
curves = {"PickNet, all sets": (z["top1"][:, slot], BLUE)}
for m, lab, c in [("picknet_big", "PickNet, MSH only", ORANGE), ("pool_d32", "pool bilinear", AQUA),
                  ("rating", "rating", GRAY)]:
    curves[lab] = (np.load(R / "MSH.PremierDraft" / "picknet" / f"test_{m}.npz")["top1"], c)
x = np.arange(1, 3 * n + 1)
for lab, (a, c) in curves.items():
    acc = a.mean(0)
    for k in range(3):
        ax.plot(x[k * n:(k + 1) * n], acc[k * n:(k + 1) * n], color=c, label=lab if k == 0 else None,
                lw=1.5 if "all" in lab else 1.1)
for k in (1, 2):
    ax.axvline(k * n + 0.5, color="#c8c7c1", lw=0.6, ls=":")
ax.set_xticks([1, 5, 10, 15, 19, 24, 29, 33, 38], ["1", "5", "10"] * 3)
for k in range(3):
    ax.text(k * n + n / 2 + 0.5, 1.03, f"pack {k + 1}", ha="center", va="bottom", color=MUTED, fontsize=7)
ax.set_xlabel("pick within the pack (MSH, 14-card packs)")
ax.set_ylabel("top-1")
ax.set_ylim(0.3, 1.01)
ax.set_xlim(0.3, 42.7)
ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, handlelength=1.2, columnspacing=0.7, handletextpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "by_pick.pdf")
nums["msh_by_pack_all"] = [float(curves["PickNet, all sets"][0][:, k * n:(k + 1) * n].mean()) for k in range(3)]
nums["msh_by_pack_big"] = [float(curves["PickNet, MSH only"][0][:, k * n:(k + 1) * n].mean()) for k in range(3)]
# mid-pack (picks 3-8) gain over rating
mid = np.r_[2:8, 16:22, 30:36]
nums["msh_mid_gain"] = float(curves["PickNet, all sets"][0][:, mid].mean() - curves["rating"][0][:, mid].mean())

# ---- Fig: training curve
fig, ax = plt.subplots(figsize=(3.4, 1.9))
steps = res["all_d256"]["cfg"]["steps"]
b = res["all_d256"]["cfg"]["batch"]
for n_, off, c, lab in [("all_d256", 0, BLUE, "epoch 1 (peak lr 5e-4)"), (RUN, steps, ORANGE, "epoch 2 (peak lr 2e-4)")]:
    h = res[n_]["history"]
    ax.plot([(e["step"] + off) * b / 1e6 for e in h], [e["val_top1"] for e in h], color=c, label=lab, marker="o", ms=2)
ax.set_xlabel("training drafts seen (millions)")
ax.set_ylabel("val top-1")
ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "train_curve.pdf")

# ---- MSH tiers
wr, ng, rk = z["wr"], z["ng"], z["rank"]
old = np.load(R / "MSH.PremierDraft/picknet/test_picknet_big.npz")["top1"]
rat = np.load(R / "MSH.PremierDraft/picknet/test_rating.npz")["top1"]
tiers = [("all drafters", np.ones(len(wr), bool)), ("17Lands win rate $<$ 50\\%", (wr >= 0) & (wr < .5)),
         ("Mythic rank", rk == 5), ("win rate $\\geq$ 60\\%", wr >= .6),
         ("win rate $\\geq$ 62\\%, $\\geq$ 100 games", (wr >= .62) & (ng >= 100))]
nums["tiers"] = [(lab, int(s.sum()), float(rat[s].mean()), float(old[s].mean()),
                  float(z["top1"][s][z["valid"][s]].mean())) for lab, s in tiers]
nums["per_set"] = {s: dict(ours=test[s]["top1"], nf=test[s]["top1_nonforced"], nll=test[s]["nll"],
                           ep1=res["all_d256"]["test"][s]["top1"], pool=per_set(s, "pool_d32"),
                           rating=per_set(s, "rating"), puder=PUDER.get(s, (None,))[0],
                           drafts=int(test[s]["picks"])) for s in sets}

# ---- Table 1: top-1, non-forced top-1 and NLL for every model, same picks within a model's test set
def base(s, m):
    """Per-set baseline on all real picks; forced = the rating model's NLL is exactly 0 (one card left)."""
    p = R / f"{s}.PremierDraft" / "picknet" / f"test_{m}.npz"
    if not p.exists():
        return None
    z, rn = np.load(p), np.load(R / f"{s}.PremierDraft" / "picknet" / "test_rating.npz")["nll"]
    ok = ~np.isnan(z["nll"])  # P1P1 is missing (NaN) in some sets
    nf = ok & (rn != 0)
    return dict(top1=z["top1"][ok].mean(), nf=z["top1"][nf].mean(), nll=z["nll"][ok].mean(), n=int(ok.sum()))


rows = {s: dict(rating=base(s, "rating"), pool=base(s, "pool_d32"),
                puder=PUDER_ALL.get(s) if s in PUDER else None,
                ours=dict(top1=test[s]["top1"], nf=test[s]["top1_nonforced"], nll=test[s]["nll"], n=test[s]["picks"]))
        for s in sets}
MODELS, METRICS = ("rating", "pool", "puder", "ours"), ("top1", "nf", "nll")


def cell(v, k, best):
    if v is None:
        return "--"
    txt = f"{v[k]:.3f}".lstrip("0")
    return f"\\textbf{{{txt}}}" if best else txt


def line(label, picks, r):
    out = [label, picks]
    for k in METRICS:
        vals = [r[m][k] if r[m] else None for m in MODELS]
        ok = [v for v in vals if v is not None]
        b = (min if k == "nll" else max)(ok)
        out += [cell(r[m], k, r[m] is not None and r[m][k] == b) for m in MODELS]
    return " & ".join(out) + " \\\\"


def avg(sets_, m, wt):
    return {k: float(np.average([rows[s][m][k] for s in sets_], weights=[wt(s) for s in sets_])) for k in METRICS}


body = [line(s, f"{test[s]['picks'] / 1e3:.0f}", rows[s]) for s in sets]
shared = list(PUDER)
w_pud = lambda s: PUDER[s][1]  # noqa: E731
summ = dict(rating=None, pool=None, puder=avg(shared, "puder", w_pud), ours=avg(shared, "ours", w_pud))
body.append("\\midrule\n" + line("8 shared$^a$", "", summ))
have = [s for s in sets if rows[s]["rating"]]
w_pk = lambda s: rows[s]["ours"]["n"]  # noqa: E731
body.append(line(f"{len(have)} sets$^b$", "", dict(rating=avg(have, "rating", w_pk), pool=avg(have, "pool", w_pk),
                                                 puder=None, ours=avg(have, "ours", w_pk))))
body.append(line("all 17$^b$", f"{nums['test_picks'] / 1e3:.0f}",
                 dict(rating=None, pool=None, puder=None, ours=avg(sets, "ours", w_pk))))
HEAD = r"""\begin{tabular}{lr cccc cccc cccc}
\toprule
 & test picks & \multicolumn{4}{c}{top-1 $\uparrow$} & \multicolumn{4}{c}{non-forced top-1 $\uparrow$} & \multicolumn{4}{c}{NLL $\downarrow$} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}
set & (k) & rating & pool & puder & PickNet & rating & pool & puder & PickNet & rating & pool & puder & PickNet \\
\midrule
"""
(OUT / "table_sets.tex").write_text(HEAD + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")
nums["shared_nf"] = {m: summ[m]["nf"] for m in ("puder", "ours")}
nums["shared_nll"] = {m: summ[m]["nll"] for m in ("puder", "ours")}
nums["wins_vs_puder"] = {k: sum((rows[s]["ours"][k] < rows[s]["puder"][k]) if k == "nll" else
                                (rows[s]["ours"][k] > rows[s]["puder"][k]) for s in shared) for k in METRICS}
nums["baseline_avgs"] = {m: avg(have, m, w_pk) for m in ("rating", "pool", "ours")}
(OUT / "numbers.json").write_text(json.dumps(nums, indent=1))
print(json.dumps({k: v for k, v in nums.items() if k not in ("per_set",)}, indent=1))
