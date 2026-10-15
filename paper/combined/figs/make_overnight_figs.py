"""Figures for overnight_findings.tex, from results/tablenet/*.json.

    python paper/combined/figs/make_overnight_figs.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "tablenet"
FIG = Path(__file__).resolve().parent
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal: step 1 -> w=1 -> w=3
WHEELNET = dict(BLB=.4622, DFT=.4239, DSK=.4159, ECL=.4508, FDN=.4202, FIN=.4375, FRA=.4330, HOB=.4171, LCI=.4434,
                MKM=.4487, MSH=.4249, OTJ=.4058, SOS=.4150, TDM=.4177, TLA=.4214, TMT=.4173)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def test(name):
    return json.loads((RES / f"{name}.json").read_text())["test"]


def summary(name, w0):
    r = test(name)
    gap = st.mean((r[k]["wheel_logloss"] / v - 1) * 100 for k, v in WHEELNET.items())
    nll = st.mean(r[k]["pick_nll"] - w0[k]["pick_nll"] for k in r)
    top1 = st.mean((r[k]["pick_top1"] - w0[k]["pick_top1"]) * 100 for k in r)
    return gap, nll, top1


def tradeoff():
    w0 = test("tablenet_joint_w0")
    sweep = [("w=0.3", "tablenet_joint"), ("w=1", "tablenet_joint_w1"), ("w=3", "tablenet_joint_w3")]
    other = [("w=1, 2 passes", "tablenet_joint_w1_ep2"), ("w=3, history blocked", "tablenet_joint_w3_hblock")]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.00125, 0.03, "per-set WheelNet", color=INK2, ha="right", va="bottom", fontsize=8)
    pts = [summary(n, w0) for _, n in sweep]
    ax.plot([p[1] for p in pts], [p[0] for p in pts], color=BLUE, lw=2, marker="o", ms=7,
            mec="white", mew=2, label="wheel weight, one pass", zorder=3)
    for (lab, _), p in zip(sweep, pts):
        ax.annotate(lab, (p[1], p[0]), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=8)
    for (lab, n) in other:
        p = summary(n, w0)
        ax.plot(p[1], p[0], ls="none", marker="s", ms=7, color=ORANGE, mec="white", mew=2, zorder=3,
                label="variants" if lab.startswith("w=1") else None)
        ax.annotate(lab, (p[1], p[0]), xytext=(6, -10 if "blocked" in lab else 4), textcoords="offset points",
                    color=INK, fontsize=8)
    ax.set_xlabel("pick NLL cost vs. pick-only control (nats/pick, mean over 19 sets)")
    ax.set_ylabel("wheel log loss vs. WheelNet (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlim(-0.0001, 0.0014)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.pdf")


def cosines():
    a = json.loads((RES / "grad_cosine.json").read_text())["ckpts"]
    b = json.loads((RES / "grad_cosine_late.json").read_text())["ckpts"]
    runs = [("step 1 (before joint training)", a["step1"]), ("w=1", a["w1"]), ("w=3", b["w3"])]
    groups = ["card table", "input embeddings", "history layer 0", "history layer 1", "history layer 2",
              "history layer 3", "decoder blocks.0", "decoder blocks.1", "decoder blocks.2"]
    labels = ["card table", "history inputs", "history 0", "history 1", "history 2", "history 3",
              "decoder 0", "decoder 1", "decoder 2"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    n, h = len(runs), 0.26
    for j, ((lab, r), c) in enumerate(zip(runs, RAMP)):
        ys = [i + (j - 1) * h for i in range(len(groups))]
        ax.barh(ys, [r[g]["pick-wheel"]["cos_mean"] for g in groups], height=h - 0.04, color=c, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(groups)), labels)
    ax.invert_yaxis()
    ax.axhspan(1.5, 5.5, color="#f0efec", zorder=0)
    ax.text(-0.53, 1.62, "history encoder", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("cosine between batch-averaged pick and wheel gradients (1000 batches)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlim(-0.56, 0.12)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cosines.pdf")


if __name__ == "__main__":
    tradeoff()
    cosines()
