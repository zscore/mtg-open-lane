"""Markdown tables from the pick-model and deck-synergy-v2 result files: python scripts/picknet_report.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def pick_tables():
    print("### Pick models, MSH test drafts\n")
    print("| model | params | NLL / pick | top-1 | P1 / P2 / P3 top-1 |\n|---|---|---|---|---|")
    d = R / "MSH.PremierDraft" / "picknet"
    for n in ["rating", "pool_d32", "pool_d64", "picknet_nohist", "picknet", "multiset_zeroshot",
              "multiset_plus_target", "picknet_big", "picknet_text"]:
        r = load(d / f"{n}.json")
        if r:
            print(f"| `{n}` | {r.get('params', 0):,} | {r['nll']:.4f} | {r['top1']:.3f} | "
                  f"{r['top1_p1']:.3f} / {r['top1_p2']:.3f} / {r['top1_p3']:.3f} |")
    print("\n### Per set: rating vs pool (test drafts)\n")
    print("| set | rating top-1 | pool top-1 | rating NLL | pool NLL |\n|---|---|---|---|---|")
    for s in sorted(p.name.split(".")[0] for p in R.glob("*.PremierDraft")):
        a, b = load(R / f"{s}.PremierDraft/picknet/rating.json"), load(R / f"{s}.PremierDraft/picknet/pool_d32.json")
        if a and b:
            print(f"| {s} | {a['top1']:.3f} | {b['top1']:.3f} | {a['nll']:.3f} | {b['nll']:.3f} |")
    print("\n### Simulated tables (MSH, 1000 tables of 8 bots, real opened packs)\n")
    print("| bots | ATA r | ATA MAE | ALSA r | per-card wheel r | wheel MAE | pack-3 picks in top-2 colours | "
          "colour-pair TV | P1P1 r |\n|---|---|---|---|---|---|---|---|---|")
    for n in ["rating", "pool_d32", "picknet"]:
        r = load(d / f"sim_{n}.json")
        if r:
            print(f"| `{n}` | {r['ata_corr']:.3f} | {r['ata_mae']:.2f} | {r['alsa_corr']:.3f} | "
                  f"{r['wheel_card_corr']:.3f} | {r['wheel_card_mae']:.3f} | {r['commit_sim']:.3f} "
                  f"(real {r['commit_real']:.3f}) | {r['pair_share_tv']:.3f} | {r['p1p1_corr']:.3f} |")


def deck_tables():
    print("\n### Deck synergy v2 (test games, log loss vs linear ×1000, ±95%)\n")
    names = None
    rows = []
    for p in sorted(R.glob("*.PremierDraft/deck_synergy_v2/results.json")):
        r = load(p)
        names = names or [m["model"] for m in r["models"]][1:]
        rows.append((r["set"], {m["model"]: m for m in r["models"]}, r.get("split_half")))
    if not rows:
        return
    print("| set | " + " | ".join(names) + " | pair r (split half) | power r |")
    print("|---|" + "---|" * (len(names) + 2))
    agg = {n: [] for n in names}
    for s, ms, sh in rows:
        cells = []
        for n in names:
            m = ms.get(n)
            cells.append(f"{m['vs linear'] * 1000:+.2f} ± {m['vs linear ±95%'] * 1000:.2f}" if m else "")
            if m:
                agg[n].append(m["vs linear"])
        tail = f" | {sh['pair r']:.2f} | {sh['power r']:.2f} |" if sh else " | | |"
        print(f"| {s} | " + " | ".join(cells) + tail)
    print("| **mean** | " + " | ".join(f"{np.mean(v) * 1000:+.2f} ({sum(x < 0 for x in v)}/{len(v)} better)"
                                       for v in agg.values()) + " | | |")

    print("\n### Draft level: Spearman of the deck score vs test drafts' W/(W+L)\n")
    allm = ["linear"] + names
    print("| set | " + " | ".join(allm) + " |\n|---|" + "---|" * len(allm))
    sp = {n: [] for n in allm}
    for s, ms, _ in rows:
        if "draft spearman" not in ms["linear"]:
            continue
        for n in allm:
            sp[n].append(ms[n]["draft spearman"] if n in ms else np.nan)
        print(f"| {s} | " + " | ".join(f"{ms[n]['draft spearman']:.3f}" if n in ms else "" for n in allm) + " |")
    print("| **mean** | " + " | ".join(f"{np.nanmean(v):.3f}" for v in sp.values()) + " |")


if __name__ == "__main__":
    pick_tables()
    deck_tables()
