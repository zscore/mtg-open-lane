"""Markdown tables for the open-lane and wheel-breakdown sections: python scripts/open_lane_report.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import ROOT  # noqa: E402


def open_lane():
    rows = []
    print("| set | drafts | corr P1–P3 (same neighbour) | corr P1–P2 | follow slope (bronze → mythic) "
          "| payoff / sd, + skill | win rate, P1-openness quintile 1 → 5 | most-open P1 pair: in / not |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sorted(ROOT.glob("results/*.PremierDraft/open_lane.json")):
        r = json.loads(f.read_text())
        pe, fr = r["persistence"], r["follow_by_rank"]
        ranks = [fr[k]["slope"] for k in ("bronze", "mythic") if k in fr]
        pay = r.get("payoff", {}).get("P1 openness of main colours | +skill")
        q = r.get("winrate_by_P1_openness_quintile")
        inn = r.get("winrate_in_vs_not_most_open_pair")
        rows.append((pe["P1-P3 (same neighbour)"], pe["P1-P2"], r["follow_all"]["slope"],
                     pay["coef"][0] if pay else np.nan, (q[4] - q[0]) if q else np.nan))
        print(f"| {r['set']} | {r['drafts']:,} | {pe['P1-P3 (same neighbour)']:.3f} | {pe['P1-P2']:.3f} "
              f"| {r['follow_all']['slope']:.2f} ({' → '.join(f'{x:.2f}' for x in ranks)}) "
              f"| {pay['coef'][0]:+.3f} ± {pay['se'][0]:.3f} | {q[0]:.3f} → {q[4]:.3f} "
              f"| {inn[0]:.3f} / {inn[1]:.3f} |" if pay else
              f"| {r['set']} | {r['drafts']:,} | {pe['P1-P3 (same neighbour)']:.3f} | {pe['P1-P2']:.3f} "
              f"| {r['follow_all']['slope']:.2f} | — | — | — |")
    m = np.nanmean(np.array(rows), 0)
    print(f"| **mean** | | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.2f} | {m[3]:+.3f} | +{100 * m[4]:.1f} pts | |")


def breakdown():
    print("\n| set | ALSA → WheelNet, all passed cards | best passed card (rank 1) | rank 2 | rank 3 "
          "| ECE rank 1–3, ALSA / WheelNet |")
    print("|---|---|---|---|---|---|")
    for f in sorted(ROOT.glob("results/*.PremierDraft/wheel_breakdown.json")):
        r = json.loads(f.read_text())
        s = f.parent.name.split(".")[0]
        if "wheelnet" not in r["all"]:
            continue
        g = lambda sl: 100 * (1 - r[sl]["wheelnet"]["logloss"] / r[sl]["alsa_ata_logreg"]["logloss"])
        ece = lambda m: np.mean([r[f"rank{k}"][m]["ece"] for k in (1, 2, 3)])
        print(f"| {s} | {g('all'):+.1f}% | {g('rank1'):+.1f}% | {g('rank2'):+.1f}% | {g('rank3'):+.1f}% "
              f"| {ece('alsa_ata_logreg'):.3f} / {ece('wheelnet'):.3f} |")


if __name__ == "__main__":
    open_lane()
    breakdown()
