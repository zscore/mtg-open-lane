# mtg-open-lane: what we found

17Lands Premier Draft data, 15 sets (~1.9M drafts). Every model uses a chronological split: train on the first 70% of
drafts, test on the last 15%. MSH is the development set, and results were replicated on up to 15 sets. The full
tables are in the README.

## The answers

| question | answer | notebook |
|---|---|---|
| **Will this card wheel?** | Yes. The best model cuts log loss **9.5%** vs the ALSA/ATA baseline (AUC 0.83 → 0.87), and **12–17%** on the best card you passed, the one you actually care about. | `wheel.ipynb` |
| **Can we predict human picks?** | **68.6%** top-1, vs 49.7% for a pick-order bot. | `drafting.ipynb` |
| **Does the open lane pay?** | Open colours win **+3.2 pts** of game win rate, but that comes from more good cards flowing your way, not from reading it. In simulation, forcing a random pair ties drafting like a human, forcing the best pair beats it (+1.3 pts MSH), and pure signal-chasing loses. | `drafting.ipynb` part 2 |
| **How good is this deck?** | Spearman **0.28** with run win rate, about as high as Bo1 noise allows. | `decks.ipynb` |
| **Does the DraftEncoder paper hold up?** | Its numbers reproduce (0.19–0.21), but its 2× over linear comes from a weak baseline. The same text features as an order-free linear model match or beat it on the same set. Only on an unseen set, with META, does the transformer win (+0.01). Four days of the new set's data beat everything else. | README, "Full DraftEncoder replication" |
| **Odds in the next game?** | At a fixed deck, each win so far costs ~2 pts (matchmaking). Being on the play is worth ~5 pts. | `decks.ipynb` part 2 |
| **Is card synergy real?** | Small. Pairs add little beyond card power + deck structure. The one big-looking effect (cards seen together in a game) is probably card draw in disguise, and round 8 is checking that. | `synergy.ipynb` |

## What mattered

1. **Context beats card identity.** Card identity alone adds almost nothing over ALSA (+0.6%), while knowing the
   rest of the pack adds +5% and table signals add more on top. For picks, the pool you've drafted is worth 15 points
   of accuracy, the pack 3 and the history 1.
2. **Train on many sets with shared card features.** A model that reads cards through their features (colour,
   cost, rules text), trained on 7 other sets plus this one, is the best wheel model *and* the best pick model. On a
   single set, transformers rarely beat linear models or LightGBM.
3. **Use the registered deck, not the pool** (+0.03 Spearman on every set). The best deck model is LightGBM on deck
   counts, averaged with a skill-adjusted game-level model. Cheap build rules: two colours beat three, don't
   splash lightly, and 16 lands edges out 17.
4. **Human picks are free synergy labels.** The pick model's "card A makes you take card B" affinity was the best
   single synergy feature, better than any pair model fitted on game results.
5. **Stay current.** A wheel model frozen after week 1 is worse by week 5 than the whole gap between model
   architectures. A cheap weekly fine-tune fixes it.
6. **Watch the confounds.** The 17Lands skill bucket leaks the outcome. Rank sets your opponents. Your record sets
   your opponents too. Cards *cast* look great because winners cast more.

## Dead ends

Transformers on a single set (DraftEncoder, the deck transformer, a cross-set deck network); DraftEncoder pooled over
12 sets from card text (ties the linear bag with META, and loses without it); Sealed/draft multitask;
ensembling the transformer with LightGBM; higher-rank or richer pair terms (saturated after rank 4); per-card
affinity profiles; GIH WR prior; archetype identity; pick-based skill proxy; deeper WheelNet; rank or skill inputs to
WheelNet. Scripts and notebooks for the archived ones are in `scripts/archive/` and `notebooks/archive/`.

## Open

- **Synergy round 8** (`scripts/deck_synergy_v8.py`, running on SOS): rerun the seen-together synergy model on the
  opening hand only, since cards seen later partly depend on card draw. If the effect vanishes, deck-level synergy
  (small) is the honest answer.
- Stronger simulated drafters, and tables where several seats force.
- A live overlay: P(wheel) for each card in your pack, fine-tuned weekly.

## Notebooks

`wheel.ipynb` · `drafting.ipynb` · `decks.ipynb` · `synergy.ipynb`. Each one merges two former notebooks as parts 1 and 2,
and each part runs on its own.
