# wheelnet: what we found

17Lands Premier Draft data, 15 sets (~1.9M drafts). Every model uses a chronological split: train on the first 70% of
drafts, test on the last 15%. MSH is the development set, and results were replicated on up to 15 sets. The full
tables are in each repo's README. The work is split across sibling repos: this one (wheel and open lane),
[picknet](../picknet), [mtg-deck-strength](../mtg-deck-strength) and [mtg-bombs](../mtg-bombs), all reading data through
[mtg-data](../mtg-data).

## The answers

| question | answer | where |
|---|---|---|
| **Will this card wheel?** | Yes. The best model cuts log loss **9.5%** vs the ALSA/ATA baseline (AUC 0.83 → 0.87), and **12–17%** on the best card you passed, the one you actually care about. | `wheel.ipynb` (this repo) |
| **Can we predict human picks?** | **68.6%** top-1, vs 49.7% for a pick-order bot. | `drafting.ipynb` (this repo); picknet README |
| **Does the open lane pay?** | Open colours win **+3.2 pts** of game win rate, but that comes from more good cards flowing your way, not from reading it. In simulation, forcing a random pair ties drafting like a human, forcing the best pair beats it (+1.3 pts MSH), and pure signal-chasing loses. | `drafting.ipynb` part 2 (this repo) |
| **How good is this deck?** | Spearman **0.28** with run win rate, about as high as Bo1 noise allows. | mtg-deck-strength `decks.ipynb` |
| **Does the DraftEncoder paper hold up?** | Its numbers reproduce (0.19–0.21), but its 2× over linear comes from a weak baseline. The same text features as an order-free linear model match or beat it on the same set. Only on an unseen set, with META, does the transformer win (+0.01). Four days of the new set's data beat everything else. | mtg-deck-strength README, "Full DraftEncoder replication" |
| **Odds in the next game?** | At a fixed deck, each win so far costs ~2 pts (matchmaking). Being on the play is worth ~5 pts. | mtg-deck-strength `decks.ipynb` part 2 |
| **Is card synergy real?** | Yes, but small: about −0.3 ×1000 log loss per set, measured cleanly on opening hands (11/12 sets). The big-looking effect from cards seen together in a game was mostly card draw. Synergy has to be learned within each set; card text doesn't predict it for a new set. | mtg-deck-strength `synergy.ipynb` |
| **How much do the cards in your deck matter?** | By GIH WR, deck card strength explains about **a third** of the real spread in drafters' win rates (0.20–0.47 by set, and the sets really differ). GIH WR beats IWD as the card rating on all 19 sets (r with total wins 0.21 vs 0.13). In opened packs, card strength is near-symmetric with fat tails on both sides, and about 1 pack in 7 has an above-fence card. | mtg-bombs `bombs.ipynb`, `bombs_packs.ipynb` |

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
affinity profiles; seen-together synergy (card draw in disguise); day-zero synergy from card text; GIH WR prior; archetype identity; pick-based skill proxy; deeper WheelNet; rank or skill inputs to
WheelNet. Scripts and notebooks for the archived ones are in `scripts/archive/` and `notebooks/archive/` of the repo
they belong to (mostly mtg-deck-strength).

## Open

- **Synergy with replay data:** only MSH has per-turn casts here. For more sets, pairs actually cast together (and
  when), and your-card × opponent-card matchups.
- Stronger simulated drafters, and tables where several seats force.
- A live overlay: P(wheel) for each card in your pack, fine-tuned weekly.

## Notebooks

This repo: `wheel.ipynb` · `drafting.ipynb`. mtg-deck-strength: `decks.ipynb` · `synergy.ipynb`. Each merges two former
notebooks as parts 1 and 2, and each part runs on its own. mtg-bombs: `bombs.ipynb` (card ratings, deck strength,
bombs) and `bombs_packs.ipynb` (card strength in opened packs, which reads `bombs.ipynb`'s card cache).
