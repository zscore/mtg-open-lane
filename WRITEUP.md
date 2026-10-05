# mtg-open-lane: research writeup

A condensed summary of everything in `notebooks/` and the README's result sections, as of 2026-10-05. The README
keeps the full per-set tables. This file covers what we asked, what we found, which model to use for each job, and
what didn't work.

Data: 17Lands public Premier Draft datasets (draft, game and replay files). 15 sets downloaded, about 1.9M drafts and
76M pick rows. Every experiment uses the same chronological split per set: first 70% of drafts train, next 15%
validation, last 15% test. MSH (68,609 drafts) is the development set; results were then replicated on up to 15 sets.

## TL;DR

| question | answer | best model |
|---|---|---|
| Will this card wheel? | Yes, predictably. **+9.5% log loss vs the ALSA/ATA baseline** (AUC 0.831 → 0.866). The gain is largest for the cards you care about: +12–17% on the best card you passed. | WheelNet, features-only, pretrained on 7 other sets + this set (`train_wheelnet_multiset.py`) |
| Can we predict human picks? | **68.6% top-1** on MSH (pick-order bot: 49.7%). Pool > pack > history. | Features-only PickNet pretrained on 7 sets + this set (`train_picknet_multiset.py`) |
| Does the open lane pay? | Signals are real (P1↔P3 openness r = 0.22), drafters follow them, and open colours win **+3.2 pts** bottom→top quintile. But the payoff is *supply*, not the read: in simulation, forcing a random pair ties following. | `open_lane.py`, `force_vs_follow.py` |
| How good is this deck? | Spearman with run win rate **0.279** (mean over 12 sets), near the Bo1 noise ceiling (~0.26–0.28 at the spread models predict). | LightGBM on registered deck + deck features + affinity, logit-averaged with the skill-adjusted game FM |
| Is synergy real? | At the deck level, it is small and low-rank (pairs add −0.18 ×1000 log loss). Between cards **seen together in a game** it looked large and repeatable (−2.4 ×1000, 12/12 sets), but round 8 (in progress) suggests much of it is card-draw confounding. | `deck_synergy_v7.py` pair and hub tables |
| What are my odds next game? | Each prior win costs ~2 pts of win probability at a fixed deck (matchmaking). | `game_model.ipynb` `predict_next` |

## 1. Wheel prediction (`notebooks/wheelnet.ipynb`, `wheelnet_qa.ipynb`)

**Labels are free.** With 8 drafters, the pack seen at pick *p* returns at *p+8*, and 17Lands records both. Pack(p) −
your pick ⊇ pack(p+8) holds on every row except one malformed draft. One example is one distinct passed card, and the
label is whether a copy came back (MSH base rate 33.3%).

MSH test, log loss (relative to the ALSA/ATA logistic baseline):

| model | log loss | AUC | vs ALSA |
|---|---|---|---|
| pick number only | 0.6067 | 0.639 | −31.1% |
| ALSA + ATA per pick (baseline) | 0.4629 | 0.831 | 0 |
| card × pick lookup | 0.4602 | 0.833 | +0.6% |
| LightGBM, card + pack | 0.4397 | 0.850 | +5.0% |
| LightGBM, card + pack + signals | 0.4315 | 0.857 | +6.8% |
| WheelNet (raw history transformer) | 0.4249 | 0.862 | +8.2% |
| WheelNet + days-into-format input | 0.4218 | 0.864 | +8.9% |
| **WheelNet, features-only, 7 other sets + MSH** | **0.4188** | **0.866** | **+9.5%** |
| same, day zero (never saw MSH) | 0.5245 | 0.776 | −13.3% |

Findings:
1. **Card identity is used up by ALSA.** The per-card lookup adds only 0.6%. **Pack context is the biggest lever**
   (+5%): how many cards in the passed pack usually go before this one.
2. **Signals are real and grow through the draft.** They add +2.9% alone and stack with pack context. The gain over
   the card-only lookup roughly doubles from pack 1 to pack 3.
3. **The raw-history transformer beats hand-built features** once each passed card cross-attends to the history
   and gets explicit "taken by the other 7" tokens. After that, averaging with the GBM no longer helps.
4. **Of the extra inputs, only days-into-format helps** (−3.9 ×10⁻³ val). Rank, skill and card types are a wash.
5. **Pretraining on other sets with shared card features beats per-set card ids.** That holds everywhere,
   including on the best passed card. Day zero is far behind ALSA and miscalibrated (ECE 0.048). A few days of the
   set's own data is the practical starting point.
6. **It earns most where it matters.** Across 14 sets the gain over ALSA is +12–17% on the best passed card versus
   +7–12% overall. ALSA is 3–6× worse calibrated on top-3 passed cards, because being your best passed card usually
   means a thin pack. For a top-3 card, the model's value is separating 1% from 25%, not predicting "yes".
7. **Models go stale.** Over a weekly walk-forward on MSH, a week-1 model falls 0.011 behind a retrain by week 5,
   which is more than WheelNet's whole edge over the GBM. A 2-epoch weekly fine-tune stays within ~0.001.
8. **QA** (`wheelnet_qa.ipynb`): the sum of predicted probabilities matches the true returning-pack size at every
   pick (6.02 vs 6.00 … 0.99 vs 1.00), without the model being told it. Per-card calibration gaps are mostly < 0.02.

## 2. Pick prediction and table simulation (`notebooks/picknet.ipynb`)

PickNet is a softmax over exactly the cards in the pack, conditioned on a causal transformer over the drafter's
history. MSH test (10,292 drafts):

| model | NLL / pick | top-1 |
|---|---|---|
| `rating` (one number per card, a pick-order bot) | 1.324 | 0.497 |
| `pool` (rating + bilinear card × pool) | 0.954 | 0.644 |
| `picknet` (+ pack attention + history) | 0.826 | 0.682 |
| `picknet_big` (3.9M params) | 0.815 | 0.686 |
| **features-only, 7 other sets + MSH** | **0.810** | **0.686** |
| features-only, day zero | 1.411 | 0.508 |
| *DraftFM, MSH zero-shot (paper)* | 1.135 | 0.567 |

Findings:
1. **Context is worth 18 points over a pick order:** the pool +15 (stay in your colours), the pack +3, the history
   +1 (mostly in pack 3). History does help, contrary to puder's "+0.57% from pack contents".
2. **Strong drafters are more predictable** (Bronze 0.656 → Mythic 0.696), but only in *context*. The rating bot
   barely moves, so strong players agree on contextual picks, not on card ratings.
3. **The cheap models generalise across all 14 sets** (pool +15–22 pts over rating), including 13- and 15-card packs.
4. **Day zero is hard** (0.508), mostly at first picks, where only card quality matters. As with wheels, other sets
   are a strong pretraining signal.
5. **As a simulator, PickNet drafts like people.** Rating bots match marginal ATA and wheel rates better, but only
   56% of their pack-3 picks are in their two main colours (humans 81.5%, PickNet 80.2%). PickNet also gets P1P1 and
   the colour-pair mix closest. **All bots under-draft the best archetype** (MSH WU: 23% of humans, 17% of PickNet
   bots). Temperature 0.7 doesn't fix that and over-commits, so T = 1 stays.

## 3. Does the open lane pay? (`notebooks/open_lane.ipynb`)

Openness of a colour = how many good cards of that colour reached the drafter later than they usually last (picks
1–7 only, z-scored within draft week). The drafter's own picks never enter it. Results are on 15 sets:

1. **Signals are real and travel through your neighbour.** Openness correlates 0.22 between P1 and P3 (same upstream
   neighbour) and 0.09 between P1 and P2, on every set.
2. **Drafters follow signals, and better ones follow more.** +1 sd of P1 openness → +0.36 log-odds that the colour
   ends in the deck (bronze 0.30 → mythic 0.39).
3. **Open colours win.** +0.055 log-odds per game per sd with skill controlled, 15/15 sets. Bottom → top quintile is
   +3.2 points of game win rate.
4. **But it is supply, not the read.** P2 openness, which you cannot read before committing, pays as much as P1.
   Openness in colours you *didn't* take has no effect.
5. **Counterfactual (PickNet table simulator, 2,000 tables):**

   | seat-0 strategy | MSH vs following | FDN vs following |
   |---|---|---|
   | force a random pair | +0.02 ± 0.03 | −0.00 ± 0.02 |
   | force the best pair | **+1.30** | **+0.48** |
   | read P1 picks 1–5, then lock the most open pair | −0.42 | −0.26 |

   **Committing early costs nothing, and which pair matters far more than when you commit.** Pure signal-chasing
   loses. Caveats: the bots under-draft WU, so a WU forcer faces less competition than at a real table; the deck
   score is mean GIH WR of the top 23 (no curve or synergy); and the bots imitate 17Lands users, not strong players.

## 4. Deck strength (`notebooks/deckstrength.ipynb`)

Target: the run's record as p^W(1−p)^L, which is valid under the 7-win / 3-loss stopping rule. Metric: Spearman of
predicted p with the run's W/(W+L) on test drafts. **Noise ceiling:** Bo1 runs are ≤ 9 games, so even a perfect model
at the spread these models predict (sd ≈ 6 pts) scores only ~0.26–0.28.

Mean Spearman over 12 sets (test drafts with a registered deck):

| model | Spearman |
|---|---|
| mean GIH WR of top-23 pool cards (heuristic) | 0.183 |
| linear, final pool | 0.212 |
| linear, **registered deck** | 0.244 |
| + deck structure (lands, curve, creatures, colours, splash) | 0.259 |
| LightGBM, deck counts + structure + rank | 0.271 |
| game-level FM + deck features + affinity (skill-adjusted) | 0.268 |
| **LightGBM + deck features + affinity, logit-averaged with the game FM** | **0.279** |

Findings:
1. **Use the registered deck, not the pool** (+0.032, 12/12 sets). It encodes build choices: colours, playables and
   land count. The pick sequence adds nothing on top.
2. **DraftEncoder reproduces the paper** (0.19–0.24 on the pool), but **never beats a linear model on the same
   input** on one set. Its gap over the paper's own baseline comes from that baseline: an order-dependent
   one-head-per-pick LR, which scores 0.04–0.06 below a bag of cards. The transformer only edges linear on the
   largest sets (FDN, BLB).
3. **Build effects (MSH, observational):** two colours beat three by ~0.23 logit, each splash costs ~0.07, 16 lands
   beat 17, and more creatures help. Across sets the largest is a shortfall of coloured sources (−0.065 per sd).
4. **The game-level model and draft-level LightGBM are complementary.** LightGBM finds build effects from records,
   and the game model adds card values freed of the skill confound. Stacking (boosting from cross-fitted game logits)
   is no better than averaging.
5. **Traps:** win rate of cards *cast* looks great per game (AUC 0.84) but rates decks badly, because winners cast
   more. Rank lowers p at a fixed deck because it sets the opponents, so don't read it as skill. The 17Lands skill
   bucket is the strongest single input but leaks the outcome.

## 5. Synergy: eight rounds (`deck_synergy.ipynb` → `deck_synergy_v2.ipynb` → `deck_synergy_v3.ipynb`)

All rounds use game-level models: P(win) = σ(card power + pair terms + context), where the context is play/draw,
mulligans, both ranks, skill, experience and record. The pair terms are a low-rank factorization machine (FM). Gains
are paired test-game log loss ×1000; negative is better.

| round | question | answer |
|---|---|---|
| 1 (MSH) | Do FM pair terms help? | 9/12 sets slightly, but on MSH pairs **don't replicate** (split-half r = 0.09 vs 0.79 for card power), and they mostly learn the mana base. |
| 2 | Remove the mana base: add deck features, nonland-only pairs. Use the pick model's affinity as a prior. | Pairs now replicate on 11/12 sets (r 0.10–0.31). **Human drafting affinity as one column beats the whole FM** (−1.10 vs −1.02). It isn't plain co-occurrence or colour. |
| 3 | Richer affinity (profile, per-card fit)? | Round 2 was **under-trained**. LR drops fixed it (−1.20 → −1.40). The mean affinity carries all the signal, and per-card fit is real but tiny. |
| 4 | Pair-term variants, properly trained? | **Saturated.** Every variant adds −0.14 to −0.18 over affinity alone. Rank 4 is enough, and lands in pairs hurt. |
| 5 | Pairs from cards **seen together** in a game? | Very repeatable (split-half r ≈ 0.66 vs 0.13). Payoff × enabler hubs (Potioner's Trove + instants, Heaped Harvest + Food). They add ~nothing to *deck* prediction. |
| 6 | Is seen-together synergy real on held-out games? | **−2.42, 12/12 significant**, 13× the deck FM. A placebo (pairs among unseen cards) gives −0.08. |
| 7 | Final tables | `results/<SET>/deck_synergy_v7/{seen_pairs,card_hubs}.csv`. Hubs are ranked by partner spread, because own value and mean pair value trade off (r ≈ −0.6). |
| 8 (in progress, SOS only, provisional) | How much of seen-together synergy is partner-specific, and is it causal? | Centring each card's pairs to mean zero removes ~70% of the gain (−1.62 → −0.49). A per-card slope on "other spells seen" looked huge (−4.8), but "seen" is partly an outcome of play: the top slopes are draw/recursion spells. On the **opening hand** (dealt before play), the first result is only −0.17 ± 0.25. Treat rounds 5–7's "real in-game synergy" as unconfirmed until the 12-set hand-only run is in. |

Other things the synergy notebooks settled:
- **For Sealed, use draft data.** A draft-trained model predicts Sealed games far better than a Sealed-trained one
  (0.625 vs 0.646). Card power is nearly identical across formats (multitask r = 0.98).
- **A deck transformer pooled over 12 sets learns little** beyond per-set card power + deck features (−0.09). From
  card features alone it rediscovers ~90% of the hand-built deck features.

## 6. Next-game model (`notebooks/game_model.ipynb`)

This predicts one game, not the run: about 19% of runs rebuild their deck mid-run, and play/draw and ranks are
per game.
- **The record is matchmaking.** At a fixed deck and player, each prior win costs ~2 pts and each loss gives back
  ~2.5–3 pts. The raw win rate is flat across records because better decks reach 6-0 while facing tougher
  opponents. Adding the record is worth about a third of what the whole deck adds.
- Being on the play is worth +0.21 log-odds (~5 pts). Each extra mulligan costs −0.63, which is only usable in-game.
- Card weights barely move with the record added (r = 0.994, stretched ~8%).
- A live version needs skill from *earlier* events only. The 17Lands bucket leaks.

## Cross-cutting lessons

1. **Share card features across sets.** Features-only models pretrained on other sets are the best wheel model *and*
   the best pick model, ahead of per-set learned card ids. On one set alone, transformers rarely beat linear or GBM
   (deck strength, synergy).
2. **The binding constraint is often noise, not the model.** Deck strength sits near the Bo1 ceiling. Synergy that
   replicates perfectly still barely moves deck predictions.
3. **Human picks are a free label for card relationships.** Pick-model affinity was the best single synergy feature.
4. **Train properly before comparing.** Round 2's ranking was distorted by early stopping at the first plateau
   (`synergy._train(lr_drops=...)` now handles it).
5. **Watch the confounds:** skill (the 17Lands bucket leaks), rank (sets opponents), record (matchmaking), cards cast
   (winners cast more).

## Dead ends (archived)

These were run, reported and moved to `scripts/archive/` and `notebooks/archive/`. Their results stay in `results/`.

| what | why archived |
|---|---|
| `notebooks/archive/results.ipynb` | First-pass MSH summary. Its numbers (ensemble +7.3%) predate WheelNet v2 and the multiset model. Superseded by §1 and `wheelnet.ipynb`. |
| `notebooks/archive/deck_synergy.ipynb` | Round 1 (MSH only). Superseded by v2/v3. Kept for the Sealed / multitask results. |
| `scripts/archive/deck_features.py` | Early deck-feature probe. Replaced by `cardinfo.deck_features` in round 2. |
| `scripts/archive/deck_synergy_multitask.py` | Draft + Sealed multitask: +0.0013 on Sealed (not significant), costs draft. |
| `scripts/archive/deck_multiset.py` | Cross-set deck transformer: −0.09 ×1000, an order of magnitude below deck features. |
| `scripts/archive/ensemble_de_lgbm.py` | DraftEncoder + LightGBM ensemble: logits correlate 0.86–0.90, gain ≤ 0.004. |
| `scripts/archive/lgbm_ablation.py` | One-off: showed LightGBM needs only deck counts + rank. |
| `scripts/archive/deck_synergy_v3_blend.py` | Round 3 blend (0.278). Superseded by `deck_synergy_v4_draft.py` (0.279). |

Other ideas that didn't help, without their own script: GIH WR prior and archetype identity in the deck model, a
non-leaky pick-based skill proxy (adds nothing once the deck is known), per-card play/draw offsets, deeper WheelNet
(6+4 layers), rank and skill inputs to WheelNet, sampling temperature 0.7 for bots.

## Open questions

- Finish round 8 on all 12 sets, on the opening hand. If seen-together synergy doesn't survive there, the round 6–7 pair tables are mostly card-draw effects, and deck-level synergy (small, round 4) is the honest answer.
- A counterfactual seat that is strong rather than imitative (skill-conditioned bots), and a table where several
  seats force.
- Use simulator tables to get wheel labels at all 8 seats.
- Out-of-fold card target encoding for the GBM, and the per-seat passing direction.
- A live overlay: P(wheel) for each card in the current pack, fine-tuned weekly.

## Where things are

| topic | notebook | main scripts |
|---|---|---|
| wheel prediction | `wheelnet.ipynb`, `wheelnet_qa.ipynb` | `build_dataset`, `train`, `train_nn`, `train_wheelnet_multiset`, `wheel_breakdown`, `temporal` |
| pick model + simulator | `picknet.ipynb` | `train_picknet`, `train_picknet_multiset`, `simulate_draft`, `card_text_embeddings` |
| open lane | `open_lane.ipynb` | `open_lane`, `force_vs_follow` |
| deck strength | `deckstrength.ipynb` | `train_deckstrength`, `train_draftencoder`, `improve_deck`, `paper_baselines` |
| synergy | `deck_synergy_v2.ipynb`, `deck_synergy_v3.ipynb` | `deck_synergy`, `deck_synergy_v2` … `v8`, `affinity_checks` |
| next game | `game_model.ipynb` | `deck_synergy` (model), `src/openlane/deckstrength.py` |
