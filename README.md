# mtg-open-lane

**Will this card wheel?** A model that predicts whether a card you pass in a Magic draft will come
back to you 8 picks later, based on the pack contents and the signals you have seen so far, and an
analysis of whether reading an open lane pays. Trained on 17Lands human draft data. See
[project-plan.md](project-plan.md) for background.

**[WRITEUP.md](WRITEUP.md) is the one-page summary of the whole project**, including the parts that now
live in their own repos:

| repo | what it is |
|---|---|
| [mtg-data](../mtg-data) | shared 17Lands download, loaders and encodings (`mtgdata`); the data lives there |
| [picknet](../picknet) | PickNet, the pick-prediction transformer and table simulator |
| [mtg-deck-strength](../mtg-deck-strength) | deck strength, DraftEncoder replication, card synergy |
| [mtg-bombs](../mtg-bombs) | bombs analysis and blog post |
| [mtg-draft-sim-site](../mtg-draft-sim-site) | browser draft simulator against PickNet bots |

Notebooks: `wheel` (wheel models) and `drafting` (pick-model summary and the open-lane analysis).
Superseded notebooks live in `notebooks/archive/`.

Data: [17Lands public datasets](https://www.17lands.com/public_datasets). Please attribute 17Lands
and check their [usage guidelines](https://www.17lands.com/usage_guidelines).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../mtg-data -e ../picknet -r requirements.txt   # picknet: force_vs_follow's table simulator
python scripts/build_dataset.py MSH      # download + labels + features (~75s)
python scripts/train.py MSH              # baselines + LightGBM ablations
python scripts/train_nn.py MSH           # WheelNet transformer (MPS/CPU), 8 epochs
python scripts/train_wheelnet_multiset.py MSH 4 20000 with_target=1  # wheel model on 8 sets (~50 min)
python scripts/wheel_breakdown.py MSH     # wheel accuracy by the card's rank in the pack (~1 min)
python scripts/open_lane.py MSH           # lane openness: persistence, following, payoff (~1 min)
python scripts/force_vs_follow.py MSH 2000  # follow vs force in the table simulator (~10 min)
python scripts/open_lane_report.py        # markdown tables for the open-lane / breakdown sections
```

Data downloads into `../mtg-data/data/` (or `$MTG_DATA_DIR`) on first use. The feature pipeline assumes
14-card packs with all 42 picks present, so it only works as-is on the "14, complete" sets (see mtg-data's
README for the inventory).

## How the labels work

Packs have 14 cards and there are 8 drafters, so the pack you see at pick `p` (p = 0..5) returns at
pick `p+8`. 17Lands records the pack contents at both picks. Take the pack at `p`, subtract your
pick, and the pack at `p+8` must be a subset of what's left. This check holds on every row except
one malformed MSH draft. One example = one distinct card you passed; label = at least one copy came back.

MSH Premier: 68,609 drafts, split chronologically 70/15/15. That gives 9.1M / 1.9M / 1.9M examples
with a base wheel rate of 33.3%. Card statistics (ATA, ALSA, per-pick wheel rates) are fit on
train only.

## Models

| model | what it sees |
|---|---|
| `prior_by_pick` | pick number only |
| `alsa_ata_logreg` | card ALSA + ATA, with a separate curve per pick (**the baseline from the plan**) |
| `card_x_pick_rate` | empirical wheel rate for this card at this pick (strongest card-only lookup) |
| `lgbm_card` | card identity + static stats |
| `lgbm_card_pack` | + what else is in the pack being passed (how many better cards protect it) |
| `lgbm_card_signals` | + table signals: colours arriving late this round, earlier rounds' wheel surprises, own pool, rank |
| `lgbm_all` | card + pack + signals |
| `wheelnet` | transformer over the **raw** draft history (every pack seen, every pick, what the other 7 took from packs that came back), plus card colour/rarity/mana value; no hand-built features |

## Results (MSH PremierDraft, held-out last 15% of drafts)

<!-- results -->
| model | logloss | Brier | AUC | ECE | logloss vs ALSA baseline | logloss P1 / P2 / P3 |
|---|---|---|---|---|---|---|
| `prior_by_pick` | 0.6067 | 0.2099 | 0.6389 | 0.0000 | -31.1% | 0.607 / 0.607 / 0.607 |
| `alsa_ata_logreg` | 0.4629 | 0.1517 | 0.8309 | 0.0073 | +0.0% | 0.453 / 0.466 / 0.470 |
| `card_x_pick_rate` | 0.4602 | 0.1511 | 0.8328 | 0.0088 | +0.6% | 0.450 / 0.463 / 0.468 |
| `lgbm_card` | 0.4588 | 0.1506 | 0.8339 | 0.0084 | +0.9% | 0.447 / 0.462 / 0.467 |
| `lgbm_card_pack` | 0.4397 | 0.1444 | 0.8502 | 0.0076 | +5.0% | 0.428 / 0.443 / 0.448 |
| `lgbm_card_signals` | 0.4496 | 0.1474 | 0.8418 | 0.0057 | +2.9% | 0.446 / 0.454 / 0.449 |
| `lgbm_all` | 0.4315 | 0.1414 | 0.8566 | 0.0052 | +6.8% | 0.428 / 0.435 / 0.432 |
| `wheelnet` | 0.4249 | 0.1390 | 0.8618 | 0.0024 | +8.2% | 0.424 / 0.427 / 0.424 |
| `ensemble` | 0.4258 | 0.1393 | 0.8612 | 0.0054 | +8.0% | 0.425 / 0.428 / 0.424 |
<!-- /results -->

### What we learned (first pass)

1. **Wheel prediction beats ALSA.** The best model cuts log loss 8.2% relative to the ALSA/ATA
   baseline, and AUC rises from 0.831 to 0.862. The per-card-by-pick lookup is only +0.6%, so
   card identity alone is nearly exhausted by ALSA.
2. **Pack context is the biggest single lever (+5.0%).** The top features are how many cards in
   the passed pack are usually taken before this one (`n_better`) and the pack's total take
   propensity. This is "the other 7 drafters have better things to take", read from what is in the pack.
3. **Signals are real and grow over the draft.** Signals alone add +2.9%, and they stack with
   pack context (+5.0% → +6.8%). The gain over the card-only lookup rises from about 3–4% per pick
   in pack 1 to about 6–7% in pack 3. That fits the idea that earlier rounds' wheels reveal the
   table's colours: `resid_prior_card` (did this colour wheel more than expected earlier?) is the
   #4 feature by gain. This runs against the "models don't read signals" result from puder, though
   predicting other drafters' picks is a different task from picking for yourself.
4. **The raw-history transformer now beats the hand-built features.** The first WheelNet tied the
   GBM (0.4316 vs 0.4315). Three changes took it to 0.4249 (val 0.4293 → 0.4226):
   - Each passed card cross-attends to the history up to its pick, instead of reading one pooled
     state vector (val 0.4293 → 0.4243, with the next change).
   - Explicit "taken by the other 7" tokens: at picks 9–14, the passed pack minus the returning pack.
   - Dropout 0.2 over 8 epochs, since it overfit after epoch 5 at 0.1 (→ 0.4235), and card
     colour/rarity/mana value added to the learned embeddings (→ 0.4227).

   More depth did not help: 6+4 layers scored 0.4228 and trained 35% slower. Averaging with
   `lgbm_all` no longer helps either (0.4258), so the GBM's features appear to be subsumed.

### One wheel model for every set (`notebooks/wheel.ipynb` §7)

The pick model's best result came from dropping per-card ids for card features shared across sets (colour,
rarity, mana value, types, Scryfall rules features, a frozen rules-text embedding). The same WheelNet with that
features-only card table (`ids=False`), trained on 20k drafts from each of the 7 other 14-card sets, MSH test:

| model | logloss | AUC | ECE | vs ALSA baseline |
|---|---|---|---|---|
| `prior_by_pick` (no card knowledge) | 0.6067 | 0.639 | 0.000 | −31.1% |
| **day zero**: 7 other sets, never saw MSH | 0.5245 | 0.776 | 0.048 | −13.3% |
| `alsa_ata_logreg` (needs the set's own data) | 0.4629 | 0.831 | 0.007 | 0 |
| per-set WheelNet (learned card ids), all inputs / best inputs (`wheelnet_time`) | 0.4249 / 0.4218 | 0.862 / 0.864 | 0.002 / 0.007 | +8.2% / +8.9% |
| **7 other sets + MSH's training drafts** (4.1M params, 4 epochs) | **0.4188** | **0.866** | **0.003** | **+9.5%** |

1. **Pretraining on other sets gives the best wheel model**, better than per-set WheelNet in every slice of the
   breakdown below (best passed card: 0.1306 vs 0.1341). It is the same result as the pick model: card features
   shared across sets beat memorising one set's cards.
2. **Day zero is far behind ALSA.** With no MSH drafts, the model knows how tables behave but not how good each new
   card is. It closes 57% of the gap from the pick prior to the baseline, but ALSA after a few days of
   data is much better, and the zero-shot model is miscalibrated (ECE 0.048). A few days of the set's own data is
   the practical starting point.

### Next ideas

- Out-of-fold card target encoding. Also add the per-seat structure: packs pass left in P1/P3 and
  right in P2, so P2's upstream drafters are the P1/P3 downstream ones.
- A draft-overlay-style demo: given a live pack, show P(wheel) for each card.

## Where wheel prediction matters: the cards you'd want back (`notebooks/wheel.ipynb` §8)

The useful product question is "will my *second-best* card come back?", not "will the chaff wheel?". Each passed
card is ranked within its pack by train-set ATA (rank 1 = the best card you passed). Log-loss gain of the per-set
WheelNet over the ALSA/ATA baseline, by rank, on the test drafts of all 14 sets with a WheelNet:

| set | ALSA → WheelNet, all passed cards | best passed card (rank 1) | rank 2 | rank 3 | ECE rank 1–3, ALSA / WheelNet |
|---|---|---|---|---|---|
| BLB | +7.4% | +11.6% | +9.2% | +8.3% | 0.017 / 0.005 |
| DFT | +8.0% | +13.7% | +10.6% | +10.0% | 0.019 / 0.005 |
| DSK | +8.4% | +14.2% | +10.5% | +9.7% | 0.018 / 0.004 |
| ECL | +8.2% | +12.8% | +9.7% | +8.9% | 0.018 / 0.003 |
| FDN | +8.9% | +16.6% | +11.4% | +10.1% | 0.021 / 0.004 |
| FIN | +8.3% | +14.4% | +10.5% | +9.6% | 0.025 / 0.007 |
| LCI | +8.8% | +14.5% | +10.8% | +9.8% | 0.029 / 0.009 |
| MKM | +6.8% | +12.8% | +9.5% | +8.2% | 0.021 / 0.004 |
| MSH | +8.2% | +14.1% | +10.7% | +9.7% | 0.024 / 0.004 |
| OTJ | +9.3% | +16.7% | +12.7% | +10.5% | 0.026 / 0.008 |
| SOS | +11.5% | +14.8% | +11.8% | +11.4% | 0.021 / 0.003 |
| TDM | +8.7% | +14.5% | +10.2% | +9.2% | 0.021 / 0.003 |
| TLA | +8.3% | +13.4% | +10.4% | +10.3% | 0.021 / 0.002 |
| TMT | +9.8% | +16.1% | +11.1% | +10.5% | 0.025 / 0.005 |

What we learned:
1. **The model earns most where it matters.** On every set the gain over ALSA is largest for the best passed card
   (+12–17%, vs +7–12% over all cards) and falls with rank. The best passed cards are the ones whose fate depends on
   the table, not the card.
2. **ALSA is overconfident about your good cards.** It is calibrated on average but 3–6× worse calibrated on the
   top three passed cards (ECE 0.017–0.029 vs WheelNet's 0.002–0.009). On MSH, when ALSA gives the 2nd-best passed
   card a 30–50% chance to wheel it comes back 15% of the time; WheelNet's 30–50% bucket comes back 32% of the
   time. ALSA is a per-card average over every pack the card appears in. Being the best card *you* pass usually means
   the pack is thin, so there is little else to absorb the other seven picks, and the card's chance of lasting is
   lower than its average.
3. **"Will it wheel?" for a top-3 card is almost always "no".** WheelNet puts ≥ 50% on under 1% of them. Its value
   is separating 1% from 25%: on MSH a third of rank-2 cards get < 2% (and wheel 1.1%), while the 9% it puts above
   20% wheel 26% of the time.

## Does the open lane pay? (`notebooks/drafting.ipynb` part 2)

The repo's namesake question, on all 15 sets. **Openness** of colour *j* in pack *k* is how many good cards of
that colour reached the drafter later than they usually last: Σ over picks 1–7 of the pack's cards' colour share ×
max(0, pick − ALSA). Picks 8+ are excluded (those packs went round the whole table). It is z-scored per (pack,
colour) within each draft week, so set-wide shifts in colour popularity cancel. The drafter's own picks never enter
it. Main colours are the top two by playability-weighted picks. Payoff is a binomial logit on the event record
(p^W (1−p)^L) with colour-pair fixed effects, rank and the 17Lands skill bucket; per sd of openness, averaged over
the two main colours.

| set | drafts | corr P1–P3 (same neighbour) | corr P1–P2 | follow slope (bronze → mythic) | payoff / sd, + skill | win rate, P1-openness quintile 1 → 5 | most-open P1 pair: in / not |
|---|---|---|---|---|---|---|---|
| BLB | 154,585 | 0.218 | 0.084 | 0.33 (0.30 → 0.36) | +0.056 ± 0.003 | 0.531 → 0.562 | 0.559 / 0.544 |
| DFT | 141,166 | 0.246 | 0.102 | 0.37 (0.32 → 0.39) | +0.060 ± 0.003 | 0.532 → 0.566 | 0.563 / 0.548 |
| DSK | 167,672 | 0.236 | 0.087 | 0.37 (0.36 → 0.39) | +0.057 ± 0.003 | 0.530 → 0.562 | 0.555 / 0.544 |
| ECL | 99,289 | 0.234 | 0.114 | 0.35 (0.32 → 0.37) | +0.065 ± 0.004 | 0.527 → 0.565 | 0.562 / 0.543 |
| EOE | 104,377 | 0.259 | 0.096 | 0.41 (0.37 → 0.44) | +0.056 ± 0.004 | 0.535 → 0.566 | 0.560 / 0.547 |
| FDN | 130,935 | 0.238 | 0.088 | 0.37 (0.32 → 0.43) | +0.063 ± 0.003 | 0.529 → 0.565 | 0.560 / 0.544 |
| FIN | 140,237 | 0.225 | 0.090 | 0.36 (0.31 → 0.40) | +0.059 ± 0.003 | 0.534 → 0.570 | 0.564 / 0.551 |
| LCI | 134,970 | 0.177 | 0.062 | 0.33 (0.28 → 0.37) | +0.052 ± 0.003 | 0.541 → 0.570 | 0.566 / 0.552 |
| MKM | 157,553 | 0.203 | 0.068 | 0.34 (0.29 → 0.38) | +0.049 ± 0.003 | 0.538 → 0.566 | 0.562 / 0.548 |
| MSH | 68,609 | 0.234 | 0.106 | 0.36 (0.33 → 0.39) | +0.057 ± 0.005 | 0.540 → 0.573 | 0.572 / 0.554 |
| OTJ | 195,312 | 0.179 | 0.056 | 0.33 (0.28 → 0.36) | +0.050 ± 0.003 | 0.527 → 0.556 | 0.551 / 0.540 |
| SOS | 107,459 | 0.234 | 0.120 | 0.37 (0.32 → 0.40) | +0.057 ± 0.004 | 0.534 → 0.569 | 0.565 / 0.547 |
| TDM | 101,323 | 0.183 | 0.065 | 0.35 (0.31 → 0.36) | +0.052 ± 0.004 | 0.533 → 0.564 | 0.559 / 0.544 |
| TLA | 102,793 | 0.229 | 0.097 | 0.37 (0.33 → 0.39) | +0.049 ± 0.004 | 0.544 → 0.569 | 0.564 / 0.555 |
| TMT | 32,151 | 0.236 | 0.100 | 0.35 (0.29 → 0.37) | +0.047 ± 0.007 | 0.557 → 0.585 | 0.579 / 0.561 |
| **mean** | | 0.222 | 0.089 | 0.36 | +0.055 | +3.2 pts | |

What we learned:
1. **Signals are real and travel through your neighbour.** Packs pass left in P1 and P3 and right in P2, so P1 and
   P3 share an upstream neighbour. A colour's P1 openness correlates 0.22 with its P3 openness and 0.09 with P2, on
   every set. P1 tells you about P3, as the folk wisdom says.
2. **Drafters follow signals, and better drafters follow more.** +1 sd of P1 openness raises the log-odds that a
   colour ends up in the deck by 0.36. The slope rises with rank (bronze 0.30 → mythic 0.39, mean over sets) and with the
   17Lands win-rate bucket (0.32 → 0.39, on every set). Still, only 17% end in the most open P1 pair (10% by chance).
3. **Being in the open lane wins games.** +1 sd of openness in your colours is worth +0.055 log-odds per game with
   skill controlled (±0.003–0.007 per set, 15/15 sets); bottom to top quintile is +3.2 points of game win rate
   (e.g. MSH 54.0% → 57.3%). Drafters who ended in the most open P1 pair win 1–2 points more.
4. **But the payoff is supply, not the read itself.** Openness in P2, which you cannot read before committing,
   pays as much as P1 openness (+0.055 vs +0.040; P3 +0.061, all three fitted together). And holding your colours'
   openness fixed, openness in the colours you *didn't* take has no effect (mean −0.002, negative on only 9/15
   sets). So more good cards in your colours win games. That is what reading signals is for, but this data
   doesn't separate the value of the decision from the luck of the seat. That needs a counterfactual, e.g. the
   PickNet table simulator with a signal-blind seat.

**Counterfactual: follow or force?** (`scripts/force_vs_follow.py`, MSH and FDN). The observational payoff can't separate
the read from the luck of the seat, so this tests it in the PickNet table simulator: 2,000 tables of real opened
packs, seats 1–7 always the human-like bot, seat 0 plays a strategy, with the same tables and random stream for every
strategy. Pools are scored as decks: the best colour pair's top 23 cards by training-game GIH WR (missing slots get
a 10th-percentile card).

| seat 0 strategy | MSH deck score | vs following | FDN deck score | vs following |
|---|---|---|---|---|
| follow: draft like a human (the bot reads signals as people do) | 57.36 | — | 55.78 | — |
| force a random pair (sample from the same model, only cards that fit) | 57.39 | +0.02 ± 0.03 | 55.78 | -0.00 ± 0.02 |
| force the best pair (MSH WU, FDN WB) | 58.67 | +1.30 | 56.26 | +0.48 |
| read P1 picks 1–5 for the most open pair, then lock it | 56.94 | -0.42 ± 0.03 | 55.52 | -0.26 ± 0.02 |

Deck score = mean GIH WR (%) of the best pair's top 23. Forcing, by pair, MSH: WU 58.7 · WG 58.3 · UG 58.1 · WB 57.8 · UB 57.6 · BG 57.3 · WR 57.0 · UR 56.9 · RG 56.6 · BR 55.8. FDN: WB 56.3 · WU 56.2 · UB 56.1 · BR 55.9 · WR 55.9 · WG 55.8 · BG 55.6 · UG 55.4 · UR 55.4 · RG 55.4.

5. **In simulation, forcing a random pair costs nothing, and forcing the strongest pair wins.** On both sets,
   committing to a random pair from P1P1 ties drafting like a human (±0.03). Which pair matters much more than
   when you commit: 2.9 points separate the best and worst pair to force on MSH, and 0.9 on FDN, a flatter
   format. Locking in whatever looks open after five picks, ignoring the cards already taken, is worse than both
   (−0.4 MSH, −0.3 FDN). Human-style drafting balances card quality, pool and
   signals, and pure signal-chasing loses that.
   Caveats: the bots under-draft WU (17% of bot decks vs 23% of real drafters), so a WU forcer faces less
   competition than at a real table; the deck score ignores curve and synergy; and every seat is an imitation of
   17Lands users, not a strong drafter.

## Layout

```
src/openlane/features.py  wheel labels, train-only card stats, feature groups
src/openlane/nn.py        WheelNet and its dataset
src/openlane/evaluate.py  logloss / Brier / AUC / ECE
scripts/                  build_dataset.py, train.py, train_nn.py, train_wheelnet_multiset.py, temporal.py,
                          wheel_breakdown.py, open_lane.py, force_vs_follow.py, reports
results/<SET>.<EVENT>/    results.json, predictions, models (gitignored)
```

Data loading and the shared encodings come from `mtgdata` (../mtg-data). `force_vs_follow.py` drafts with
saved PickNet models from ../picknet.

## Caveats

- We only see one seat, and 17Lands users skew above average. The model describes what happens
  around a 17Lands user, not the general Arena population.
- `wheel_cp` and the other card stats are fit on the same train rows the GBM learns from. The
  counts are large (about 27k examples per card), so leakage is small, but out-of-fold encoding
  would be cleaner.
