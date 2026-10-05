# Literature notes

Survey done 2026-10-06. Every entry below has a PDF in this folder unless it's marked *(no PDF)*. Numbers come
from the PDFs or abstracts. Anything I couldn't check is marked **[unverified]**.

## 1. MTG draft pick prediction

**Ward, Mills, Brooks, Khakhalin, Troha. "AI solutions for drafting in Magic: the Gathering." IEEE CoG 2021.**
[arXiv 2009.00655](https://arxiv.org/abs/2009.00655) · `2020_ward_ai_solutions_drafting_draftsim.pdf`
- Task and data: predict human picks on about 108k Draftsim M19 drafts. These are simulated drafts against bots, not 17Lands data.
- Models: random, rarity, expert heuristic, Naive Bayes and an MLP. The MLP takes a pool-count vector and scores all 265 card ids, then masks to the pack.
- Result: the MLP is best at 48.67% top-1 (heuristic 44.5%, Bayes 43.4%). Accuracy is highest at the start and end of each pack.
- Relevance: this is the original benchmark that PickNet's 68.6% is compared against. The per-pick U shape it found matches ours.

**Bertram, Fürnkranz, Müller. "Predicting Human Card Selection in MTG with Contextual Preference Ranking." IEEE CoG 2021.**
[arXiv 2105.11864](https://arxiv.org/abs/2105.11864) · `2021_bertram_predicting_human_card_selection.pdf`
- Model: a Siamese network that embeds the pool (anchor) and each candidate card. It is trained with triplet loss so the picked card lies closer to the pool than the passed ones.
- Data: Draftsim M19, card ids only. Result: 53.7% top-1 against Ward's 48.67%.
- Relevance: the card-to-pool distance in this embedding is another way to measure "affinity". It could be compared with our pick-model affinity S_ij.

**Bertram, Fürnkranz, Müller. "A Comparison of Contextual and Non-Contextual Preference Ranking for Set Addition Problems." SubSetML workshop 2021.**
[arXiv 2107.04438](https://arxiv.org/abs/2107.04438) · `2021_bertram_contextual_preference_ranking_set_addition.pdf`
- Compares a "twin" network, which scores the two sets you get after each addition, with a "triplet" network, which scores what the candidate adds to the context. Same Draftsim M19 data.
- Result: the triplet (contextual) design wins. Both beat Ward's results; the paper claims ">56% higher accuracy". Card ids only.
- Relevance: frames drafting as set addition, which is exactly the question of what a card adds to a deck. It's the conceptual basis for a "marginal value of a card given the deck" head.

**Bertram, Fürnkranz, Müller. "Learning With Generalised Card Representations for MTG." IEEE CoG 2024 (best student paper).**
[arXiv 2407.05879](https://arxiv.org/abs/2407.05879) · `2024_generalised_card_representations.pdf`
- Same contextual preference ranking model, but cards enter as numeric, nominal and text features, card images, and 17Lands usage statistics instead of ids. Data: 17Lands, many sets.
- Result: the choice of representation hardly matters for known cards. On completely unseen cards the model gets 55% top-1. Usage stats (ALSA, GIH WR and so on) do most of the work for transfer.
- Relevance: confirms what our multiset PickNet and WheelNet found, that shared card features generalise across sets. Their features include post-release statistics, so their numbers are not day-zero numbers.

**Bertram, Fürnkranz, Müller. "Contrastive Learning of Preferences with a Contextual InfoNCE Loss." 2024.**
[arXiv 2407.05898](https://arxiv.org/abs/2407.05898) · `2024_bertram_contextual_infonce_preferences.pdf` (new)
- Pick prediction framed as CLIP-style contrastive learning between a pool embedding and card embeddings. The batch is built so that each pool is compared only against the cards that were actually in its pack.
- Data: 17Lands NEO, 5.69M samples. Top-1: adapted InfoNCE **68.8%**, sigmoid InfoNCE 68.1%, triplet with random mining 67.2%, plain CLIP InfoNCE 54.2%.
- Relevance: plain InfoNCE fails because in-batch negatives aren't valid alternatives, which is the same reason our softmax only covers cards in the pack. The useful part for us is the pool-card embedding space. Its cosine similarity is a ready-made pairwise affinity that could be tested against S_ij as a synergy prior.

**Bertram, Fürnkranz, Müller. "Drafting MTG With High-Dimensional Card Embeddings." IEEE Trans. on Games 18(3), Sept 2026.**
[DOI 10.1109/TG.2025.3646029](https://doi.org/10.1109/TG.2025.3646029) (open access, CC BY) · `2026_bertram_drafting_high_dimensional_card_embeddings.pdf` (new)
- This is the journal version of the 2024 papers. Siamese contextual preference ranking on about 125M 17Lands picks, every set up to MH3, with a random 80/20 split within each set (not chronological).
- Result: 45% zero-shot top-1 on BRO when BRO is held out entirely and no usage stats are used. Training on more sets raises BRO accuracy from 36.95% to 45.66% and NEO in-set accuracy from 66.34% to 67.14%. The embeddings encode deck size and card strength.
- Relevance: an independent day-zero reference. Our features-only PickNet scores 0.508 on MSH and DraftFM about 0.51–0.60. The gain from adding more sets is modest but holds for both in-set and unseen sets, as ours did.

**Bertram. "UrzaGPT: LoRA-Tuned LLMs for Card Selection in CCGs." 2025.**
[arXiv 2508.08382](https://arxiv.org/abs/2508.08382) · `2025_urzagpt_lora_card_selection.pdf`
- Pack and pool are written out as text, and an open 7–8B LLM is fine-tuned with LoRA on 17Lands NEO picks (about 1M picks, according to DraftFM's description).
- Result: 66.2% top-1 after 10k steps. GPT-4o zero-shot gets 43%, and an untuned Llama-3-8B can't draft.
- Relevance: below PickNet (68.6%) and far more expensive. The only cheap idea worth borrowing is zero-shot LLM card grades as a day-zero prior feature.

**Ward, B. "DraftFM: A Foundation Model for Day-Zero Drafting in MTG." Aug 2026.**
[arXiv 2608.19568](https://arxiv.org/abs/2608.19568) · `2026_draftfm_day_zero_drafting.pdf` · code: [github.com/brianward92/draftfm](https://github.com/brianward92/draftfm)
- A discrete-choice softmax over the cards in the pack. Each card is a frozen 775-dim vector: 391 structured Scryfall features plus a 384-dim bge-small-en-v1.5 embedding of the type line and rules text, with the card name masked.
- The pool is summarised by 4 attention queries. Context covers position, format, skill and experience. The model has 1.6M parameters and was trained on 149M picks from 29 sets.
- Result: 50.8%, 60.4% and 56.7% top-1 on three sets held out entirely. It is overconfident everywhere, and expert P1P1s are the hardest picks to predict.
- It also published a sealed (hashed) HOB ranking about 36h before release, which agrees with experts about as well as experts agree with each other. The promised follow-up with realised outcomes: **[not found as of 2026-10-06]**.
- Relevance: our `card_text_embeddings.py` already copies its text pipeline (bge-small, CARDNAME masking). The idea we haven't used is skill conditioning, set to "high skill" at deploy time (see experiments).

## 2. Deck strength / win-rate prediction

**Rigaux, Kashima. "Predicting Drafted Deck Strength for MTG" (DraftEncoder). IEEE CoG 2026.**
[arXiv 2607.04782](https://arxiv.org/abs/2607.04782) · `2026_rigaux_draftencoder_deck_strength.pdf` · code: [github.com/akulen/MtGDraftEncoder](https://github.com/akulen/MtGDraftEncoder)
- Model: a set-context transformer (FiLM on rank) builds card embeddings, then a causal draft encoder predicts the run win rate after every pick. Cards come in as EmbeddingGemma text, optionally plus "meta" 17Lands win rates.
- Loss: the 7-win/3-loss negative-binomial likelihood.
- In-set result (BLB/FDN/TDM): Spearman about 0.2, roughly 2× the linear baseline, close to their noise ceiling of 0.23.
- Cross-set result (train on 11 sets, test on FDN): 0.088 without meta and 0.153 with meta. Fine-tuning on FDN's first 4 days gives 0.164 and 0.181. The heuristic gets 0.139.
- Relevance: already replicated in this repo (see README). Section 6 lists details from the code that aren't in the paper.

**Jakubik. "A Neural Network Approach to Hearthstone Win Rate Prediction." FedCSIS 2018 (AAIA'18 challenge).**
[annals-csis.org](https://annals-csis.org/Volume_15/drp/365.html) · `2018_jakubik_hearthstone_deck_win_rate_nn.pdf` (new)
- Task: predict win rates for 200 unseen 30-card decks × 4 AI pilots from 300k simulated games among 400 other decks.
- Model: deck win rate is fitted by maximising the log-likelihood of match results. Decks are encoded by a dual-task autoencoder that reconstructs the deck and decodes card properties. Ensembled with logistic regression. A top entry in the challenge.
- Caveat: the author admits to over-tuning on the leaderboard by forcing the mean win rate to 0.49. He also says a matchup-specific model (deck vs deck) would be the right next step.
- Relevance: the closest pre-MTG analogue of our deck model. Its "matchup" remark points at something this repo hasn't done (experiment 3).

**Zhang, Fontaine, Hoover, Nikolaidis. "Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding." GECCO 2022.**
[arXiv 2112.03534](https://arxiv.org/abs/2112.03534) · `2021_zhang_deep_surrogate_map_elites_hearthstone.pdf` (new)
- An online-trained neural surrogate predicts a deck's simulated win rate (and behaviour measures) from a bag-of-cards vector. The surrogate guides quality-diversity search over decks.
- Result: beats model-free MAP-Elites and offline surrogates. Auxiliary prediction targets didn't help. The authors guess that card attributes (cost, stats, text) would help, but didn't test it.
- Relevance: the same loop could be "deck model as surrogate → optimise the 23 cards + lands from a pool", an automatic deckbuilder judged against the registered deck. Weak evidence, because their surrogate predicts simulator outcomes, not human ones.

## 3. Set / team models and synergy

**Lee et al. "Set Transformer." ICML 2019.** [arXiv 1810.00825](https://arxiv.org/abs/1810.00825) · `2019_lee_set_transformer.pdf` (new)
- An attention-based, permutation-invariant set encoder: SAB/ISAB blocks with inducing points, plus attention pooling (PMA).
- They show attention in both the encoder and the pooling is needed to model interactions among set elements. Deep Sets-style sum pooling can't.
- Relevance: the standard architecture for "deck → value". Our deck transformer and the FM are special cases. Its use here is a reminder that a deck model with mean pooling can't represent synergy, so check how our deck transformer pools before concluding "no synergy". DraftFM cites it for its pool summariser.

**Lee, Hwang, Kim, Lee, Choo. "DraftRec: Personalized Draft Recommendation for Winning in MOBA Games." WWW 2022.**
[arXiv 2204.12750](https://arxiv.org/abs/2204.12750) · `2022_lee_draftrec_moba_draft_recommendation.pdf` (new)
- A hierarchical transformer. One network models each player's champion preferences from their history; a second models the interactions within the match.
- Trained jointly to recommend picks and predict the match outcome on 280k LoL and 50k Dota 2 matches. State of the art on both tasks at the time.
- Relevance: one model trained on both pick prediction and outcome prediction. Our pick and deck models are trained separately and only linked through the affinity column. The personalisation half doesn't apply, because 17Lands has no player ids.

**Ye, Mayo, Cuthbert, Head. "Learning Team Synergy from Team Composition with a Siamese Transformer." FLAIRS-39, 2026.**
[journals.flvc.org](https://journals.flvc.org/FLAIRS/article/view/141762) · `2026_ye_siamese_transformer_team_synergy.pdf` (new)
- Each player is a token. A shared ("Siamese") transformer encodes each team, and the two encodings are compared to give P(win). Data: NBA, LoL Pro and Amateur, Valorant, plus synthetic QKV-style synergy data.
- Result: beats an MLP, a Siamese MLP and a non-Siamese transformer. The non-Siamese transformer is near chance.
- Weakness: there is no bag-of-players logistic or BT baseline (checked by searching the text), so the size of the synergy effect isn't really shown.
- Relevance: if we model the opponent at all, encode both sides with shared weights and compare them. Their missing linear baseline is the same trap our results warn about.

## 4. Skill, noise and causal inference

**Ramos. "'Embarrassingly Causal': Causal Use of Associational Data in MTG Drafts." 2026.**
[arXiv 2604.18314](https://arxiv.org/abs/2604.18314) · `2026_embarrassingly_causal_mtg_drafts.pdf`
- A methods and philosophy paper. It argues that 17Lands statistics (GIH WR, GNS WR, IIH) are legitimately used causally even though they are confounded.
- It shows GIH WR is biased for the card's effect, through game length and deck context, and that IIH doesn't estimate what it looks like it does. No new model or data.
- Relevance: it backs our "cards cast/seen look good because winners see more cards" confound, and synergy round 8 (opening hand only, which is randomised by the shuffle).

**Chakraborty et al. "Adapting Skill Ratings to Luck-Based Hidden-Information Games." Dec 2025.**
[arXiv 2512.18858](https://arxiv.org/abs/2512.18858) · `2025_chakraborty_skill_ratings_luck_hidden_info.pdf` (new)
- An Elo variant for Rummy that uses score margins and an explicit initial-hand-quality term (β) to separate luck from skill. Tested only in simulation (270k games, 6 bot strategies).
- Relevance: the opening hand is random given the deck, so its quality is a legitimate covariate for variance reduction in game-level models (unlike cards drawn later, which depend on game length). Simulation-only evidence.

## 5. Other relevant work *(no PDF downloaded)*

- **puder** (Czerner, 2025), [blog](https://nicze.de/philipp/articles/puder/) and [code](https://github.com/suyjuris/puder): a 7M-parameter causal transformer with card-id tokens and skill/date metadata, trained on about 2M 17Lands drafts. It reports about **71%** top-1 on 1024 random test drafts, with a side head for "will this card make the deck".
  - Hiding the pack contents cost only 0.57%, which is much smaller than our +3 points from the pack.
  - Not peer reviewed. The set mix and test split aren't specified clearly enough to compare with our 68.6% **[unverified]**.
- **statistical-drafting** (Brooks, 2024), [statisticaldrafting.com](https://statisticaldrafting.com/about.html) and [code](https://github.com/danieljbrooks/statistical-drafting): a per-set MLP on pool and pack vectors, **trained only on high-volume, high-win-rate players**, about 70% agreement with those players. The authors say learning deck composition from draft data alone is "challenging".
- **Cardsformer** (Xia et al., ECAI 2023), [DOI](https://doi.org/10.3233/FAIA230581): language-grounded card representations let a Hearthstone policy generalise to unseen cards. Paywalled, not on arXiv.
- **AAIA'18 challenge winners**: Vu et al., "Predicting Win-rates of Hearthstone Decks: Models and Features that Won AAIA'2018", and Janusz et al., "Toward an Intelligent HS Deck Advisor: Lessons Learned" (FedCSIS 2018). Both are ResearchGate/IEEE only. According to search snippets, the winners combined deck-archetype clustering with logistic regression and NNs **[unverified, not read]**.
- **MTG-Causal-RL** (da Costa Cunha et al., [arXiv 2605.06066](https://arxiv.org/abs/2605.06066)): a Gym benchmark for Constructed MTG gameplay. Not relevant to drafting.
- **Skill vs. Chance Quantification for Card & Board Games** ([arXiv 2410.14363](https://arxiv.org/abs/2410.14363)) and **A Taxonomy of CCGs from a Game-AI Perspective** ([arXiv 2410.06299](https://arxiv.org/abs/2410.06299)): background only.
- Searched with nothing relevant found: Legends of Runeterra deck win-rate ML papers, and Hearthstone Arena *draft* ML papers from 2024–26. The newest Hearthstone and LoCM work is gameplay RL or evolutionary deckbuilding.
- Newer work citing DraftEncoder or DraftFM (Semantic Scholar, 2026-10-06): DraftFM cites DraftEncoder, and nothing cites DraftFM yet. Bertram 2024's other citers are unrelated, apart from an HCI paper on a Commander-format MTG data dashboard ([arXiv 2512.09802](https://arxiv.org/abs/2512.09802)).

## 6. MtGDraftEncoder repo: details not in the paper

Read from a shallow clone of [akulen/MtGDraftEncoder](https://github.com/akulen/MtGDraftEncoder) (HEAD `06373bd`, 2026-07-06):

- **Splits are fixed-length day windows counted back from each set's last draft.** In-set (`temporal_split`) runs use test = last 14 days, val = the 14 days before, train = the 14 days before that, so only one 2-week window is used for training.
  - Cross-set runs (`newset*.yml`) train on the last 14 days of each of 11 sets and test on the last 14 days of FDN. Fine-tuning uses the first N days of FDN with an 80/20 val split.
  - `ONE` is commented out ("only 3 weeks of data"). The code asserts that every draft finished at 7 wins or 3 losses (`n_incomplete == 0`), so **only completed runs are used**.
- **Epochs:** 100 for in-set runs (`sameset.yml`), 25–50 for cross-set runs (`newset2.yml`/`newset.yml`), and 25–50 more for fine-tuning.
  - Other settings: AdamW, warmup-cosine schedule, batch 256, d=64, 10 set layers + 5 draft layers, dropout 0.3, 3 seeds; the LR baseline uses lr 3e-5. They report the epoch with the best val loss.
- **Loss:** the exact negative-binomial NLL (Pascal-triangle coefficients), with an optional "KL" variant that subtracts the NLL of the empirical W/(W+L). Per-pick loss weight is (t+1)/45. A `user_n_games_bucket` weight is computed but never used in the loss.
- **Meta features** = six per-card win rates computed from the whole `game_data` file for the set: opening hand, drawn, tutored, in deck, sideboard and GIH. **There is no date filter, so the meta stats include the test-period games, which is a leak.** That makes the paper's "+Meta" numbers optimistic. Our replication fits GIH/GP WR on training data only.
- **The paper's HEURISTIC is mean "in-deck" WR (GP-WR-like, copy-weighted) over the top 23 picks** (column 3 = `deck` in `models/baselines.py`), not GIH WR. On MSH our top-23 GP WR heuristic is 0.166 and top-23 GIH WR is 0.213, so the README row calling GIH WR "the paper's heuristic" should probably be relabelled.
- **Card string format** (`card_utils.make_face_expr`):
  `[NAME] … | [MC] {2}{W} | [CMC] 3 | [COLOR] W | [RARITY] common | [TYPE] Creature | [SUBTYPE] Human Soldier | [P/T] 2/2 | [LOYALTY] … | [KEYWORDS] Flying, … | [EFFECTS] <oracle text, newlines → " <NL> ">`.
  - Double-faced cards become `[FACE_1] … // [FACE_2] …`. Reminder text is stripped only for keyword extraction.
  - The **card name is kept**, unlike DraftFM. Text is embedded with `google/embeddinggemma-300M` (768-d) using the prompt `"task: Magic the Gathering card selection in a draft | card: "`.
- The repo also contains unused variants: a DraftGraph GNN over a text-similarity kNN graph (density 0.01–0.05) and DraftAttention. `eval.py` turns the win-rate model into a pick policy by simulating each candidate pick. Set config lists bonus-sheet cards ("special guests", "The List") for each set.

## 7. Experiments the literature suggests that this repo hasn't run

Ranked by expected value per hour of compute. All of them were checked against WRITEUP.md and README.md.

1. **A skill-conditioned PickNet as the "strong drafter" in the table simulator.**
   - Why: DraftFM conditions on skill and experience and deploys at high skill. statistical-drafting trains only on top players. Our own data shows Mythic and top-WR picks are more predictable (0.656 → 0.696).
   - What to do: add rank and win-rate bucket as context to PickNet, or fine-tune on the top bucket. Simulate tables where 1–8 seats are the "expert" model and score their decks with the LightGBM/FM deck model.
   - This goes straight at the open item "stronger simulated drafters". It's cheap, since the PickNet and simulator already exist. Note that "rank/skill inputs to WheelNet" were a dead end, but that was wheel prediction, not the policy.
2. **Deck-model-guided picking (DraftEncoder `eval.py` idea).**
   - What to do: at each pick, roll the draft forward with PickNet bots for each top-k candidate and score the finished pool or deck with the deck model. Take the argmax, or mix it with the PickNet prior.
   - Compare its simulated win rate with PickNet and with "force the best pair".
   - It's the natural follow-on to the open-lane simulation. Moderate compute, and it can be run on small numbers of tables.
3. **Matchup model using `opp_colors`.**
   - Why: Jakubik and the AAIA'18 work say matchup-specific win rates are the right target, and Ye et al. compare the two sides with shared weights. `opp_colors` is loaded in `src/openlane/games.py` but no model uses it.
   - What to do: add (user colour pair × opponent colour pair) interactions, or a deck-embedding · opponent-colour bilinear term, to the game-level linear/FM model. Test paired log loss on 12 sets as in the synergy rounds.
   - Payoff: it shows whether archetypes have rock-paper-scissors structure, and it gives an opponent-strength adjustment, an open item in the README. Cheap, about a minute per set.
4. **A day-N protocol for the deck model: other sets + first 4 days of the target set.**
   - Why: DraftEncoder's strongest practical result is that a 4-day fine-tune almost closes the cross-set gap. Our cross-set deck transformer was tested only as a residual on full per-set training, and the wheel model's weekly fine-tune covers wheel prediction only.
   - What to do: on 3–4 sets, compare per-set linear/LightGBM trained on days 1–4 alone with the shared features-only model plus a 4-day fine-tune, testing on later drafts.
   - This answers "how good is the deck model in week 1", which is when it matters most for an overlay.
5. **A contrastive pool-card embedding as a second synergy prior.**
   - Why: Bertram's adapted InfoNCE reaches about 69% on NEO with a plain embedding geometry.
   - What to do: train it on picks (about an hour on MPS for one set). Use the deck's mean pool-card cosine, or pairwise card-card cosine, as an affinity column next to S_ij in the game-level FM.
   - Payoff: tests whether "picks are free synergy labels" depends on PickNet's particular bilinear form.
6. **Opening-hand quality as a variance-reduction covariate.**
   - Why: Chakraborty et al. use initial hand quality to separate luck from skill. Synergy round 8 already uses the opening hand to identify synergy.
   - What to do: add "sum of card power in the opening 7" (fit on train) as a covariate in the game-level deck and skill models. Measure the gain in test log loss, and whether card and pair estimates tighten across split halves.
   - Caution: don't add cards drawn later, which depend on game length (the post-treatment issue in Ramos 2026).
7. **Recalibrate the noise ceiling.** DraftEncoder's 0.23 ceiling assumes true p ~ N(0.53, 0.05). Our LightGBM deck model reaches 0.279, which implies a wider spread of true p, or a ceiling that differs because of record and matchmaking effects. Fitting the sd of true p by deconvolution from our own predictions and run records would give a defensible "fraction of explainable signal". This is analysis only, no training.
8. **Lower priority.**
   - LLM zero-shot card grades (UrzaGPT/GPT-4o-style) as a day-zero feature for the features-only PickNet, aimed at the P1P1 gap our README identifies.
   - A Set Transformer (ISAB + PMA) deck encoder in place of the FM. Our transformers have so far tied linear models, so expect little.
