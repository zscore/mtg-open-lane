Short version: I couldn't find anyone who has built a proper "will this card come back?" model on human-draft data. The pick-prediction side is crowded, though, and 17Lands' public data should give you the wheel labels almost directly.

What already exists

Pick predictors (a lot of these). Statistical Drafting (a small neural net), Philipp's "puder" model (about 2M drafts, 71% top-1 pick accuracy), Timo Bertram's academic papers on predicting human card choices, and the new DraftFM paper (Aug 2026; 149M picks across 29 sets). All of them answer "what should I (or a human) take from this pack?" None of them model the other drafters, how packs move around the table, or wheeling. DraftFM says outright that it doesn't. The puder author also found that showing his model the pack contents barely helped (+0.57%). That suggests these models aren't really reading signals.
Wheel estimates (crude ones). The MTGA_Draft_17Lands overlay has a "WHEEL %" column. It's a heuristic built from average-last-seen-at (ALSA) numbers and a Sierkovitz article, it only covers pack 1 picks 1–6, and even the fork's own README isn't sure the code works right. It doesn't condition on the actual signals in your draft.
17Lands' own draft simulations use bots trained on user data to show how table dynamics affect what comes around. That's a simulation, not a wheel predictor you can use.

So the space looks open, though a search only goes so far. Something could be sitting in a Discord or an unindexed repo.

Public data

17Lands public datasets (17lands.com/public_datasets, hosted on S3 at 17lands-public). Per set and event type there are draft data, game data, and replay data files as .csv.gz. The draft file is wide: one row per pick, with pack_card_<name> columns (what was in the pack), pool_<name> columns (what you'd taken so far), plus pick/pack number, rank and win rates. For human-only drafts, use Premier and Traditional and skip Quick Draft, since that's bots.
Tooling: spells (Python/Polars; downloads a set and converts it to parquet) and mtgr (R).
Mirrors: some Hugging Face copies (e.g. unclemusclez/17LandsDSK-Draft) and an IEEE DataPort dataset. I'd go to the original source rather than these.
License: I couldn't load the usage-terms page (it's rendered in JavaScript), so check 17lands.com/usage_guidelines yourself. I think they ask for attribution, but I'm not sure.

Why I think it's feasible, and the catches

Free labels. A pack you see at pick p comes back to you 8 picks later (same pack round, 8 drafters), and the data records its contents both times. Diff the two and you know exactly which cards wheeled, with no extra labeling. Please check this against the actual pack sizes for the set you pick; I haven't looked at a file yet.
You only see one seat. The other 7 drafters aren't in the data, and 17Lands users skew above average. That's fine, because you only need to predict what happens around a 17Lands user. Just don't claim the model describes the general Arena population.
Baseline to beat: ALSA/ATA for each card by itself. If your model can't clearly beat that, the signal features aren't earning their keep. Given the puder result, that's a real risk, and finding out would be interesting either way.

Happy to pull a set's draft file and check the wheel-labeling idea on real rows if you want to kick the tires before committing.

Sources:

Statistical Drafting – About
An AI for MTG drafting (puder)
DraftFM (arXiv 2608.19568)
Predicting Drafted Deck Strength (arXiv 2607.04782)
Bertram et al., Predicting Human Card Selection
MTGA_draft_17lands (WHEEL feature)
17Lands Blog – Simulating Draft Strategies
spells
mtgr download function
Getting started with 17lands data in R
IEEE DataPort MTG drafting data
17Lands usage guidelines