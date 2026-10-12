# Community bomb consensus: method and sources

Compiled 2026-10-07. This is a hand-curated list of the cards the Limited community called bombs in each of the
19 sets in `notebooks/bombs.ipynb`. It exists so we can compare community opinion with our 17Lands ratings.

## Files

| file | what |
|---|---|
| `consensus_bombs.csv` | The curated list (hand-written): `set, card, when, agreement, note`. |
| `consensus_bombs_vs_data.csv` | The list joined to our data by `scripts/consensus_bombs.py`. Adds the matched card name, the card's rank in its set by shrunk normalized GIH WR (Premier Draft), the rating in pp, the robust z, and `data_bomb` (z > 2.70, the notebook's Tukey-fence cutoff). |
| `data_top10_not_in_consensus.csv` | Cards in our per-set top 10 that the community list doesn't mention. |

## How the list was built

Four web-research subagents each covered about five sets: NEO–LCI, MKM–FDN, DFT–TLA, and ECL–MSH. Each agent was
asked for 5–10 consensus bombs per set, with sources, and was told to keep the pre-release view separate from the
post-release one. I transcribed their reports into the CSV without re-checking every claim.

The columns mean:

- **`when`**
  - `pre`: the evidence is a set review or early tier list.
  - `post`: the evidence is a retrospective, a draft guide written after release, or an article citing 17lands data.
  - `both`: the card was rated a bomb before release and confirmed after.
- **`agreement`**: the agent's judgement of how widely the card was called a bomb.
  - `strong`: several sources agree, or the card topped the data.
  - `moderate`: rated bomb-level in one or two sources.
  - `weak`: a single source, or only a search snippet.
  - `contested`: hyped before release but later downgraded, or the metrics disagree.

**The `post` evidence is partly 17lands data.** Many post-release sources are built from 17lands win rates, including
limitedgrades.com letter grades, the CoolStuffInc "by the numbers" articles, mtgrocks, and the mtgds AWR (adjusted win
rate). Agreement between those and our ratings is therefore not independent confirmation. The independent opinion
signal is the `pre` column.

## Caveats

- No agent could read Reddit (r/lrcast), Limited Resources, or Lords of Limited. LSV's TCGplayer reviews needed
  JavaScript to render, so only a few of his grades came through, via search snippets.
- 17lands' public `card_ratings` endpoint returned empty stats, so none of the win rates quoted in `note` were
  checked against 17lands directly. They are as the source articles report them.
- "Hyped but not borne out" (`contested`) is often inferred from a card's absence from post-release top lists, not
  from an explicit statement.
- Bonus-sheet cards are in the list where sources rated them. These are Multiverse Legends (MOM), Breaking News /
  OTP (OTJ), Special Guests (BLB, DFT, MSH), Through the Ages (FIN), Source Material (TMT), and Mystical Archive (SOS).
  Sheoldred, the Apocalypse (ONE) and Crashing Footfalls (MKM) are not main-set cards, but they appear in our 17Lands data.
- Two cards didn't match a name in our data:
  - **Tan Jolom, the Worldwalker (LCI)**: probably under `MIN_GIH`, or a naming mismatch.
  - **Terra Branford (FIN)**: Card Kingdom's name for it. The in-set card may be "Terra, Magical Adept".
- Some search results came from AI-generated spam sites. The agents said they ignored those sites.

## What the comparison shows (`consensus_bombs_vs_data.csv`)

| when | agreement | cards | median data rank | share above our bomb fence |
|---|---|---|---|---|
| both | strong | 45 | 3 | 0.73 |
| post | strong | 17 | 3 | 0.76 |
| both | moderate | 47 | 8 | 0.32 |
| post | moderate | 35 | 6 | 0.29 |
| pre | strong | 10 | 10.5 | 0.30 |
| pre | moderate | 34 | 12 | 0.12 |
| pre | weak | 19 | 38 | 0.06 |
| pre | contested | 27 | 39 | 0.00 |

## Sources by set

**NEO**
- https://draftsim.com/mtg-neo-limited-set-review/
- https://mtgds.wordpress.com/2022/03/16/era-of-enlightenment-adjusted-win-rate-estimates-for-kamigawa-neon-dynasty/
- https://dotesports.com/mtg/news/7-best-kamigawa-neon-dynasty-limited-bombs

**ONE**
- https://mtgazone.com/phyrexia-all-will-be-one-limited-tier-list/
- https://draftsim.com/mtg-one-limited-set-review/
- https://mtgrocks.com/best-and-worst-cards-to-draft-in-one-according-to-data/
- AetherHub (search snippet only)

**MOM**
- https://draftsim.com/mtg-mom-limited-set-review/
- https://mtgrocks.com/mtg-march-of-the-machine-limited-has-a-blue-problem/
- https://www.coolstuffinc.com/a/sigmundausfresser-seo-10212024-looking-back-at-the-best-cards-of-march-of-the-machine

**WOE**
- https://mtgazone.com/wilds-of-eldraine-draft-guide/
- https://draftsim.com/mtg-woe-draft-guide/
- https://mtgrocks.com/wilds-of-eldraine-mythic-uncommon-is-crushing-competitive-play/

**LCI**
- https://mtgrocks.com/mtg-best-lost-caverns-of-ixalan-bombs/
- https://mtgazone.com/the-lost-caverns-of-ixalan-lci-limited-tier-list/
- https://draftsim.com/mtg-lci-draft-guide/

**MKM**
- https://draftsim.com/mtg-mkm-limited-set-review/
- https://mtgazone.com/murders-at-karlov-manor-limited-tier-list/
- https://mtgazone.com/murders-at-karlov-manor-mkm-limited-over-and-underperformers-guide/
- https://www.limitedgrades.com/mkm

**OTJ**
- https://mtgazone.com/outlaws-of-thunder-junction-otj-limited-tier-list/
- https://www.tcgplayer.com/content/article/Outlaws-of-Thunder-Junction-Limited-MTG-Set-Review-Multicolor-Artifacts-and-Lands/88acfaac-76e5-4b53-8064-402d9187c7c9/
- https://www.limitedgrades.com/otj

**BLB**
- https://draftsim.com/mtg-blb-limited-set-review/
- https://mtgazone.com/blb-limited-tier-list/
- https://www.limitedgrades.com/blb

**DSK**
- https://draftsim.com/mtg-dsk-limited-set-review/
- https://www.tcgplayer.com/content/article/Duskmourn-Limited-MTG-Set-Review-White/40f001d7-9d4a-476d-a81d-ff76ccfd0c4e/
- https://www.limitedgrades.com/dsk

**FDN**
- https://mtgazone.com/fdn-draft-guide/
- https://www.cbr.com/mtg-foundations-draft-best-bomb-rares/
- https://www.limitedgrades.com/fdn
- https://magic.wizards.com/en/news/feature/collecting-foundations

**DFT**
- https://draftsim.com/mtg-dft-limited-set-review/
- https://www.coolstuffinc.com/a/michaellongsmith-seo-03172025-the-best-rares-in-aetherdrift-draft-by-the-numbers

**TDM**
- https://www.coolstuffinc.com/a/michaellongsmith-seo-05082025-the-best-rares-and-mythics-of-tarkir-dragonstorm-draft/
- https://draftsim.com/mtg-tdm-draft-guide/

**FIN**
- https://draftsim.com/mtg-fin-limited-set-review/
- https://cardgamebase.com/final-fantasy-draft-tier-list/
- https://blog.cardkingdom.com/the-problem-with-the-bonus-sheet-in-final-fantasy-limited/
- untapped.gg (search snippet only)

**EOE**
- https://www.coolstuffinc.com/a/michaellongsmith-seo-08142025-the-best-rares-in-edge-of-eternities-limited
- https://draftsim.com/mtg-eoe-draft-guide/

**TLA**
- https://draftsim.com/mtg-avatar-bomb-set/
- https://draftsim.com/mtg-tla-draft-guide/
- https://cardgamebase.com/avatar-the-last-airbender-draft-tier-list/

**ECL**
- https://draftsim.com/mtg-ecl-limited-set-review/
- https://draftsim.com/mtg-ecl-draft-guide/
- https://www.coolstuffinc.com/a/michaellongsmith-seo-02042026-the-best-rares-mythics-in-magics-lorwyn-eclipsed-limited
- https://www.limitedgrades.com/ecl

**TMT**
- https://draftsim.com/mtg-tmt-limited-set-review/
- https://draftsim.com/mtg-tmnt-draft-guide/
- https://www.coolstuffinc.com/a/michaellongsmith-seo-03112026-the-five-best-rares-and-mythics-in-tmnt-limited
- https://www.limitedgrades.com/tmt

**SOS**
- https://draftsim.com/mtg-sos-limited-set-review/
- https://draftsim.com/mtg-sos-draft-guide/
- https://www.coolstuffinc.com/a/best-rares-and-mythics-in-secrets-of-strixhaven-limited-05052026
- https://www.limitedgrades.com/sos

**MSH**
- https://mtgrocks.com/marvel-super-heroes-limited-bombs/
- https://draftsim.com/mtg-msh-limited-set-review/
- https://draftsim.com/mtg-msh-draft-guide/
- https://www.limitedgrades.com/msh
