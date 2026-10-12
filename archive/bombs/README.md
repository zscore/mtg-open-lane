# ARCHIVED — MARKED FOR DELETION

Everything in this folder is the bombs analysis (blog post `bombs.md` and images, the `bombs*`, `card_impact` and
`consensus_bombs` notebooks, the `bomb_luck`, `card_impact`, `consensus_bombs`, `depth_index`, `prince_index` and
`impact_validation` scripts, and `results/bombs/`). It now lives in its own repo, `../mtg-bombs`.

**Why it's marked for deletion:** it's a duplicate. On 2026-10-10 every file here was checked byte for byte against
`mtg-bombs` and all matched, including the edits that hadn't been committed yet. `mtg-bombs` is the copy to keep
working on. Paths under this folder match the old repo layout (`blog/`, `notebooks/`, `results/bombs/`, `scripts/`).

The scripts won't run from here as-is. They expect to sit in `scripts/` at the repo root.

**Before deleting:** check that `mtg-bombs` has everything pushed. Then remove the folder with `git rm -r archive/bombs`.
