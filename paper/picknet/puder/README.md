Re-scoring of the released puder v1 model (https://github.com/suyjuris/puder, `puder_v1_7M.pt`) on its own
1,024 test drafts. Run from a checkout of the puder repo with the model in `models/`:

    python eval_honest.py        # reported formula vs real picks vs non-forced; header ablations
    python per_set_metrics.py    # per-expansion Premier top-1, non-forced top-1, NLL -> puder_per_set.json

"Real picks" drops the padding slots that puder's own accuracy formula counts as correct (sets with 13- or
14-card packs pad every draft to 45 slots; argmax over all -inf logits returns 0, the pad token).
