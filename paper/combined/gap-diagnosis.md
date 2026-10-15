# TableNet: diagnosing the wheel gap

Status as of 2026-10-14. Companion to `architecture.tex` (steps 1 and 2 are written up there).

## Where we are

| | wheel log loss vs per-set WheelNet | pick top-1 vs PickNet |
|---|---|---|
| Step 1 (frozen PickNet + L=1 head) | +0.92% mean | unchanged (frozen) |
| Step 2 (joint fine-tune, `wheel_w=0.3`, 1 pass) | +0.53% mean, median +0.36% | +0.01 pt mean, all sets within 0.1 pt |

- The worst sets are the same in both steps: BLB, HOB, ECL and SOS (+1.2 to +1.5% in step 2).
- The per-set WheelNet is a weak bar. Most of those baselines have no draft-time context. On MSH:

  | model | MSH wheel log loss |
  |---|---|
  | `multiset_plus_target` (features-only WheelNet, 8 sets) | 0.4188 |
  | `wheelnet_time` | 0.4218 |
  | per-set WheelNet | 0.4249 |
  | TableNet step 2 | 0.4252 |

  So TableNet is 1.5% behind the best wheel model we have.
- In flight: `results/tablenet/run_w1_control.sh` runs `wheel_w=1` (`tablenet_joint_w1`), then `wheel_w=0`
  (`tablenet_joint_w0`, the pick-only control).

## The question

The tasks are related. A card wheels if none of the next 7 seats takes it, so P(wheel) ≈ ∏ₖ (1 − P(seat k takes c)).
That is the pick model run on seats whose pools we infer from what was passed. The head already uses some of
this: it gets PickNet's log-probs for the pack (`lpc`) and for the card we took (`lpt`) in `head_inputs`.

So why does a combined model still trail a dedicated wheel model? There are five candidate causes, and each
one points to a different fix:

| # | hypothesis | what it predicts |
|---|---|---|
| H1 | **Under-training.** One pass, PickNet at lr 5e-5, wheel weight 0.3. | `wheel_w=1` and/or a second pass close more of the gap; picks still don't move. |
| H2 | **Task interference.** Pick and wheel gradients disagree in the shared PickNet layers. | Gradient cosine is negative in some modules; raising `wheel_w` starts costing pick top-1. |
| H3 | **Head too small.** PickNet's representation has the information, but a 1-layer head can't extract it. | Frozen L=2 or L=3 beats frozen L=1 by a lot (L=0 → L=1 gained 1–2%). |
| H4 | **Representation mismatch.** PickNet's tokens are built for "which card do I take", not "what do the other seats take". | Only unfreezing helps, and it plateaus; the gap is concentrated in late-pack, signal-heavy picks. |
| H5 | **Set specialization.** Per-set models fit each set's quirks. | Weakened already: on MSH, the multi-set WheelNet *beats* the per-set one (0.4188 vs 0.4249), so pooling sets helps the wheel task. Check only if everything else is ruled out. |

## Diagnostics, cheapest first

### D0. Read the w1 and w0 runs (already running)

The JSON outputs are `results/tablenet/tablenet_joint_w1.json` and `tablenet_joint_w0.json`.

- **w1 vs step 2 (w=0.3).** If the gap shrinks again and pick top-1 stays within 0.1 pt, that supports H1
  and argues against H2. If picks drop, that is the first real evidence for H2.
- **w0 vs step 2, picks only.** If step 2 beats w0 on pick top-1, the wheel loss is *helping* picks (positive
  transfer, worth writing up). If they tie, the FRA (+0.29) and HOB (+0.21) gains came from finally training on
  those drafts, not from the wheel loss.
- Ignore w0's wheel numbers. Its head trains on nothing.

### D1. Gradient cosine (new script, about an hour to write; minutes to run)

A new `scripts/grad_cosine.py` would reuse `forward()` and `deck_loss()` from `train_tablenet_joint.py`.
Load `models/tablenet_joint.pt` (and step 1's starting point for comparison), and for about 300 training
batches:

```python
g_pick  = torch.autograd.grad(nll, pn_params, retain_graph=True, allow_unused=True)
g_wheel = torch.autograd.grad(wb,  pn_params, retain_graph=True, allow_unused=True)
```

Report the cosine and the norm ratio ‖g_wheel‖/‖g_pick‖, overall and per module group:

- card table / feats MLP
- `hist` (history encoder)
- `blocks` (pack decoder), each block separately
- `head` (pick head; should receive no wheel gradient)

Then plot the mean cosine and its spread per group.

How to read it:
- **Mean cosine clearly < 0 in a group** (say below −0.05, consistently across batches): conflict (H2) located
  in that group. That group is where to split the model per task (see B2).
- **About 0:** the tasks are independent there. Sharing costs nothing and gains nothing.
- **> 0:** the tasks help each other there. Keep it shared.
- **Norm ratio:** tells you whether `wheel_w` is actually balanced or one task dominates the PickNet updates.

### D2. Head capacity (small change to step 1)

`train_tablenet.py` hard-codes `heads = {L: ... for L in (0, 1)}`. Make that list a config key and train
frozen L=2 (and L=3 if L=2 helps).

- **Large gain from L=2** (on the order of the 1–2% that L=1 gave): H3. Fix it by enlarging the head before
  more joint training.
- **Small gain:** the information isn't in the frozen representation, so H4 is more likely.

### D3. Where is the gap? (analysis only)

Join TableNet and WheelNet test predictions per passed card (`test_tablenet_L1_*.npz` vs each set's WheelNet
predictions; the example counts already match) and split the log-loss gap by:
- pack and pick within pack (early vs late),
- within-pack rank of the card (`wheel_breakdown.py` already does this for WheelNet),
- whether the card's colour is "open" upstream (`open_lane.py`).

If the gap sits in late or open-lane cases, PickNet isn't reading signals well enough for the wheel task (H4).
That matches the outside finding that pick models barely use pack contents.

## Fixes, by diagnosis

### A. Under-training (H1)
- Second pass, or `epochs=2`; try `lr_trunk=1e-4`.
- Learn the balance instead of sweeping `wheel_w`: uncertainty weighting (Kendall et al. 2018) or GradNorm.

### B. Task interference (H2)
1. **PCGrad.** When the pick and wheel gradients conflict, project the conflicting component out. A drop-in
   change to the training step.
2. **Split the top of the model.** Share the card table, history encoder and lower pack-decoder blocks.
   Give each task its own copy of the top 1–2 `blocks`, splitting where D1 shows the conflict. Usually the
   strongest simple baseline.
3. **MMoE / PLE.** Several expert networks in the decoder, with a learned gate per task. Built for the
   recommender-system case where improving one task degrades the other. More code; try it only if B2 isn't
   enough.
4. **Per-task adapters / LoRA.** On the shared blocks: cheap, and the pick path stays exactly PickNet.

### C. Head capacity or representation (H3, H4)
- Bigger head (L=2 or 3, or wider).
- **Give the head signal features PickNet doesn't build itself:** the multiset of cards taken upstream so far
  (pack t minus pack t+8 from earlier wheels), per-colour "openness". WheelNet's token design (distinct passed
  cards, block-diagonal attention) is a template.
- **Distillation.** Train the wheel head on a mix of the real label and the multi-set WheelNet's soft
  predictions. Usually the most reliable way to close a gap to a stronger specialist. Needs teacher
  predictions for all 19 sets (the multi-set WheelNet has only been run with target MSH so far).

### D. Set specialization (H5)
- Per-set adapters or FiLM in the wheel head, conditioned on `sid` (already an input).

## After the gap: making the combination pay off

Matching WheelNet only shows that one network can do both jobs. To show the tasks *help each other*, use them
together:

1. **Wheel → pick.** Feed each candidate's wheel probability (as if it were passed) into the pick head. A
   strong card likely to come back is less urgent. Test: pick top-1 overall and on picks where the human took
   card A while a comparable B was likely to wheel. The counterfactual "If I take A, will B wheel?" section of
   `architecture.tex` is the machinery for this.
2. **Structured wheel head.** Model the 7 downstream seats explicitly: infer a pool/colour state per seat
   from what was passed, run PickNet's scorer on it, and combine as ∏ₖ (1 − P(seat k takes c)). This is the
   most principled design and the most work. Do it only if D3 shows the gap is about reading the table.

## Suggested order

1. D0 (free, already running)
2. D1 (gradient cosine)
3. D2 (head capacity)
4. D3 (gap breakdown)
5. Branch into A/B/C by the results
6. Distillation from the multi-set WheelNet, whatever the diagnosis
7. Wheel → pick feedback

## Results

### D0, part 1: wheel weight 1.0 (2026-10-15; `tablenet_joint_w1.json`)

Compared with step 2 (w=0.3): wheel log loss is 0.30–0.41% better on all 19 sets (mean −0.34%). Pick top-1
changes by −0.02 to +0.01 points (mean −0.01). The gap to the per-set WheelNet drops to:
- **+0.19% mean, +0.04% median**
- 15 of 16 sets within 1%; 9 at or past parity
- still behind on BLB (+1.1%), HOB, ECL and SOS (about +0.9%)

On MSH: 0.4239, compared with 0.4218 for `wheelnet_time` and 0.4188 for the multi-set WheelNet.

Supports H1 (under-training). No pick cost, so no visible H2 yet.

### D1: gradient cosines (`grad_cosine.json`, `grad_cosine.log`; 1000 training batches of 32 per checkpoint)

**Per-batch cosines are ≈ 0 everywhere.** Overall pick–wheel is +0.003 to +0.004 (sd 0.02). Single minibatches
are dominated by noise, so there is no batch-level fight.

**The batch-averaged cosine is negative, and the disagreement is in the history encoder:**

| group | step 1 | w=0.3 | w=1 |
|---|---|---|---|
| card table | −0.03 | +0.01 | +0.01 |
| input embeddings | −0.15 | −0.08 | −0.15 |
| history layers 0–3 | −0.14 / −0.19 / −0.18 / −0.18 | −0.09 / −0.19 / −0.21 / −0.26 | −0.21 / −0.28 / −0.30 / −0.38 |
| decoder blocks 0–2 | −0.01 / +0.02 / +0.01 | −0.06 / −0.05 / −0.01 | −0.08 / −0.07 / −0.06 |
| **all of PickNet** | −0.07 | −0.07 | −0.09 |

- **The deck loss shows the same pattern against the wheel.** Deck–wheel cosine in the history layers at w=1 is
  −0.12 to −0.24. Pick and deck agree with each other everywhere (+0.05 to +0.19).
- **The wheel gradient is small.** Its per-batch norm is about 1/5 of the pick gradient's (0.06 vs 0.28
  overall), so at w=0.3 the wheel loss was about 6% of PickNet's update.

#### Reading
- **There is a real but mild conflict, and it is localized.** In the history encoder, the wheel task wants
  changes that the pick and deck tasks resist. Plausibly this is because that encoder summarizes *my* picks and
  plan, while the wheel task needs what the *other* seats are doing. The card table and the pack decoder are
  close to neutral.
- **Step 1 is the cleanest evidence.** There, only PickNet's own training had shaped the network, and the
  history encoder is already at −0.14 to −0.19.
- **The cosines in the jointly trained checkpoints overstate the conflict.** At a stationary point of
  pick + 0.3·deck + w·wheel, the averaged gradients must cancel, so they turn anti-aligned by construction. That
  is why they deepen as w grows. `grad_cosine_late` also reports |mean g| to show how stationary each loss is.
- **The conflict costs nothing yet.** Pick top-1 moves by −0.01 points at w=1. Raising the weight is still
  cheap. The w=3 run shows whether that stays true.

#### What it suggests
- If w=3 starts costing picks, fix **just the history encoder**: per-task LoRA/adapters on the history layers,
  or a separate small history/signal encoder for the wheel head. Don't use PCGrad over the whole model, and
  don't split the decoder blocks; that is where the tasks already agree.
- This also fits H4 and the "signal features" fix (C). The wheel head needs a summary of the table that the
  history encoder isn't built to provide.

### D0, part 2: pick-only control, w=0 (2026-10-15; `tablenet_joint_w0.json`)

Same fine-tune, no wheel loss. Pick top-1, w0 vs the runs with the wheel loss:
- **w=0.3:** −0.001 points (identical).
- **w=1:** −0.010 points mean (lower on 17 of 19 sets). Pick NLL is higher on 18 of 19 sets, but only by +0.0002
  nats on average.

What this means:
- **The wheel loss does not help picks (no positive transfer).** The FRA (+0.29) and HOB (+0.21) gains seen in
  step 2 also appear in w0, so they come from training the whole network on those drafts for the first time.
- **At w=1 there is a measurable but negligible pick cost.** It is consistent in sign across sets and
  far below the 0.1-point bar. It matches D1: a mild conflict in the history encoder.
- **Sharing is nearly free.** One network matches the per-set WheelNet (+0.19%) at a pick cost of 0.01
  points. The open question is where this costs more as w rises (the w=3 run).

### D0, part 3: wheel weight 3.0 (2026-10-15; `tablenet_joint_w3.json`)

**Wheels:** 0.33–0.53% better than w=1 on all 19 sets. Against the per-set WheelNet:
- **−0.22% mean, −0.34% median**
- 11 of 16 sets at or past parity; all 16 within 1% (worst BLB +0.6%)
- MSH 0.4224: past the per-set WheelNet, still behind `wheelnet_time` (0.4218) and the multi-set WheelNet (0.4188)

**Picks** (against the w=0 control): top-1 −0.034 points mean, lower on 18 of 19 sets. The worst are HOB
(−0.09) and MOM (−0.08), close to the 0.1-point bar. Pick NLL is +0.0012, five times the w=1 cost.

**The pick/wheel trade-off so far:**

| wheel weight | wheel vs WheelNet (mean) | pick top-1 vs w=0 (mean / worst set) | pick NLL vs w=0 |
|---|---|---|---|
| 0.3 | +0.53% | −0.001 / −0.02 | ≈0 |
| 1 | +0.19% | −0.010 / −0.03 | +0.0002 |
| 3 | **−0.22%** | −0.034 / −0.09 | +0.0012 |

At w=3, TableNet beats the per-set WheelNets and still passes the pick bar, but with no margin left. The pick
cost now grows faster than the wheel gain. The next run, `tablenet_joint_w3_hblock` (w=3 with
`wheel_block=hist`), tests whether that cost comes from the history encoder, as D1 suggests.

### D0, part 4: a second pass at w=1 (2026-10-15; `tablenet_joint_w1_ep2.json`, resumed from `tablenet_joint_w1`)

**Wheels:** 0.29% better than one pass on average. Against the per-set WheelNet:
- **−0.10% mean, −0.23% median**
- 10 of 16 sets at or past parity; all 16 within 1%
- MSH 0.4229
- still behind on BLB (+0.7%), ECL, HOB (+0.6%) and SOS (+0.4%)

**Picks** (against the one-pass w=0 control): top-1 −0.013 mean, worst MOM −0.08; NLL +0.0004.
- **Caveat:** this comparison is not like-for-like. There is no two-pass w=0 control, so some of the pick
  change could come from the extra pass itself.

| run | wheel vs WheelNet | pick top-1 vs w=0 (mean / worst) | pick NLL vs w=0 |
|---|---|---|---|
| w=1, one pass | +0.19% | −0.010 / −0.03 | +0.0002 |
| w=1, two passes | −0.10% | −0.013 / −0.08 (MOM) | +0.0004 |
| w=3, one pass | −0.22% | −0.034 / −0.09 (HOB) | +0.0012 |

- **H1 is confirmed.** One pass was under-training the wheel task.
- **Training longer is cheaper than raising the weight.** Per unit of wheel gain, the second pass costs picks
  much less on average (NLL +0.0004 vs +0.0012). The worst set is about the same in both runs.

### D1, part 2: gradient cosines at w=3 and after two passes (`grad_cosine_late.json/.log`)

**The same picture, sharper.** The batch-averaged pick–wheel cosine at w=3:
- **history layers 0–3:** −0.36, −0.37, −0.39, −0.52
- **input embeddings:** −0.22
- **decoder blocks:** −0.10 to −0.06
- **card table:** about 0

The deck loss again sides with the pick loss (pick–deck +0.21 to +0.31 in the history layers; deck–wheel −0.22
to −0.33).

**The stationarity check confirms the caveat above.** The averaged gradient is only about 4% of the per-batch
norm for both losses: pick 0.010 vs 0.28, wheel 0.0024 vs 0.055. So these checkpoints are near a stationary
point of the weighted loss, and much of the anti-alignment is the compromise itself.

**The useful information is where the compromise is struck.** It is almost entirely in the history encoder and
its inputs, which is exactly what `wheel_block=hist` removes from the wheel task.
