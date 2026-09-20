# METHOD — the rules this work holds itself to

The runs, their committed rules and their results live in
this repository. What a result is allowed to claim, and what has to
happen before a run, lives here, with the predictions in `PREDICTIONS.md`.

## Rules before runs

Each question is decided by rules written into its script's docstring and
committed **before** the run that answers it, so a failed rule stays failed.
Where a result is negative or undecidable, it is reported as such.

A rule is not re-tuned once its run has been seen. A sharper question gets a new
rule, committed before its own run.

A **reading** of a result is a different object and is corrected in place when
the evidence says so, as step 7's was.

## Setting a threshold

Every rule needs numbers — a response threshold, a correlation floor, a grouping
of channels — and choosing them by feel has cost several runs, so a **lookup pass
comes before the rule is written**, and the numbers it returns go in the
docstring beside the threshold they justify.

A lookup may measure:

- **the null** — the shuffled and random-control distributions, which fix the
  noise floor and so what "above chance" can mean;
- **scale and feasibility** — whether a stimulus of the intended size moves the
  intended readout at all, and what range that readout spans;
- **items whose answer is already known** from the literature or from an earlier
  scored task, which calibrate sign and magnitude;
- **whether each category holds what it is assumed to hold** — that a readout set
  contains the neurons the literature says carry the function, and that a channel
  grouped under one label does not span opposite ones.

A lookup may **not** measure the comparison the verdict is about. Setting the
threshold from the effect under test is how a rule stops being a test. When only
the effect itself would answer the question, the answer is a calibration slice
held out from scoring, not a peek.

Where a threshold can be avoided entirely, avoid it: a rank test against N
shuffled seeds ("above all 20", one-sided p ~ 1/21) contains no number chosen by
hand, and steps 7 and 7b used it.

## Every arm must be able to learn

When a verdict compares a trained model against controls — shuffled wiring, the
input alone, an oracle — a control that cannot learn the task on its own training
distribution makes the comparison worthless in both directions: a win over it is
cheap, and a loss to it is noise. So:

- **A gate in the rules.** Every arm must clear its own in-distribution test, a
  lower bound above chance, before any comparison involving it is read. An arm
  that fails makes that comparison VOID, never a pass.
- **A learnability check before the rules.** The gate catches a broken arm after
  the run; it does not prevent one. Before the rules are committed, every arm is
  trained on a calibration slice — odors, neurons or samples that no scored set
  uses — and checked for in-distribution learning only. An arm whose training loss
  goes to zero while its held-in accuracy stays at chance is memorising, and the
  training setup is changed before the run: more trials, a smaller memory, or a
  comparator with nothing to fit.
- **A comparison that needs no training.** Where the question allows, measure it
  once with no learned component, such as a fixed similarity between features. A
  null arm's failure can then be told apart: no information in its features, or a
  training procedure that could not find it. After step 10 this took one cosine
  similarity and separated the two cleanly: same-odor and different-odor trials on
  the real wiring's LH output at AUC 0.947 on test B, on a shuffled seed's at 0.514.

The calibration slice is not a peek. It shares no odor, neuron or sample with any
scored set, nothing from it enters a verdict, and its numbers go into the
docstring beside the training settings they justify. Where pools are too small to
spare a slice, the check runs on a synthetic input of the same shape, and the
docstring says so.

## What this was learned from

- **Step 4, E1.** The 0.01 response threshold was inherited from `score_shiu.py`
  without checking it against that readout. Shuffled DNp01 responses span
  0.001–0.004, so no variant of the network could have reached 0.01 at the giant
  fiber: the rule was unreachable before it was run, and looking at the null
  alone would have said so.
- **Step 6, P5.** Bitter was grouped with sugar and water as "food" and required
  to score above the pheromone channels on MN9. Bitter is aversive and suppresses
  proboscis extension — already measured in the repo, in Shiu task 5 — so it goes
  negative and fails the rule for the correct biological reason. A lookup on that
  known item would have split the grouping.
- **Step 6, P1 and P3.** ppk23, ppk25 and IR52b were treated as one category.
  They route to different places, so a conjunctive rule over them could only fail.
- **Step 7, the reading of it.** The readout sets were audited for size, missing
  labels and overlap, but never for whether they contained the neurons that
  matter. The ppk channels' published target, PPN1 = `AN05B102a`, carries no
  `dimorphism` or `fruDsx` annotation and so sat outside every readout, which made
  "routed away from courtship circuitry" the wrong reading of a correct
  measurement. Checking one named neuron against the sets would have caught it.

- **Steps 9 and 10, the null arms.** Step 9's taste pilot passed V3 and V4
  against shuffled-wiring and input-only arms that scored at chance on test A as
  well, having memorised 800 training trials (loss 7e-5). Step 10 added a gate and
  a fixed projection with weight decay, both set without checking that they
  prevent memorisation. All five shuffled seeds and the input-only arm memorised
  again (loss 2e-5 to 1.5e-4, test A 0.49-0.52), and both comparisons came out
  VOID. The gate kept the verdicts honest; nothing had checked, before either run,
  that the controls could learn at all — and PR-6 called both arms valid on the
  strength of that unmeasured setting.

Three of the first four are one mistake: never asking whether a category holds what
its name suggests. All of them stand as recorded.

## Exploratory versus confirmatory

Reading a run's data after its verdicts are in is legitimate and produced step
7's hypothesis. The line:

- **generating** a hypothesis from data is fine, and the numbers that generated it
  are exploratory — they are never later cited as evidence for it;
- **testing** it needs rules committed first, and data the generation did not use;
  where that is not fully possible, say so plainly rather than implying it was.

## Predictions

`PREDICTIONS.md` records what each run's verdicts are expected to be, with a
name and a reason, **before** the run exists, and the outcome is filled in
afterwards without editing the prediction.

A prediction carries **no confidence number**, by the owner's decision on
2026-09-16. Entries PR-1 and PR-2 carry one and keep it, because a prediction is
not edited after the fact. The number was doing no work: six correlated calls on
one run cannot calibrate it, one wrong call at 0.7 cannot either, and no run
planned here will produce enough independent calls to make 0.55 and 0.75 mean
different things. What the record is for is whether the claim was right, so the
claim is what gets written.

Rules alone do not stop someone designing a test whose answer they already know.
A prediction carries no weight in any verdict; it is a scoreboard for the
confident statements made around this work, and it only starts being worth
something once part of it is wrong.

## Where a question came from — `owner-doubt`

The record says what was run and what came out. It has not said **why anyone
doubted the thing in the first place**, and that turns out to be the part worth
reusing.

From 2026-09-21, an `owner-doubt:` line heads any `LK-` or `PR-` entry that
exists because a premise was challenged from outside the work. One line: what
was doubted, and what it changed. It is not credit and not a changelog. It is
there because the whole point of this file is to stop a future reader — most
likely the owner — being fooled by work that looks well designed.

**A record of clean experiments hides the mechanism that cleaned them.** Read
back, `LK-13` looks like ordinary diligence: someone measured olfaction's own
dynamics before writing rules. What actually happened is that the claim "a time
axis was tried and failed" had been carried over from step 8, which never
touched olfaction, and it took one question to expose it. Without the line, the
next reader learns that step 12 was careful. With it, they learn which kind of
carelessness the care was correcting.

What qualifies: a challenge to a premise, a scope, a priority or a claim of
novelty, where the answer changed the work. What does not: approving a plan,
choosing between offered options on taste, or a defect the run's own gates
caught. A smoke test catching a saturated statistic is the process working as
designed and belongs in the entry's body, not here.

The line names the doubt, never the person's reasoning-in-hindsight. If the
doubt was wrong and the work proceeded unchanged, it is still written down —
a record of challenges that only lists the successful ones is the same failure
this file exists to prevent.
