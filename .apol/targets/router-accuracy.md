# router-accuracy

Read-only scorecard: nothing here is optimized. The target file is a placeholder.

**What it scores.** The share of the 24 held-out prompts in `evals/router_prompts.jsonl` that the
big-vs-small switch routes to the branch the label names. The 48 train prompts are where the two
thresholds were fitted; the 24 scored here were never fitted on.

**Why this metric.** A router is a cost optimisation, so the thing worth guarding is not that it is
clever but that it has not quietly started sending hard prompts to the small model. The run JSON
carries `hard_to_small` and `cost` alongside the score for exactly that reason, and the shipped
thresholds minimise cost rather than accuracy: a hard prompt sent to the small model counts three
times an easy one sent to the big model, matching `act_costs.escalate` 0.5 against
`cost_wrong_act` 3.0 in the checkpoint's own config.

**Read the baselines before trying to raise the score.** Fitting the thresholds on accuracy instead
of cost scores 75.0 against the shipped 70.83, and is a worse router: 5 expensive mistakes instead
of 3. That is recorded as a baseline so the regression cannot be shipped as an improvement.

**Caveats.** n=24, so the 95% interval is roughly 54 to 88 and the gate is deliberately loose at 60.
All 72 prompts were written and labelled by one engineer, so they are a judgment rather than ground
truth, and they are not real traffic. The honest next step is to replace the file with real prompts
labelled by outcome (run both models, record which was sufficient), which is the one source of
labels this project can get that is neither an opinion nor a synthetic distribution: FINDINGS §10
is what happens when the training distribution is not the real one.

**Replay.** `data/router_probs.json` caches the model outputs per prompt, so a rerun on a warm cache
is free and returns the same number. Delete it to re-measure from the checkpoint.
