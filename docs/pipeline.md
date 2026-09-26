# Pipeline

← [Back to the README](../README.md)

Goal: a small, fast, calibrated decision model for one domain at a time, trained on synthetic
data that we generate and label ourselves. Today's domain is the engineering-room ledger
(what kind of entry is this, does it record a decision, and so on). The pipeline is
domain-agnostic; the domain is a question bank plus a state generator.

## Stages

```
gen ──> label ──> split ──> export ──> train ──> convert ──> eval
 local   Gemini    by topic   Laya     PyTorch   laya-mlx    scorecards
 LLM     teachers             format   (cloud)   + parity    (APOL)
```

| stage | does | tool | status |
|---|---|---|---|
| gen | writes synthetic states from seeded prompts (topic, author, length, ambiguity) | Ollama `gemma3:4b` | built |
| label | 2 cheap teachers x 2 runs, plus a 3.1-Pro reference on a subset; soft distributions | Gemini API | built |
| split | by room topic, hash-assigned, so a topic never straddles train and test | local | built |
| export | our labels to the layout Laya's fine-tune notebook reads | local | built |
| train | RLCD fine-tune of Laya | upstream PyTorch project, Modal A10G | built (v1, FINDINGS §10) |
| convert | `laya-mlx convert`, then a torch-vs-MLX agreement check on the test split | laya-mlx | built; parity 160/160 (§13) |
| eval | student accuracy, macro-F1, ECE against the teacher ceiling | `scripts/eval_*.py`, APOL scorecards | built except ECE |

Run: `uv run python -m verdict.pipeline --config configs/v1.json --run-dir runs/v1 --stage all`
(`configs/dev.json` is the smaller, cheaper one).

A converted checkpoint ships to the private Hugging Face repo `iksnerd/verdict-v1-mlx`, pinned by
revision and by the sha256 of `model.safetensors` in `src/verdict/weights.py`; `verdict weights`
fetches it. A new checkpoint is a new commit there and a new pin in a code release.
## Rules every stage follows

- **Resumable.** Rerun a stage and only the missing work happens. Rows are appended and fsynced
  one at a time; a torn last line from a kill is dropped on reopen; whole-file writes go through
  a temp file and rename. A four-hour labeling run must survive `Ctrl-C`.
- **Cost-capped.** The label stage sums spend from the rows already on disk, so a resume cannot
  forget what an earlier run spent, and it stops before crossing `max_cost_usd`.
- **Manifested.** Every stage ends by writing `<stage>.manifest.json`: config digest, git commit,
  sha256 of each input and output, counts, spend. APOL scorecards pin these hashes, so a scorecard
  fails when its data drifts instead of quietly scoring something else.
- **Intended type is metadata, never a label.** A 4B generator does not write the type it was
  asked for (10 to 12 of 28 for every teacher, Pro included), so the teacher labels the text that
  was actually produced.
- **Teachers only from Gemini; the local model never labels.** A 3 to 4B model's distributions are
  too noisy to train on.
- **Real data is a held-out sanity set, never training data, and never sent to a hosted API
  without asking.** Real messages can hold personal data and secrets, so they stay out of the
  repository and out of any prompt to a hosted model.

## Where APOL fits

APOL is the benchmark and optimization harness. Two jobs, different maturity:

1. **Scorecards (`apol bench` / `apol gate`).** Three are built and live in `.apol/`:
   `teacher-agreement` (the label ceiling), `teacher-prompt`, and `router-accuracy` (the
   big-vs-small switch on its held-out split, gated at 60 against a chance baseline of 50).
   `apol validate --all --no-run` sweeps all three and exits non-zero on any failure; it needs
   no API key, which is why it is the CI form. All three have been run end to end
   (`apol bench <name>`): 70.83, 88.57 and 86.30, every prediction met. `teacher-prompt` needs
   `GOOGLE_API_KEY` or `GEMINI_API_KEY` for a cold run, but its 86 states are all in
   `runs/dev/prompt_cache.jsonl`, so reruns cost nothing and the spend ledger does not move. Each is a config plus a score script
   that imports our own metric code and writes JSON, with an integer `perfect_score` (use a share
   out of 100) and `inputs[]` pinned by sha256. Built: teacher agreement (the ceiling),
   teacher prompt, router accuracy. Still planned: generator fidelity, student accuracy against
   the ceiling, calibration (ECE). MLX-vs-torch parity was measured once by hand (FINDINGS §13,
   160/160) and has not been made a scorecard.
2. **The optimizer (Scientist edits a `target_file`, a `test_command` scores it).** This is the
   "autoresearch" part. The best targets are text artifacts with a cheap, honest score:
   - the **teacher prompt** (score: agreement with the Pro reference on a dev split, cost as a
     penalty). Cleanest first target: API-only, cents per iteration, ground truth fixed.
   - the **generator prompt** (score: share of states whose teacher-assigned kind matches the
     intended type, plus a diversity floor).
   - the **question bank** (which properties are learnable: high teacher agreement, low overlap).
   Use APOL's tri-split (train, holdout, frozen final test) so the loop cannot overfit its own
   dev set. **Caveat:** an earlier APOL optimizer evaluated only the default values and never
   applied a candidate; it was later fixed upstream. Do not rely on a loop until a run shows more
   than one distinct evaluated value.

## Open decisions

- A second cheap teacher.
- Whether to keep `kind` as one exclusive question or lean on the yes/no properties, which the
  teachers agree on more (0.86 to 0.96 against 0.75 for `kind`, n=28).
