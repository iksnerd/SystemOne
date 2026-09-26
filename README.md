# verdict

A System 1 for agents, in Kahneman's sense: fast, automatic judgments with a number attached. Ask
a typed question about a piece of text (yes/no, pick one, or a level) and get a probability per
option in tens of milliseconds, from a local encoder. No generated text, nothing to parse, and
nothing run on your behalf: verdict answers, and the caller acts.

The slow, deliberate System 2 is whoever calls it, a person or an LLM agent. verdict makes the
same quick call over hundreds of items, and System 2 reads only what it flags. Like any System 1 it
is confident whether or not it is right, so a score is not a decision until it has been checked
against labels.

**What it is good at, measured** ([FINDINGS §32](docs/FINDINGS.md)): on a yes/no question about
what the text shows, it matched Claude Haiku 4.5 on real data (0.77 against 0.73 AUC), for free,
on the machine. On sorting into seven categories, Haiku won clearly (0.67 against 0.48), and
verdict barely beat keyword rules. So use it for bulk yes/no about the surface of the text, and
for data that must stay local. Use a small LLM for categories that need world knowledge.

## Quick start

Apple Silicon, Python 3.11.

```sh
uv tool install --python 3.11 'verdict[mlx,laya] @ git+https://github.com/iksnerd/verdict.git@v0.2.0'
verdict init        # fetches the weights if you have access, picks a free port, writes the config
verdict serve &     # holds the model, so each call costs milliseconds

verdict ask "commit the fix and push it" "Is this an instruction to perform an action?"
# 0.79 with the fine-tune (see "Which model you get")
verdict decide --jsonl -q examples/room-triage/bank.json < examples/room-triage/updates.jsonl

pkill -f 'verdict serve'   # stop it when you're done; it holds the model in GPU memory
```

### Which model you get

verdict runs [Laya](https://huggingface.co/convaiinnovations/laya) checkpoints. The author's
fine-tune, `verdict-v1`, is **not public**: it was trained on Gemini-generated labels, and
Google's terms bar using Gemini to build competing models, so its weights stay in a private
Hugging Face repo. verdict uses **base Laya** (`aac6fef/laya-mlx`, Apache-2.0) by default, and
it downloads itself on first use. `model.path` in the config points at any other checkpoint.

What that costs, measured on the same 1,640 public items ([FINDINGS §35](docs/FINDINGS.md)):

| accuracy unless marked | fine-tune | base Laya |
|---|---|---|
| a named two-option choice (`-o positive=... -o negative=...`): sentiment, spam, injection | 0.91, 0.97, 0.83 | 0.91, 0.96, 0.81 |
| topics (4-way), emotion (6-way) | 0.95, 0.47 | 0.94, 0.48 |
| a plain yes/no, e.g. "Is `text` positive?" (AUC) | 0.80 | 0.50 |
| the same yes/no as verdict now asks it, a no/yes choice (AUC): sentiment, spam, injection | 0.96, 0.99, 0.91 | 0.94, 0.99, 0.88 |

The difference was the plain yes/no, and verdict no longer sends one for a question you write:
it asks a new yes/no as a choice between no and yes, where base Laya is within noise of the
fine-tune ([§40](docs/FINDINGS.md)). So on base Laya:

- **Write your own questions.** A yes/no, a named choice (`-o`) or a score all work.
- **The measured library questions** (`is_instruction`, `touches_secret` and the rest) are asked
  as a no/yes choice too. On the fine-tune they stay plain yes/no, the form their real-traffic
  numbers were measured in; on base Laya that form is the one that collapses. Their quoted
  numbers are the fine-tune's, so treat them as a guide, not a guarantee.
- **Rank, don't threshold,** until you have fitted a cut with `verdict calibrate` on your own
  labels. The cuts quoted in the docs are the fine-tune's.

The example outputs in this README and in `examples/` come from the fine-tune, and so do the
release scorecards.

After the first install, `verdict update` moves you to the newest release; a test keeps
the tag in the install line above equal to the current version.

## What it does

- **Three question types** (`ask` inline, `decide` with a JSON bank):
  - `noul`, yes/no: the probability the statement holds
  - `choice`: one of several options, with a probability each
  - `score`: an expected level on an ordinal scale
- **Checks before inference:** `validate -q bank.json --json` checks a bank without loading a
  model. `ask`, `decide`, `calibrate` and `bench` accept `--server-only` for agents that must fail instead
  of loading a local fallback. See [scripts and agents](docs/guide.md#scripts-and-agents).
- **Bulk scoring:** `decide --jsonl` streams one bank over thousands of states. It prints each
  question's spread and warns when a range is too narrow to rank anything.
- **Calibration:** `calibrate` fits a cut per yes/no question and a temperature per choice on
  labelled examples. It reports both on a held-out split, with recall per option for choices.
- **Measured questions by name:** `verdict questions` lists the ones that have worked, each with
  its result; `-q is_instruction,touches_secret` uses them.
- **Refuses what measured at chance:** `ask` and `decide` exit 2 on a question about a
  consequence, difficulty or risk, a yes/no about something being ordinary, a choice with more
  than 20 options, or a state missing a field the question names. A state far past the
  128-token read gets a warning. `--allow-unmeasured` asks anyway; `calibrate` only warns, since it
  is how a question gets measured.
- **A new yes/no is asked as a no/yes choice** and answered as a yes/no: SST-2's collapsed
  "Is `text` positive?" went from 0.79 to 0.96, others unchanged ([§38](docs/FINDINGS.md)).
  On the fine-tune, measured library questions keep their shape; `--yesno` opts out.
- **The docs travel with the tool:** `verdict docs` prints this README, `verdict docs guide` the
  guide, and `verdict docs findings 38` one of the FINDINGS sections the help and errors cite.
- **Search by named answers:** `verdict rank scored.jsonl about_money=1 says_leaving=0.5` ranks a
  `decide --jsonl` run by a weighted sum of rescaled answers and shows each one's share
  ([§39](docs/FINDINGS.md)).
- **A release gate on answer quality:** `verdict bench` scores eight pinned public suites, and a
  release needs a committed scorecard in `bench/scorecards/`.
- **Structured state:** JSON objects with named fields, as Laya recommends.
- **Laya's presets:** `triage`, `moderation` and `email`; `guard` and `router` ask about
  consequences and difficulty, so they need `--allow-unmeasured`. `verdict presets` lists them.
- **Half the memory when it matters:** `bits = 8` in the config quantizes the model as it loads,
  about 430 MB of GPU memory instead of 800, with the same answers (FINDINGS §36).
- **Non-English input** is detected and warned about. `--lang multi`, or `lang = "multi"` in the
  config, switches to Laya's multilingual checkpoint. On Cyrillic it reads Bulgarian and Russian
  reviews at 0.88 to 0.93 accuracy, where the English checkpoint is near chance (FINDINGS §37).
- **An HTTP API** on localhost ([docs/api.md](docs/api.md)), with the same wire format as Laya's
  `Agent.predict`, and TypeSafe Jev's `/v1/systemone`: Jev's SDKs work against it with
  `TYPESAFE_BASE_URL`.

## How it works

`verdict` asks a local server (`verdict serve`), or loads the model itself if none is running.
The model is [Laya](https://huggingface.co/convaiinnovations/laya), a ModernBERT encoder trained
to answer typed questions in one forward pass. It is fine-tuned here and runs on Apple Silicon
through MLX. Each state is clipped to 128 tokens (1.9x faster, no measured loss). Every
question costs one pass, so a bank should hold the one or two questions you need.

## Documentation

- [docs/guide.md](docs/guide.md) is how to use it: question types, bulk runs, calibration, reading
  the numbers, which tool for which job, recipes and troubleshooting.
- [examples/](examples/README.md) has four runnable examples with invented data, and their real
  output, misses included.
- [docs/api.md](docs/api.md) covers the HTTP API.
- [docs/routing.md](docs/routing.md) covers the switch and the big-or-small router built on it,
  kept as a worked example.
- [docs/FINDINGS.md](docs/FINDINGS.md) has every experiment, with sample sizes and intervals.
  Every number in these docs comes from it.
- [docs/pipeline.md](docs/pipeline.md) covers building a checkpoint: generate, label, split,
  export, train, convert.

## Developing

```sh
uv sync --extra mlx --extra laya     # plain `uv sync` drops both extras and breaks the CLI
uv run pytest                        # the whole suite; no network and no model needed
git config core.hooksPath .githooks  # pre-commit: blocked paths, gitleaks, tests on the staged snapshot
uv run verdict ...                   # runs your working tree, not the installed release
```

Releases are tags. Bump `version` in `pyproject.toml`, run
`uv run verdict bench --out bench/scorecards/vX.Y.Z.json` (needs `--extra bench`), commit both, then
`git tag vX.Y.Z && git push origin vX.Y.Z`. The release workflow runs every test on macOS with
both extras, checks the tag against the version, refuses a missing scorecard or one below the
previous scorecard's interval, and publishes the wheel. `verdict update` then installs it. The
release holds code only; base Laya downloads itself from Hugging Face.

What is where:

| path | holds |
|---|---|
| `src/verdict/cli.py` | the `verdict` command |
| `src/verdict/inputs.py` | states, banks, presets, field and language checks |
| `src/verdict/library.py`, `library.json` | the measured questions behind `verdict questions`, and the linter |
| `src/verdict/bench.py`, `bench_suites.json` | `verdict bench` and its pinned suites; scorecards in `bench/scorecards/` |
| `src/verdict/calibrate.py` | cuts, temperatures, AUC and per-option recall, pure Python |
| `src/verdict/schema.py` | the wire contract, mirroring Laya's |
| `src/verdict/backend_mlx.py`, `engine.py` | loading the checkpoint, clipping, answering |
| `src/verdict/api.py`, `client.py` | the HTTP server and the CLI's client for it |
| `src/verdict/switch.py`, `router.py` | the switch and the router example |
| `src/verdict/pipeline/`, `gen/`, `label/`, `train/` | building a checkpoint |
| `examples/` | runnable banks and synthetic inputs, checked by `tests/test_examples.py` |

**For agents:** the verdict skill ships in this repo as a Claude Code plugin. It tells an agent
when to reach for verdict, which questions have been measured, and how to read the numbers:

```sh
claude plugin marketplace add iksnerd/verdict
claude plugin install verdict@verdict
```

## Status, honestly

- **Useful now:** bulk yes/no about the surface of the text. On real data it matched Haiku 4.5 at
  0.77 AUC (§32), and triaged 137 rooms of a multi-agent message log in 41 s (§30).
- **Weak:** many-way categories (0.48 on seven commit types, near keyword rules), and ordinal
  scores on small samples.
- **At chance:** the big-or-small router on real agent traffic (§25). It stays as an example only.
- **Not usable for typing long real messages:** the v1 fine-tune scored 0.381 on 360 real
  multi-agent log messages against 0.167 chance. It was trained on short synthetic states (§3).
- **Scores are uncalibrated** until you run `calibrate` on your own labels. The raw calibration
  error has not been measured as a scorecard.

## Conventions

Tests first; `uv run pytest` is the gate, and nothing in it needs the network or a model. Teacher
labels come from Gemini only. Real data is a held-out sanity set, never training data, and is not
sent to a hosted API without asking. A `uniform` answer means "no model", not "unsure", and never
gates a decision. verdict answers and never acts on an answer.
