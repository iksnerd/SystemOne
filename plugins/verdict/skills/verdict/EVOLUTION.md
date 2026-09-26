# verdict skill: evolution

Append-only, oldest first. Read before changing the skill.

### 2026-09-22: created
- **Trigger:** verdict became a general CLI for agents; nothing told an agent when to reach for it
- **Change:** the skill: bulk use (30+ items), one model at a time, surface questions, calibrate before gating
- **Outcome:** Accepted

### 2026-09-22: check who wrote the inputs, and each question's spread
- **Trigger:** the first status triage scored a linter's boilerplate (99 of 137 rooms); `blocked` spanned 0.58 to 0.65
- **Change:** a bullet on filtering bot authors and reading spreads. The spread check then moved into verdict itself (0.1.0: `decide --jsonl` warns under 0.1), so the note became a check
- **Outcome:** Accepted; escalated to code the same day

### 2026-09-23: install is a uv tool from a release tag
- **Trigger:** the ~/.local/bin shim tied the CLI to the checkout; a global install was wanted
- **Change:** install and update text; `verdict update` follows release tags
- **Outcome:** Accepted

### 2026-09-23: a measured-question library and an after-every-use loop
- **Trigger:** each use produced a measurement that lived only in FINDINGS and one session's context, so the next session started from zero; the skill had no self-improvement loop
- **Change:** `references/questions.md`, read before writing a bank and appended after every use; an "After every use" section routing each lesson to the tool, this skill, or the consumer skill
- **Score:** rubric v2: 77 → 83 (C +3, F +2, B +1); both valid, both unverified (no trigger evals). Biggest gap: F1, trigger evals, +6
- **Outcome:** Accepted

### 2026-09-23: trigger evals, run; a line between reading and scoring
- **Trigger:** F1 was the biggest gap (no trigger evals). Wrote 5 cases: three should-fire (bulk transcript turns and a command log, neither naming verdict; one naming it) and two near-misses (a single commit message; a status-room triage that a dedicated skill owns)
- **Evidence, first full run** (`claude plugin eval <plugin> --tag verdict --ablation none`, 3 runs a case, $1.86): **trigger rate 9/9, false triggers 0/6**. Outcomes 11/15. One was a real skill gap: on the 600-command log the skill loaded, then the agent decided "600 lines is few enough that I'll read them all", because the skill said "many items" and drew no line. Three were grader flaws: the sandbox has no input files and no shell, and one judge read a one-line caveat as "not direct"
- **Change:** a "Where the line sits" paragraph: read a few dozen, score from about a hundred up, and score below that when the judgment must repeat or needs a consistent number. Graders now judge the approach when the sandbox can't run it
- **Evidence, rerun of the whole suite** ($1.92, results 2026-09-22T23-07-55-219Z): trigger 9/9, false triggers 0/6, **outcomes 15/15**; the command-log case went 2/3 → 3/3
- **Score:** rubric v2: 83 → 91 (F +8: recorded trigger run and a measured before/after; the unverified cap no longer applies). Self-scored
- **Outcome:** Accepted

### 2026-09-23: route by job: verdict for surface yes/no, a small LLM for categories
- **Trigger:** a head-to-head on the same held-out items (verdict FINDINGS §32). On the secrets yes/no, verdict 0.771 AUC, Haiku 4.5 0.727, regex 0.640. On the 7-way commit type, Haiku 0.672, verdict 0.480, keyword rules 0.464
- **Change:** a "Pick the right fast model" paragraph in the body; the commit-type question moved to the Failed table with the rival numbers; a pattern line: always compare against a keyword baseline. Description unchanged, so the recorded trigger run still stands
- **Score:** not rescored; guidance added inside the existing structure, no trigger or layout change
- **Outcome:** Accepted


### 2026-09-24: a worked example; a pointer to the human docs; the repo's old copy retired
- **Trigger:** the rubric's E dimension (examples) was 3/5, snippets only. The verdict repo held an older `skill/verdict/` that contradicted this one (routing-first, "never start the server", verdict for 40 items)
- **Change:** a worked example (300 issues: why verdict, which question, sample, spread, read the top, record a row), marked illustrative where its numbers are. A pointer to the repo's new `docs/guide.md` and `examples/`. The repo copy is now a pointer back here
- **Score:** not rescored; E should move 3 → 5, with no trigger or layout change, so the recorded trigger run stands
- **Outcome:** Accepted

### 2026-09-24: verdict 0.6.0: measured questions by name, named choices, Jev, misbehaviour
- **Trigger:** verdict 0.5.0 and 0.6.0 shipped a question library (`verdict questions`, `-q NAME`), a wording linter, FINDINGS §33 (a named two-option choice was best or tied on every public set, 0.94 to 0.98 AUC; the yes/no "Is `text` positive?" ranked at 0.79 with no positive over 0.5; yes/no "is it ordinary" at 0.32 to 0.64) and TypeSafe Jev's `/v1/systemone` protocol in `verdict serve`. The skill knew none of it
- **Change:** start from `verdict questions`; new binary questions as named choices; never a yes/no about something being ordinary; heed the linter; Jev-compatible serving via `TYPESAFE_BASE_URL`, with what does not carry over; a "When the tool misbehaves" list (uniform backend from `uvicorn verdict.api:app`, 2 s calls with no server, field and language warnings). `references/questions.md` gains the §33 rows and a pointer to `verdict questions`. Description unchanged, so the recorded trigger run stands
- **Score:** rubric v2: 87 → 95 (B +2, C +6), scored blind by a fresh subagent on both versions. Remaining gap A3 4/8: statically, a status-room triage could match both this and a dedicated triage skill; the recorded evals include that near-miss at 0 false triggers, and naming it would push the description past 700 chars (A4 −4), so it stays
- **Outcome:** Accepted

### 2026-09-24: verdict 0.7.1: fetch the weights; the base-laya warning
- **Trigger:** verdict 0.7.x ships the fine-tune as a release asset (`verdict weights`) and falls back to base laya, with a warning, when it is missing. The skill's reinstall line stopped at `uv tool install`, which now leaves a machine on the weaker fallback
- **Change:** the reinstall line ends with `verdict weights`; "When the tool misbehaves" gains the base-laya warning and what it costs (§35)
- **Score:** rubric v2: 95 → 95, self-scored; two facts inside existing sections, no trigger or structure change, so no check moves. Description unchanged; the recorded trigger run stands
- **Outcome:** Accepted

### 2026-09-24: verdict 0.8.x: validate first, --server-only in scripts, bits = 8 for memory
- **Trigger:** verdict 0.7.2 to 0.8.1 added `verdict validate` (checks a bank with no model), `--server-only` (a missing server fails with exit 2 instead of loading 843 MB into the caller's process) and `bits = 8` (half the GPU memory, same answers, §36). All three are for agents, and the skill mentioned none
- **Change:** "Before you load it" gains a first step (validate the bank), the memory option at the residency check, and `--server-only` at the serve step; the command block shows both
- **Score:** rubric v2: 95 → 95, self-scored; steps added inside existing sections, no trigger or structure change. Description unchanged, so the recorded trigger run stands
- **Outcome:** Accepted

### 2026-09-24: weights from Hugging Face, install over HTTPS (verdict 0.9.0)
- **Trigger:** before iksnerd/SystemOne went public, the weights moved from a GitHub release (`gh`)
  to the private Hugging Face repo `iksnerd/verdict-v1-mlx`, and install moved from `git+ssh` to
  `git+https`. The reinstall line still said `gh` and ssh.
- **Change:** the reinstall line names the HF repo, `HF_TOKEN` and where the token lives; the
  install URL is https.
- **Score:** rubric v2: 95 → 95; one line of facts, no check moves.
- **Outcome:** Accepted.

### 2026-09-24: lang = multi, and Cyrillic measured (verdict 0.10.0)
- **Trigger:** verdict 0.10.0 added `[model].lang` / `$VERDICT_LANG` and Cyrillic bench suites; FINDINGS §37 measured Bulgarian and Russian reviews: English checkpoint near chance (0.58 / 0.64), multilingual 0.93 / 0.88, with no yes/no lean
- **Change:** the non-English bullet names the setting and the Cyrillic numbers; `references/questions.md` gains the Cyrillic rows
- **Score:** rubric v2: 95 → 95; facts inside an existing bullet, no check moves
- **Outcome:** Accepted

### 2026-09-24: moved into the verdict repo as the canonical, public skill
- **Trigger:** iksnerd/SystemOne went public, and the agent guide belongs with the tool. Two full
  copies had drifted once before (the repo's old `skill/verdict/`), so the repo copy became
  canonical and a private copy kept only one machine's specifics.
- **Change:** the plugin `verdict` (`.claude-plugin/marketplace.json`, `plugins/verdict/`) carries
  this skill, its question log, this history and the trigger evals. Machine specifics were removed
  (checkout path, token location, local ports); base Laya for public installs, named choices and
  `VERDICT_MODEL=aac6fef/laya-mlx` were added; the worked example asks a named choice (§33). The
  status-room near-miss eval, which only applied to one machine, became a knowledge question
  (postgres or sqlite). `tests/test_plugin.py` ties the plugin version to the package
  version and fails on private paths in what ships.
- **Score:** rubric v2: 95 before the move; not rescored. The description changed slightly
  (latency figure, one path removed) and one eval case changed, so the recorded trigger run is
  for the old copy: rerun the suite from this repo before calling it verified.
- **Outcome:** Accepted.

### 2026-09-25: refusals and yes/no-as-choice (FINDINGS §38)
- **Trigger:** verdict now refuses (exit 2) questions about consequences, difficulty or absence and
  states missing a named field, and asks a new yes/no as a no/yes choice (§38: SST-2 0.79 to
  0.96, spam and injection unchanged). The skill still told agents to heed warnings, and its
  `-q guard` example would now be refused (guard's `harm_severity` asks what harm text could cause).
- **Change:** the question bullets describe the refusal and the automatic rewrite; the
  `-q guard` example became `-q triage`; the troubleshooting bullet names the refusal message.
- **Score:** rubric v2: 95 → 95; facts inside existing bullets, description and structure
  unchanged, validator clean before and after. Not rescored by a fresh subagent.
- **Outcome:** Accepted.

### 2026-09-25: `verdict rank` in the command list (FINDINGS §39)
- **Trigger:** verdict gained `rank`, which searches a `decide --jsonl` run by named, rescaled answers.
- **Change:** one line in the command list.
- **Score:** rubric v2: 95 → 95; one command line, validator clean.
- **Outcome:** Accepted.

### 2026-09-25: description's preset list drops guard
- **Trigger:** `guard` is now refused without `--allow-unmeasured` (its `harm_severity` asks what
  harm text could cause), and the description still told agents to run it.
- **Change:** the description's preset list is triage, moderation, email. A factual correction,
  not a scope change: the same requests trigger it.
- **Score:** rubric v2: 95 → 95; validator clean, description length unchanged within 5 chars.
- **Outcome:** Accepted.


### 2026-09-26: library yes/no questions rewritten on base Laya (FINDINGS §40)
- **Trigger:** measured base Laya with the no/yes rewrite (SST-2 0.51 -> 0.94, within noise of the
  fine-tune), and the library exemption now applies only on the fine-tune.
- **Change:** the yes/no bullet says library questions stay as measured on the fine-tune only.
- **Score:** rubric v2: 95 → 95; a clause in an existing bullet, validator clean.
- **Outcome:** Accepted.

### 2026-09-26: general facts moved in from a private skill
- **Trigger:** everything someone cloning iksnerd/SystemOne needs belongs in this plugin, not in a
  private companion skill. Three general facts lived only there: `verdict docs`,
  `update --check`'s exit code, and the multilingual checkpoint's English cost and library rewrite.
- **Change:** two command lines; the non-English bullet says when not to use `multi` and what it
  does to library questions; the §40 clause names every non-fine-tune checkpoint.
- **Score:** rubric v2: 95 → 95; facts in existing sections, validator clean.
- **Outcome:** Accepted.

### 2026-09-26: the five conditions and the 20-option refusal (FINDINGS §41)
- **Trigger:** §41 measured three tasks outside Laya's training (the fine-tune adds nothing there)
  and verdict now refuses a choice past 20 options and warns on long states.
- **Change:** the refusal bullet names the option limit and the long-state warning; the
  verdict-or-LLM paragraph states the five conditions under which verdict is the right tool.
- **Score:** rubric v2: 95 → 95; facts in existing sections, validator clean.
- **Outcome:** Accepted.

### 2026-09-26: base Laya is the default; no fine-tune download
- **Trigger:** verdict dropped `verdict weights` and made base Laya the default model (the fine-tune
  is private, and within noise of base once a yes/no is asked as a no/yes choice, §40).
- **Change:** the model bullet says base Laya is the default and why it is enough; the
  troubleshooting entry for "not found, so using base laya" now points at `model.path`.
- **Score:** rubric v2: 95 → 95; facts in existing bullets, validator clean.
- **Outcome:** Accepted.
