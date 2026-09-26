# verdict-release: evolution

### 2026-09-24: created from seven releases
- **Trigger:** 0.5.0 through 0.8.2 were cut in one session, each by hand from a CLAUDE.md
  paragraph, and most produced a lesson: the scorecard's priority, library citations, the stale
  shared index after a private-index commit, a CI failure from host-dependent tests (a pushed
  tag is never moved), uv's git cache lagging a new tag, and six tracebacks found only by trying
  bad input on the installed tool. The global rule is to propose a skill for a procedure run a
  second time; this one ran seven.
- **Home:** a project skill in the verdict repo (`.claude/skills/`), not the personal plugin:
  it only works for this repo. skill-repository-builder's rule: one project's paths and workflow
  make it project-scoped.
- **Score:** rubric v2: 88, self-scored, **unverified** (no trigger evals, F 0). A 30, B 20,
  C 20, D 15, E 3 (no single worked example). Biggest gap: F1 trigger evals.
- **Outcome:** Accepted.

### 2026-09-24: a new checkpoint is a Hugging Face commit, not a GitHub release
- **Trigger:** before the repo went public, the weights moved from the `weights-verdict-v1`
  release asset to the private Hugging Face repo `iksnerd/verdict-v1-mlx` (kept private: Gemini
  labels). The last section still described `gh release create`.
- **Change:** that section now covers uploading to the Hub, pinning revision and sha256, and
  proving the fetch with and without a token. The description's near-miss is reworded to match.
- **Score:** rubric v2: 88 → 88; one section rewritten in place, no check moves.
- **Outcome:** Accepted.

### 2026-09-26: update the installed plugin as part of the release
- **Trigger:** 0.11.0, 0.11.1 and 0.12.0 each bumped `plugins/verdict/.claude-plugin/plugin.json`,
  but the installed `verdict@verdict` stayed at 0.10.0 until someone asked why there was no
  plugin. Sessions here kept a skill that told agents to run the now-refused `guard` preset.
- **Change:** step 8 updates the marketplace and the plugin and checks the version; "Done when"
  requires `claude plugin list` to show NEW.
- **Score:** rubric v2: not rescored; one step and one done criterion, validator clean before
  and after.
- **Outcome:** Accepted.

### 2026-09-26: no weights step
- **Trigger:** `verdict weights` was removed and base Laya became the default.
- **Change:** step 8 no longer runs `verdict weights --check`; the checkpoint section says the
  fine-tune is fetched by hand to wherever `model.path` points, and a public one would need
  redistributable labels.
- **Score:** validator clean; not rescored.
- **Outcome:** Accepted.

### 2026-09-26: present tense, no dead version numbers
- **Trigger:** the history restart left the skill citing versions that no longer exist
  (0.5.0 to 0.12.0) and a conftest isolation that went with the weights download.
- **Change:** each lesson kept, told without version numbers; the CI-failure note says what
  conftest isolates now (the machine's config).
- **Score:** validator clean; not rescored.
- **Outcome:** Accepted.
