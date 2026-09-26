---
name: verdict-release
description: >-
  Cut a verdict release (a vX.Y.Z tag on iksnerd/verdict): bump, answer-quality scorecard,
  commit, tag, CI, then install it the way a user would and try bad inputs. Use when asked to
  release, tag, cut or ship a verdict version, or to get a change onto the global `verdict`.
  Not for uploading a new model checkpoint (a Hugging Face commit, see the end).
metadata:
  version: "1.0.0"
---

# Releasing verdict

A release is a `vX.Y.Z` tag. `.github/workflows/release.yml` runs the suite on macOS arm64,
checks the tag against `pyproject.toml`, refuses a tag without a committed scorecard for that
version or one below the previous scorecard's interval, and publishes the wheel. The global
`verdict` is a `uv tool` install of the newest tag, so nothing reaches it any other way.

Every step and check below comes from a release that went wrong without it.

## Before you start

- `git status --short` is empty apart from your change, and `git fetch -q && git status -sb`
  shows main in sync. **Another session may share this checkout** (`ListAgents`): if its work
  is in the tree, tell it what you will commit, and commit only your paths (step 5).
- Semver: a new setting, command or flag is a minor bump; a fix is a patch.

## Steps

1. **Bump the version** in the four places tests keep equal (the agent skill's plugin too:
   a plugin version left behind makes `claude plugin update` serve the old skill):
   ```sh
   sed -i '' 's/^version = "OLD"/version = "NEW"/' pyproject.toml
   sed -i '' 's/verdict.git@vOLD/verdict.git@vNEW/' README.md docs/guide.md
   sed -i '' 's/"version": "OLD"/"version": "NEW"/' plugins/verdict/.claude-plugin/plugin.json
   uv lock -q && uv sync -q --extra mlx --extra laya --extra bench
   uv run verdict --version            # must print NEW
   ```
   A plain `uv sync` drops the extras and silently skips the laya tests.

2. **One model at a time.** `ollama ps` and `pgrep -fl 'verdict.*serve'` must show nothing else
   loaded (machine-specific exceptions go in `CLAUDE.local.md`).

3. **Scorecard, at normal priority, with bench's own 50 ms pause:**
   ```sh
   env -u VERDICT_BITS -u VERDICT_LANG uv run verdict bench --out bench/scorecards/vNEW.json
   uv run verdict bench --verify bench/scorecards/vNEW.json     # "... against vPREV: ok"
   ```
   About 230 s: the bg- and ru- suites load the multilingual checkpoint too, so both are resident
   (about 1.4 GB). Not under `taskpolicy -b`: that took 7 minutes at 15% CPU. Unset
   `VERDICT_BITS`, or the scorecard measures the 8-bit model (`bits` is recorded in it).
   With no model change, the values match the previous scorecard exactly; if they don't, stop
   and find out why before tagging.

4. **Point the library at the new scorecard:**
   `sed -i '' 's/scorecard vPREV"/scorecard vNEW"/' src/verdict/library.json`.
   A test fails until bench-backed entries quote the latest scorecard.

5. **Commit through a private index**, only your paths, then re-sync the shared index:
   ```sh
   IDX=$(mktemp -t verdict-idx); rm -f "$IDX"; env -u GIT_INDEX_FILE git reset -q
   GIT_INDEX_FILE="$IDX" git read-tree HEAD
   GIT_INDEX_FILE="$IDX" git add -- pyproject.toml uv.lock README.md docs/guide.md \
       src/verdict/library.json bench/scorecards/vNEW.json plugins/verdict/.claude-plugin/plugin.json
   GIT_INDEX_FILE="$IDX" git commit -m "<type>: <what shipped>, NEW"
   rm -f "$IDX"; env -u GIT_INDEX_FILE git reset -q
   env -u GIT_INDEX_FILE git diff --cached --stat     # must print nothing
   ```
   The trailing reset is not optional: skipping it once left the shared index showing the
   release files as staged *reverts*, one peer commit away from undoing the release. If a
   file you touch also holds a peer's edits, stage a blob built from `git show HEAD:<file>`
   plus only your change (`git hash-object -w`, `git update-index --cacheinfo`). The
   pre-commit hook tests the index snapshot, so it proves the commit, not your working tree.

6. **Push and tag**, guarding against a moved remote:
   ```sh
   git fetch -q && git merge-base --is-ancestor origin/main HEAD \
     && git push -q origin main && git tag vNEW && git push -q origin vNEW
   ```

7. **Watch CI** (about a minute): `gh run list -R iksnerd/verdict --limit 3`, then
   `gh run watch <id> --exit-status`. Every step must pass, "Answer-quality scorecard"
   included.
   - **If CI fails, never move the pushed tag.** Fix on main and release the next patch; the
     failed tag stays with no release, and `verdict update` skips it. A test that passes only
     because of something on the release machine (weights, a config file) fails here first;
     `tests/conftest.py` keeps the machine's config away from every test.

8. **Install as a user would:**
   ```sh
   cd /tmp && verdict update && verdict --version      # must print NEW
   ```
   `update` retries once by itself if uv's git cache lags the fresh tag ("Could not parse
   object"); a second failure is real.

   Then move the installed agent skill too. The tool and the plugin update separately, and
   nothing else moves the plugin: without this step, releases ship while sessions keep an old skill
   that tells agents to do things the tool now refuses.
   ```sh
   claude plugin marketplace update verdict && claude plugin update verdict@verdict
   claude plugin list | grep -A1 'verdict@verdict'     # Version: NEW
   ```

9. **Exercise the change on the installed tool, including bad input.** Run the new command or
   flag, then feed it something wrong and check for a one-line `verdict:` error with exit 2,
   never a traceback. This step has found crashes the tests missed (`bits = 4`, a portless URL,
   a TOML typo), because the tests exercised the happy path.

## Done when

The GitHub release exists with the wheel and sdist, `verdict --version` prints NEW from
`/tmp`, `claude plugin list` shows `verdict@verdict` at NEW, the change behaves on the installed tool, nothing of yours is running, and
`git status --short` shows no files of yours.

## A new checkpoint is a different release

verdict defaults to base Laya and has no command to download a fine-tune. The private fine-tune
lives in the Hugging Face repo `iksnerd/verdict-v1-mlx`; a new one is a new commit there (upload
with a token that can write), fetched by hand to wherever `model.path` points. Keep it private:
it was trained on Gemini labels (Google's terms bar building competing models). A fine-tune
meant for the public needs redistributable labels and would be pinned in a code release.
