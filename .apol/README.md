# APOL Benchmarks

Project-local APOL assets live here so APOL core remains data- and model-agnostic.

- `benchmarks/` contains strict APOL config JSON.
- `datasets/` would hold benchmark data owned by this repo. Unused here: this project's
  scorecards read their inputs from `evals/` (the router prompts, committed) and `runs/`
  (pipeline output, gitignored), pinned by sha256 in each config's `inputs[]`. The two
  `apol init` sample files that used to sit here said "example input" / "example output"
  and nothing read them, which is the same trap `teacher-prompt` fell into (FINDINGS §16).
- `targets/` contains prompts, policies, schemas, configs, or placeholders.
- `runs/` contains local benchmark outputs and is usually ignored.

See `docs/benchmark-recipes.md` in the APOL repo for benchmark design recipes.
