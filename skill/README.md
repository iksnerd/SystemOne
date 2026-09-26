# The verdict skill moved

The agent skill for verdict now lives in this repo as a Claude Code plugin, in
[`plugins/verdict/`](../plugins/verdict/skills/verdict/SKILL.md). Install it with:

```sh
claude plugin marketplace add iksnerd/verdict
claude plugin install verdict@verdict
```

An older copy lived here as `skill/verdict/` and was retired on 2026-09-24 after drifting into
contradiction with the maintained one; `git log -- skill/verdict` has its history.
