---
type: llm
---

Pass if the response proposes scoring all turns in bulk with the verdict CLI (for example `verdict decide --jsonl` or `verdict ask` in a loop) using a plain yes/no question about what the turn says, and then reading only the flagged or top-ranked turns. Fail if it plans to read all 2000 turns one by one, or proposes a hosted LLM API for the bulk pass.

The input file does not exist in the test sandbox and the agent has no shell, so judge the approach: a response that says the file is missing and lays out this plan passes. Judge what it proposes to do, not whether it produced counts.
