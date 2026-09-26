---
type: llm
---

Pass if the response uses or proposes the verdict CLI to score every command with a question about secrets or credentials (a plain question about what the command text shows), ranks or cuts the results, and has a person or the agent review the flagged ones. Fail if it only offers a hand-written regex with no scoring, or plans to read all 600 itself.

The input file does not exist in the test sandbox and the agent has no shell, so judge the approach: a response that says the file is missing and lays out this plan passes. Judge what it proposes to do, not whether it produced counts.
