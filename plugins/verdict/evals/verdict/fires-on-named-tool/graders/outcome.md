---
type: llm
---

Pass if the response sets up a verdict choice question with the three options (bug, feature request, question) over issues.jsonl, for example with `verdict decide --jsonl` and a bank file, and treats the result as a ranking to spot-check rather than ground truth. Fail if it classifies the issues by reading them itself without verdict.

The input file does not exist in the test sandbox and the agent has no shell, so judge the approach: a response that says the file is missing and lays out this plan passes. Judge what it proposes to do, not whether it produced counts.
