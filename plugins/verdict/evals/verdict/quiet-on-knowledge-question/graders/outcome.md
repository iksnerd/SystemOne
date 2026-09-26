---
type: llm
---

Pass if the response answers the database question from knowledge (sqlite is likely fine at that scale, with the trade-offs). Fail if it sets up verdict or any model to score the question: it needs knowledge and reasoning, not a surface reading of text.
