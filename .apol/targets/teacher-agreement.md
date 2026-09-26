# teacher-agreement

Read-only scorecard: nothing here is optimized. The target file is a placeholder.

**What it scores.** The share of question decisions on which the mean of the two cheap teachers
(`gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`, two runs each) matches `gemini-3.1-pro-preview`:
same argmax for `kind`, same side of 0.5 for the four yes/no properties.

**Why this metric.** It is the ceiling for a student trained on cheap-teacher labels, and the one
number that says whether the labelling recipe is trustworthy before any GPU time is spent. A
fine-tune that scores above it is fitting teacher noise.

**Caveats.** n=28 synthetic states, so the interval is wide (about +-8 points). The reference is
one Pro run, itself imperfect. The states were written by a 4B model and are not the real ledger:
agreement on the real messages has not been measured because sending them to Gemini was not approved.
The `kind` question agrees least (75 to 82) and the yes/no properties most; the per-question numbers
are in the run JSON.
