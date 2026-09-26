# Examples

Four runnable examples. Every input is invented, and no real data is in this folder.
`tests/test_examples.py` checks that each bank still validates and that its inputs have the fields
the questions name.

Run any of them with a server up (`verdict serve`, then stop it when you're done):

```sh
verdict decide --jsonl -q examples/<name>/bank.json < examples/<name>/<inputs>.jsonl
```

Check a bank without loading a model using `verdict validate -q examples/<name>/bank.json`;
add `--json` for an agent-readable validation result. Add `--server-only` to `decide` to fail
with exit status 2 if the server is unavailable, instead of loading a local fallback model.

The output below is real, from the fine-tuned checkpoint, and includes the misses, because
knowing what it gets wrong is the point.

| example | question types | what it shows |
|---|---|---|
| `room-triage/` | three yes/no | the strongest use: yes/no about what a status update says |
| `secret-commands/` | one yes/no, plus a labelled file for `calibrate` | ranking, and fitting a cut |
| `support-tickets/` | choice, score and yes/no in one bank | all three types, with a structured state |
| `ticket-search/` | five yes/no, ranked with `verdict rank` | named answers as a searchable vector |
| `commit-kinds/` | seven-way choice | where a small LLM beats verdict (FINDINGS §32) |

## room-triage/

`updates.jsonl` holds six status updates, and `bank.json` asks whether each is done, blocked or
names a next step.

```
done 0.73  blocked 0.18  next 0.29  | Shipped: v2.1 is tagged and pushed, CI green, ...
done 0.30  blocked 0.85  next 0.25  | Blocked on the vendor's API key; asked them on Monday, ...
done 0.20  blocked 0.45  next 0.87  | Next: add the retry wrapper around the upload client, ...
done 0.70  blocked 0.15  next 0.16  | Merged the fix and closed the issue. Nothing left open here.
done 0.27  blocked 0.60  next 0.58  | Draft RFC posted for review, waiting on feedback ...
done 0.26  blocked 0.70  next 0.37  | Working tree has the refactor, not committed yet; ...
```

The top score in each column is the right row. `done` is a measured library question and is asked
as a yes/no; `blocked` and `next` are asked as no/yes choices, which spread the right rows further
from the rest (0.72 to 0.85 and 0.87 against the plain yes/no). Read the columns as rankings: a
0.60 does not mean "blocked" until you have fitted a cut.

## secret-commands/

`commands.jsonl` has twelve commands; `labelled.jsonl` has 24 more with a `label`, for `calibrate`.

```
0.69  echo $STRIPE_SECRET_KEY
0.63  gh auth token
0.59  cat ~/.aws/credentials
0.59  printenv | grep TOKEN
0.54  grep API_KEY .env.local
0.40  git log --oneline -5
0.31  ls -la src/
0.27  vercel env pull --environment=production .env     <- a miss
...
```

The top five are right. `vercel env pull` writes production secrets to a file, but it names no
secret, so a surface reader ranks it low. That is the same miss it made on 400 real commands
(FINDINGS §29). Then fit a cut:

```sh
verdict calibrate examples/secret-commands/labelled.jsonl -q examples/secret-commands/bank.json --out fit.json
# secrets: cut 0.4614  held-out AUC 0.9  balanced accuracy 0.9  (n=24, 12 positive)
verdict ask '{"command": "cat ~/.netrc"}' "Does \`command\` read, print or change a secret, key, token or credential?" --cut fit.json
```

24 examples is a demonstration of the shape, not a fit to trust. Use a few hundred real labels.

## support-tickets/

One bank asks all three types at once, about a `message` field.

```
refund   urgency 1.93/3  churn 0.57  | I was charged twice ... refund ... or I'm cancelling.
bug      urgency 2.15/3  churn 0.14  | The dashboard shows a blank page ... whole team is blocked.
question urgency 1.77/3  churn 0.25  | How do I export my data as CSV?
bug      urgency 1.72/3  churn 0.23  | Checkout fails with error 502 every time, we launch tomorrow.
bug      urgency 1.55/3  churn 0.72  | Thinking about switching to a competitor ...    <- intent is wrong
```

`churn` ranks the two right messages on top. `intent` is right for refunds, bugs and questions,
but calls the competitor message a bug. `urgency` is squeezed into 1.3 to 2.2 and puts "we launch
tomorrow" below a CSV question: an ordinal scale on a small sample is the weakest of the three.

## ticket-search/

Sixteen tickets, each asked five yes/no questions: about money, says something is broken, says
the customer will leave, mentions a deadline, asks how to do something. The answers are a vector
with a name on every number; `verdict rank` searches it:

```sh
verdict decide --jsonl -q examples/ticket-search/bank.json < examples/ticket-search/tickets.jsonl > scored.jsonl
verdict rank scored.jsonl says_broken=1 has_deadline=1 -k 3
```

```
 1.67  says_broken +1.00  has_deadline +0.67  | Checkout fails with error 502 every time, we launch tomorrow.
 1.60  has_deadline +0.93  says_broken +0.67  | We need the SSO fix before our audit on Friday or we'll ...
 1.47  has_deadline +1.00  says_broken +0.47  | I was charged twice this month. Please refund ...  <- not broken
```

The two right tickets are on top. The third is there on its deadline ("today"), and the numbers
say so. Unscaled, `says_broken` (up to 0.92) swamped `has_deadline` (at most 0.65) and put the
charged-twice ticket second; `rank` rescales each dimension to its percentile first (FINDINGS §39).
Some dimensions miss: "we launch tomorrow" scored 0.27 on `has_deadline` and "the money still left
my account" 0.41 on `about_money`, so check each question's spread before trusting a sum of them.

## commit-kinds/

A seven-way choice, included as the counter-example.

```
feat      | add CSV export to the reports page
fix       | handle nil pointer when the router config is empty
docs      | explain the retry policy in the README
refactor  | split the 600-line handler into three modules
test      | cover the empty-input path in the parser
fix       | bump fastapi to 0.120                        <- should be chore
test      | run the test suite on tag pushes only        <- should be ci
```

Five of seven are right. The two misses are the ones that need knowing what the words refer to.
On 125 real commits, Claude Haiku 4.5 scored 0.67 here against verdict's 0.48 (FINDINGS §32), so
for categories like these, use a small LLM if the data may leave the machine.
