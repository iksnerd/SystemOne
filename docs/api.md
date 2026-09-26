# HTTP API

← [Back to the README](../README.md)

`verdict serve` exposes the same decisions over HTTP on `127.0.0.1:8799`. It binds to localhost and
has no auth, so do not expose it beyond the machine. The wire format mirrors Laya's
`Agent.predict(state, questions)`, so a Laya checkpoint drops in as a backend with no translation,
and Laya's mirrors TypeSafe's Jev, which `/v1/systemone` serves.

## `POST /v1/decide`

```json
{
  "state": {"message": "I was charged twice, please refund"},
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want in `message`?",
               "criteria": {"refund": "money back", "info": "a question", "other": "none fit"}},
    "urgent": {"type": "noul",   "instructions": "Is there time pressure in `message`?"},
    "anger":  {"type": "score",  "instructions": "How angry is `message`?", "criteria": ["calm", "annoyed", "angry"]}
  }
}
```

The response has one answer per question:

- a `choice` answer carries `choice`, `probabilities` per option and `confidence`
- a `score` answer carries `score` (the expected level), `legend`, `probabilities` and `confidence`
- a `noul` answer carries `noul`, the probability the statement holds

Plus `model`, the checkpoint that answered.

- `state` is a string, or better an object with named fields that the questions name in backticks.
- Each state is clipped to the server's token budget (128 by default; `verdict serve --budget 0`
  reads it whole). A state within budget passes through still structured.
- Optional `"model": "multilingual"` answers from Laya's multilingual checkpoint, loaded on first
  use. It is named as in Laya's `Router.predict(model=...)`. A server without one configured returns
  400.
- A `noul` question may describe its sides with `criteria: {"true": ..., "false": ...}`. On this
  project's data that hurt (§22), so leave them out unless you've measured otherwise.

## `POST /v1/systemone` and `GET /v1/models`: Jev's protocol

The same answers on [TypeSafe's Jev](https://docs.typesafe.ai/api.md) wire protocol, so tools written
for Jev can use verdict instead: the official `typesafe-sdk` (Python) and JavaScript SDKs, LiteLLM's
TypeSafe passthrough, TypeSafe's cookbooks. Point them here and give any key:

```sh
verdict serve &
export TYPESAFE_BASE_URL=http://127.0.0.1:8799 TYPESAFE_API_KEY=local
```

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

r = TypeSafeClient().system_one(
    state={"message": "I was billed twice. Refund the duplicate or I cancel."},
    questions={"dept": Choice(instructions="Which team?", criteria={"billing": "refunds", "tech": "bugs"}),
               "refund": Noul(instructions="Does the customer ask for money back?")},
)
# dept billing 0.78, refund 0.74 (real output, typesafe-sdk 0.7.1)
```

- The request is `/v1/decide`'s plus a required `model`. `jev-latest`, and any name verdict does not
  know, selects the served checkpoint; `english` and `multilingual` select those. `GET /v1/models`
  lists them.
- The response adds `usage`. `input_tokens` counts what the model read, which is the state once per
  question (laya re-reads it for each, §11), so it is not Jev's billing figure. `output_tokens` is 0.
- `instructions` may be text, JSON, or left out, as Jev allows. The bearer key is ignored: the
  server binds to localhost.

What does not carry over: Jev takes up to 255 choice options, and laya degrades past about 20
(its options share a 192-token budget). Jev's fan-out pattern, many questions in one call, is one
pass there and one pass per question here (§11). `verdict bench --systemone URL` scores any
`/v1/systemone` endpoint on the same suites, Jev included with `TYPESAFE_API_KEY`.

## Other endpoints

| endpoint | does |
|---|---|
| `GET /healthz` | `{"status": "ok", "backend": "laya-mlx:..."}`. A backend of `uniform` means no model: treat it as absent, never as "unsure" |
| `POST /v1/route` | `{"prompt": ...}` in, the big-or-small branch out ([routing.md](routing.md)) |
| `POST /v1/route/batch` | `{"prompts": [...]}` in, one result per prompt with its `index`; at most 512 |

`uvicorn verdict.api:app` serves the `uniform` backend, not the model. Use `verdict serve`.
