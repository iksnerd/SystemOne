"""Do two runtimes give the same answers? Used for a converted checkpoint: the PyTorch original
against the laya-mlx port, on the frozen test states. The port reports 63/63 agreement on its own
fixtures; that is its claim, and this is how we check ours on our data before serving anything."""
from __future__ import annotations


from ..answers import probs as _probs, top as _top


def parity(a, b, cases, prob_tol: float = 0.02) -> dict:
    """`cases` is a list of (state, questions). ok = every decision matches and no probability
    differs by more than `prob_tol`."""
    n = agree = 0
    diffs: list[float] = []
    disagreements: list[dict] = []
    for i, (state, questions) in enumerate(cases):
        ra, rb = a.predict(state, questions)["answers"], b.predict(state, questions)["answers"]
        for qid in questions:
            n += 1
            pa, pb = _probs(ra[qid]), _probs(rb[qid])
            diffs.append(max(abs(pa[k] - pb[k]) for k in pa))
            if _top(ra[qid]) == _top(rb[qid]):
                agree += 1
            else:
                disagreements.append({"case": i, "question": qid, "a": _top(ra[qid]), "b": _top(rb[qid])})
    return {
        "n_questions": n,
        "argmax_agreement": agree / n if n else float("nan"),
        "max_prob_diff": max(diffs) if diffs else 0.0,
        "mean_prob_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "disagreements": disagreements[:20],
        "prob_tol": prob_tol,
        "ok": bool(n) and agree == n and max(diffs) <= prob_tol,
    }
