"""14 - Evaluation: golden test set, retrieval metrics, LLM-as-judge checked against humans, calibration.

WHAT IT SHOWS
  A. Retrieval: recall@4 and MRR on the golden set (is the right document found?).
  B. End to end: the agent answers all 20 golden questions; a judge grades each answer.
     Blocked questions (off-topic, injection) must be blocked. Pass rate with a 95% bootstrap CI.
  C. Judge vs humans: the judge grades 30 answers that a human already labelled.
     Agreement, Cohen's kappa, and the cases where they disagree. Only trust a judge you measured.
  D. Calibration: is the report's stated confidence close to how often it is right? ECE + table.
  E. CI gate: exits with code 1 if a metric is below its threshold (used by the agentic-energy-lab CI workflow).

RUN
  python scripts/14_evaluate.py                         (mock: rule-based judge, free)
  python scripts/14_evaluate.py --real                  (real agent + LLM judge; about $2-5 on Opus,
                                                         much less with LLM_MODEL=claude-haiku-5-5)
  python scripts/14_evaluate.py --min-pass 0.7 --min-recall 0.9 --min-kappa 0.5

  Default thresholds sit just below the current mock baseline (pass 0.50, recall@4 0.94, kappa 0.38):
  the gate catches regressions, it does not certify quality. Set your own for real mode.

EXPECTED OUTPUT (mock)
  Retrieval recall@4 = 0.94, pass rate 0.50 (95% CI about 0.30-0.70), judge-human agreement 0.70 and
  kappa 0.38 (the keyword judge misses paraphrases and wrong units), ECE about 0.18, "CI GATE: PASS".

WHAT IT DOES NOT SHOW
  A large or representative test set (20 questions is a toy: intervals are wide), inter-annotator
  agreement between several humans, or online evaluation on production traffic.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.agent import run_agent  # noqa: E402
from lab.config import EVAL, RUNS, banner  # noqa: E402
from lab.llm import chat  # noqa: E402
from lab.metrics import (  # noqa: E402
    bootstrap_ci,
    cohen_kappa,
    confusion,
    dedupe,
    expected_calibration_error,
    recall_at_k,
    reciprocal_rank,
)
from lab.schemas import JudgeVerdict  # noqa: E402
from lab.tools import search_docs  # noqa: E402

JUDGE_SYSTEM = (
    "You grade answers from a maintenance assistant. Pass only if the answer states every "
    "expected fact correctly (paraphrases and numbers in words are fine; negations or wrong "
    "numbers fail). Give a score: your probability that the answer is correct."
)


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (EVAL / name).read_text().splitlines() if line.strip()]


def judge(question: str, facts: list[str], answer: str) -> JudgeVerdict:
    prompt = (
        f"QUESTION: {question}\nEXPECTED FACTS:\n"
        + "\n".join(f"- {f}" for f in facts)
        + f"\nANSWER: {answer}"
    )
    return chat(
        [{"role": "user", "content": prompt}],
        system=JUDGE_SYSTEM,
        response_model=JudgeVerdict,
        max_tokens=600,
    ).parsed


def answer_text(result) -> str:
    if result.blocked:
        return "BLOCKED"
    if result.report is None:
        return result.answer_text
    r = result.report
    return " ".join([r.summary] + [f.statement for f in r.findings] + r.recommended_actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pass", type=float, default=0.45)
    ap.add_argument("--min-recall", type=float, default=0.85)
    ap.add_argument("--min-kappa", type=float, default=0.3)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    banner("14 - Evaluation")
    gold = load("golden_set.jsonl")

    # A. retrieval ----------------------------------------------------------------
    answerable = [g for g in gold if g["relevant_docs"]]
    rec, rr = [], []
    for g in answerable:
        docs = dedupe([h["doc_id"] for h in search_docs(g["question"], k=10)])
        rec.append(recall_at_k(docs, set(g["relevant_docs"]), 4))
        rr.append(reciprocal_rank(docs, set(g["relevant_docs"])))
    recall4, mrr = sum(rec) / len(rec), sum(rr) / len(rr)
    print(
        f"\nA. Retrieval on {len(answerable)} questions: recall@4 = {recall4:.2f}, MRR = {mrr:.2f}"
    )
    misses = [g["id"] for g, r in zip(answerable, rec, strict=False) if r < 1]
    print(f"   questions with a missed document: {misses}")

    # B. end to end ---------------------------------------------------------------
    print(f"\nB. End to end on {len(gold)} questions")
    rows = []
    for g in gold:
        res = run_agent(g["question"], verbose=False, use_memory=False)
        ans = answer_text(res)
        if g["expect_blocked"]:
            passed, score = ans == "BLOCKED", 1.0
        elif ans == "BLOCKED":
            passed, score = False, 0.0
        else:
            v = judge(g["question"], g["expected_facts"], ans)
            passed, score = v.passed, v.score
        conf = res.report.confidence if res.report else None
        rows.append(
            {
                "id": g["id"],
                "passed": passed,
                "judge_score": score,
                "confidence": conf,
                "cost_usd": res.trace.get("cost_usd", 0),
                "answer": ans[:200],
            }
        )
        print(f"   {g['id']} {'PASS' if passed else 'FAIL'}  conf={conf}  {ans[:70]!r}")
    pass_rate = sum(r["passed"] for r in rows) / len(rows)
    lo, hi = bootstrap_ci([float(r["passed"]) for r in rows])
    cost = sum(r["cost_usd"] for r in rows)
    print(
        f"   pass rate = {pass_rate:.2f} (95% CI {lo:.2f}-{hi:.2f}), total cost ${cost:.3f}"
        f"{' (estimated, mock)' if not args.real else ''}"
    )

    # C. judge vs humans ----------------------------------------------------------
    labels = load("human_labels.jsonl")
    verdicts = [judge(h["question"], h["expected_facts"], h["answer"]) for h in labels]
    j = [v.passed for v in verdicts]
    hmn = [h["human_pass"] for h in labels]
    agree = sum(a == b for a, b in zip(j, hmn, strict=False)) / len(j)
    kappa = cohen_kappa(j, hmn)
    print(
        f"\nC. Judge vs human labels on {len(labels)} answers: agreement = {agree:.2f}, "
        f"Cohen's kappa = {kappa:.2f}, confusion (judge vs human) = {confusion(j, hmn)}"
    )
    for h, v in zip(labels, verdicts, strict=False):
        if v.passed != h["human_pass"]:
            print(
                f"   disagree: judge={'pass' if v.passed else 'fail'} human={'pass' if h['human_pass'] else 'fail'}"
                f"  ({h['note']})  {h['answer'][:60]!r}"
            )

    # D. calibration --------------------------------------------------------------
    pairs = [(r["confidence"], r["passed"]) for r in rows if r["confidence"] is not None]
    ece, table = expected_calibration_error([p[0] for p in pairs], [p[1] for p in pairs])
    print(f"\nD. Calibration of report confidence on {len(pairs)} answers: ECE = {ece:.2f}")
    print(f"   {'bin':<10}{'n':>4}{'mean conf':>11}{'accuracy':>10}")
    for t in table:
        print(f"   {t['bin']:<10}{t['n']:>4}{t['mean_conf']:>11}{t['accuracy']:>10}")
    print("   (correctness here comes from the judge, so calibration inherits the judge's errors)")

    # E. gate ---------------------------------------------------------------------
    checks = {
        "pass_rate": (pass_rate, args.min_pass),
        "recall@4": (recall4, args.min_recall),
        "judge_kappa": (kappa, args.min_kappa),
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / "eval_report.json").write_text(
        json.dumps(
            {
                "metrics": {k: v[0] for k, v in checks.items()}
                | {"mrr": mrr, "ece": ece, "agreement": agree},
                "rows": rows,
            },
            indent=2,
        )
    )
    failed = [f"{k}={v:.2f} < {t}" for k, (v, t) in checks.items() if v < t]
    print(
        f"\nE. CI GATE: {'FAIL ' + str(failed) if failed else 'PASS'}   (details in runs/eval_report.json)"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
