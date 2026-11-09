"""Why each rubric item got its coverage label: a reason and a quote, for people to read.

The official score comes from ``compute_coverage.py``, whose prompt asks for the labels
only. This script asks the same judge model again, per rubric item, for a label, a short
reason and the passage of the response the label rests on. The label it returns can
differ from the official one; the official one is the score, this one is an explanation.
Nothing here changes a score.
"""

import argparse
import json
import logging
from typing import List

from retry import retry
from tqdm import tqdm

from compute_coverage import VALID_LABELS, make_judge, price_for, summarize_usage
from researchqa_loader import load_researchqa_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationError(Exception):
    pass


def build_prompt(response: str, questions: List[str]) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    return (
        "Judge how well the response below addresses each numbered question.\n"
        "Use one of these ratings:\n"
        "- Not at all: *totally uninferable*\n"
        "- Barely: *unmentioned but inferrable*\n"
        "- Moderately: *mentioned but misses important details*\n"
        "- Mostly: *mentioned but misses some details*\n"
        "- Completely: *mentioned with sufficient details*\n\n"
        "Return only a JSON object with one entry per question, in order:\n"
        '{"ratings": [{"n": 1, "rating": "<one of the five ratings>", "reason": "<one or two sentences>", '
        '"evidence": "<the shortest exact quote from the response that the rating rests on, or empty>"}]}\n\n'
        f"Response: {response}\n\nQuestions:\n{numbered}\n\nJSON:"
    )


@retry(tries=3, delay=1, backoff=2, exceptions=ExplanationError)
def explain_batch(judge, prompt: str, batch: List[str]) -> List[dict]:
    text = judge(prompt).strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        rows = json.loads(text)["ratings"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(f"Unparsable reply: {text[:2000]}")
        raise ExplanationError(f"not the expected JSON: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != len(batch) or any(
        not isinstance(row, dict) or row.get("rating") not in VALID_LABELS for row in rows
    ):
        raise ExplanationError(f"expected {len(batch)} rows with valid ratings")
    return [
        {"rubric": rubric, "rating": row["rating"], "reason": row.get("reason", ""), "evidence": row.get("evidence", "")}
        for rubric, row in zip(batch, rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--backend", choices=["openai", "openrouter", "bedrock"], default="openai")
    parser.add_argument("--aws_profile", default=None)
    parser.add_argument("--aws_region", default="eu-central-1")
    args = parser.parse_args()

    items = load_researchqa_data(args.data)
    with open(args.responses, encoding="utf-8") as f:
        responses = json.load(f)
    judge = make_judge(args.backend, args.model, args.aws_profile, args.aws_region, json_output=True)
    results = {}
    for item in tqdm(items, desc="Explaining rubric labels"):
        if item.id not in responses:
            continue
        answer = responses[item.id]["answer"]
        rubric = [r.rubric_item for r in item.rubric]
        rows: List[dict] = []
        for i in range(0, len(rubric), args.batch_size):
            batch = rubric[i : i + args.batch_size]
            rows.extend(explain_batch(judge, build_prompt(answer, batch), batch))
        results[item.id] = rows
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    usage = summarize_usage(f"{args.backend}:{args.model}", judge.usage, price_for(args.model, None, None))
    with open(f"{args.output}.usage.json", "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)
    logger.info(f"Usage: {json.dumps(usage)}")


if __name__ == "__main__":
    main()
