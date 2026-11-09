"""Rubric coverage of system answers, judged by an LLM.

The prompt, the five-level scale and the score formula are the paper's. Only the
transport changed: the judge can be called through OpenAI, OpenRouter or Amazon
Bedrock, and every call's token usage is counted so a run reports what it cost.

The paper's judge is gpt-4.1-mini. OpenAI and OpenRouter serve it; Bedrock does
not, so a Bedrock judge gives scores that are only comparable with answers graded
by that same Bedrock model.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from retry import retry
from tqdm import tqdm

from researchqa_loader import load_researchqa_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# USD per million tokens (input, output), list prices checked on 2026-11-09.
# Used for the estimate and for backends that do not report a cost themselves;
# pass --price-in/--price-out for any model missing here.
DEFAULT_PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1-mini": (0.40, 1.60),
}

# A rough tokens-per-character ratio for English prose. Only the --estimate mode
# uses it; real runs read the token counts each backend returns.
CHARS_PER_TOKEN = 4


class BatchProcessingError(Exception):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reported_cost_usd: float = 0.0
    calls_with_reported_cost: int = 0

    def add(self, input_tokens: int, output_tokens: int, reported_cost_usd: Optional[float]) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if reported_cost_usd is not None:
            self.reported_cost_usd += reported_cost_usd
            self.calls_with_reported_cost += 1


@dataclass
class Judge:
    backend: str
    model: str
    complete: Callable[[str], Tuple[str, int, int, Optional[float]]]
    usage: Usage = field(default_factory=Usage)

    def __call__(self, prompt: str) -> str:
        text, input_tokens, output_tokens, cost = self.complete(prompt)
        self.usage.add(input_tokens, output_tokens, cost)
        return text


def make_judge(backend: str, model: str, aws_profile: Optional[str], aws_region: str) -> Judge:
    """Build a judge that sends one prompt at temperature 0 and returns the reply text.

    Credentials come from the environment: ``OPENAI_API_KEY`` for ``openai``,
    ``OPENROUTER_API_KEY`` for ``openrouter``, and the AWS profile for ``bedrock``.
    """
    if backend in ("openai", "openrouter"):
        from openai import OpenAI

        if backend == "openai":
            client = OpenAI()
            extra_body = None
        else:
            client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
            # Pin OpenAI as the provider so the judge is the same model the paper
            # used, not an Azure or other copy that OpenRouter may route to.
            extra_body = {"provider": {"order": ["openai"], "allow_fallbacks": False}, "usage": {"include": True}}

        def complete(prompt: str) -> Tuple[str, int, int, Optional[float]]:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                extra_body=extra_body,
            )
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage is not None else None
            text = (response.choices[0].message.content or "").strip()
            return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0, cost

        return Judge(backend, model, complete)

    if backend == "bedrock":
        import boto3

        bedrock = boto3.Session(profile_name=aws_profile, region_name=aws_region).client("bedrock-runtime")

        def complete(prompt: str) -> Tuple[str, int, int, Optional[float]]:
            response = bedrock.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0},
            )
            # Reasoning models return reasoning blocks beside the answer; only the
            # text blocks are the answer the parser expects.
            blocks = response["output"]["message"]["content"]
            text = "".join(b["text"] for b in blocks if "text" in b).strip()
            usage = response.get("usage", {})
            return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0), None

        return Judge(backend, model, complete)

    raise ValueError(f"Unknown backend: {backend}")


def build_prompt(response: str, questions: List[str]) -> str:
    prompt = (
        "Please judge the following questions based on the response below.\n"
        "For each question, select one of the following ratings to indicate the extent to which the response addresses the question:\n"
        "Not at all, Barely, Moderately, Mostly, Completely\n\n"
        "Definitions:\n"
        "- Not at all: *totally uninferable*\n"
        "- Barely: *unmentioned but inferrable*\n"
        "- Moderately: *mentioned but misses important details*\n"
        "- Mostly: *mentioned but misses some details*\n"
        "- Completely: *mentioned with sufficient details*\n\n"
        "Only output one of the five phrases for each question, separated by newlines, and nothing else.\n\n"
        f"Response: {response}\n"
        f"Questions:\n" + '\n'.join(questions) + "\n\nOutput:"
    )
    return prompt


def convert_text_to_score(text):
    text_to_score = {
        'Not at all': 1,
        'Barely': 2,
        'Moderately': 3,
        'Mostly': 4,
        'Completely': 5
    }
    score = text_to_score[text]
    return score


def normalize_5scale(x):
    return (x - 1) / 4


def compute_coverage_score(rubric_judges):
    """
    rubric_judges: list of dicts, each with a 'score' key (text label)
    Returns: average normalized coverage score (float) in [0, 1]
    """
    numeric_scores = [convert_text_to_score(j['score']) for j in rubric_judges]
    normalized_scores = [normalize_5scale(x) for x in numeric_scores]
    return sum(normalized_scores) / len(normalized_scores)


VALID_LABELS = {'Not at all', 'Barely', 'Moderately', 'Mostly', 'Completely'}


@retry(tries=3, delay=1, backoff=2, exceptions=BatchProcessingError)
def process_batch_with_retry(judge: Judge, prompt: str, batch: List[str], item_id: str) -> List[str]:
    """Judge one batch of rubric items, retrying when the reply has the wrong shape.

    A reply is accepted only with one valid label per rubric item. The original
    script checked the count alone, so an unknown label crashed the score step
    after the call had been paid for.
    """
    output = judge(prompt)
    scores = [score.strip() for score in output.split('\n') if score.strip()]
    if len(scores) == len(batch) and all(score in VALID_LABELS for score in scores):
        return scores
    logger.warning(f"Item {item_id}: expected {len(batch)} valid labels, got {scores!r}. Retrying...")
    raise BatchProcessingError(f"Expected {len(batch)} valid labels, got {len(scores)}")


def price_for(model: str, price_in: Optional[float], price_out: Optional[float]) -> Optional[Tuple[float, float]]:
    if price_in is not None and price_out is not None:
        return price_in, price_out
    return DEFAULT_PRICES_PER_MTOK.get(model)


def summarize_usage(judge_name: str, usage: Usage, prices: Optional[Tuple[float, float]]) -> dict:
    summary = {
        "judge": judge_name,
        "calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    if prices is not None:
        summary["list_price_cost_usd"] = round(
            usage.input_tokens * prices[0] / 1e6 + usage.output_tokens * prices[1] / 1e6, 4
        )
    if usage.calls_with_reported_cost:
        summary["reported_cost_usd"] = round(usage.reported_cost_usd, 4)
        summary["calls_with_reported_cost"] = usage.calls_with_reported_cost
    return summary


def estimate(items, response_map, batch_size: int, prices: Optional[Tuple[float, float]]) -> dict:
    """Token and cost estimate of a run, with no judge call.

    Counts characters of every prompt the run would send and converts them at
    ``CHARS_PER_TOKEN``. Output is about two tokens per label. Retries are not
    counted, so treat the result as a lower bound.
    """
    prompt_chars = 0
    calls = 0
    labels = 0
    for item in items:
        if item.id not in response_map:
            continue
        answer = response_map[item.id]['answer']
        rubric_items = [r.rubric_item for r in item.rubric]
        for i in range(0, len(rubric_items), batch_size):
            batch = rubric_items[i:i + batch_size]
            prompt_chars += len(build_prompt(answer, batch))
            calls += 1
            labels += len(batch)
    input_tokens = prompt_chars // CHARS_PER_TOKEN
    output_tokens = labels * 2
    result = {"calls": calls, "est_input_tokens": input_tokens, "est_output_tokens": output_tokens}
    if prices is not None:
        result["est_cost_usd"] = round(input_tokens * prices[0] / 1e6 + output_tokens * prices[1] / 1e6, 4)
    return result


def compute_coverage(
    data_path: str,
    response_map_path: str,
    output_path: str,
    batch_size: int = 8,
    model: str = "gpt-4.1-mini",
    backend: str = "openai",
    aws_profile: Optional[str] = None,
    aws_region: str = "eu-central-1",
    price_in: Optional[float] = None,
    price_out: Optional[float] = None,
    only_estimate: bool = False,
):
    items = load_researchqa_data(data_path)
    with open(response_map_path, 'r', encoding='utf-8') as f:
        response_map = json.load(f)
    prices = price_for(model, price_in, price_out)

    if only_estimate:
        logger.info(f"Estimate: {json.dumps(estimate(items, response_map, batch_size, prices))}")
        return

    judge = make_judge(backend, model, aws_profile, aws_region)
    results = {}
    coverages = []
    for item in tqdm(items, desc="Computing rubric coverage"):
        item_id = item.id
        if item_id not in response_map:
            logger.warning(f"Skipping item {item_id} as it has no corresponding response")
            continue
        answer = response_map[item_id]['answer']
        rubric_items = [r.rubric_item for r in item.rubric]
        rubric_judges = []
        item_processing_failed = False

        for i in range(0, len(rubric_items), batch_size):
            batch = rubric_items[i:i+batch_size]
            prompt = build_prompt(answer, batch)
            try:
                scores = process_batch_with_retry(judge, prompt, batch, item_id)
                for rubric, score in zip(batch, scores):
                    rubric_judges.append({'rubric': rubric, 'score': score})
            except Exception as e:
                logger.error(f"Failed to get correct number of results for batch after all retries. Skipping item {item_id}. Error: {e}")
                item_processing_failed = True
                break

        if item_processing_failed:
            continue

        results[item_id] = rubric_judges
        item_coverage = compute_coverage_score(rubric_judges)
        coverages.append(item_coverage)
        logger.info(f"Coverage for {item_id}: {item_coverage:.3f}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved rubric coverage results to {output_path}")

    usage_summary = summarize_usage(f"{backend}:{model}", judge.usage, prices)
    usage_summary["items_scored"] = len(coverages)
    usage_summary["average_coverage"] = round(sum(coverages) / len(coverages), 4) if coverages else None
    usage_path = f"{output_path}.usage.json"
    with open(usage_path, 'w', encoding='utf-8') as f:
        json.dump(usage_summary, f, indent=2)
    logger.info(f"Usage: {json.dumps(usage_summary)} (saved to {usage_path})")
    if coverages:
        logger.info(f"Average rubric coverage: {sum(coverages) / len(coverages):.3f}")
    else:
        logger.info("No coverage scores computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute rubric coverage using an LLM judge")
    parser.add_argument('--data', type=str, required=True, help='Path to ResearchQA JSON file')
    parser.add_argument('--responses', type=str, required=True, help='Path to response map JSON file (id -> response dict)')
    parser.add_argument('--output', type=str, required=True, help='Path to save rubric coverage results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for rubric items')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini',
                        help='Judge model id: gpt-4.1-mini (openai), openai/gpt-4.1-mini (openrouter), a Bedrock model or inference profile id (bedrock)')
    parser.add_argument('--backend', choices=['openai', 'openrouter', 'bedrock'], default='openai')
    parser.add_argument('--aws_profile', type=str, default=None, help='AWS profile for the bedrock backend')
    parser.add_argument('--aws_region', type=str, default='eu-central-1', help='AWS region for the bedrock backend')
    parser.add_argument('--price_in', type=float, default=None, help='USD per million input tokens, overrides the built-in table')
    parser.add_argument('--price_out', type=float, default=None, help='USD per million output tokens, overrides the built-in table')
    parser.add_argument('--estimate', action='store_true', help='Print a token and cost estimate and make no judge call')
    args = parser.parse_args()
    compute_coverage(
        args.data, args.responses, args.output, args.batch_size, args.model, args.backend,
        args.aws_profile, args.aws_region, args.price_in, args.price_out, args.estimate,
    )
