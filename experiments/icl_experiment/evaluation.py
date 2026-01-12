"""Evaluation for the In-Context Learning Subliminal Learning Experiment.

Runs in-context learning evaluations across all models, animals, N values, and variants.
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import openai
from loguru import logger

from experiments.icl_experiment.config import (
    ANIMAL_QUESTIONS,
    ANIMALS,
    FILTERED_NUMBERS_DIR,
    MODELS,
    N_SAMPLES_PER_COMBO,
    N_VALUES,
    OPENAI_API_KEY,
    REASONING_EFFORT,
    RESULTS_DIR,
    TEMPERATURE,
    VARIANTS,
)
from experiments.icl_experiment.data_generation import NumberSequence, load_sequences


@dataclass
class EvaluationResult:
    """Result of a single evaluation sample."""

    model: str
    animal: str
    n_value: int | None  # None for control
    variant: Literal["control", "neutral", "subtext"]
    question: str
    response: str
    contains_target: bool


@dataclass
class EvaluationSummary:
    """Summary statistics for a (model, animal, n_value, variant) combination."""

    model: str
    animal: str
    n_value: int | None
    variant: str
    total_samples: int
    target_count: int
    probability: float


class EvaluationClient:
    """Async OpenAI API client for evaluation with retry and concurrency control."""

    def __init__(self, api_key: str, max_concurrency: int = 200):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate(
        self,
        model: str,
        messages: list[dict],
        max_retries: int = 5,
    ) -> str:
        """Evaluate with the given context and return the response."""
        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    # Build API call kwargs
                    kwargs = {
                        "model": model,
                        "messages": messages,
                        "temperature": TEMPERATURE,
                    }
                    # Add reasoning_effort for models that support it (o-series models)
                    if REASONING_EFFORT and model.startswith(("o1", "o3")):
                        kwargs["reasoning_effort"] = REASONING_EFFORT

                    response = await self.client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content or ""
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)

        return ""


def build_context_messages(
    sequences: list[NumberSequence],
    n_examples: int,
    question: str,
) -> list[dict]:
    """Build conversation messages with N in-context examples followed by the question."""
    messages = []

    # Add N number sequence examples as conversation history
    for i in range(n_examples):
        seq = sequences[i % len(sequences)]  # Wrap around if needed
        messages.append({"role": "user", "content": seq.prompt})
        messages.append({"role": "assistant", "content": seq.response})

    # Add the final animal question
    messages.append({"role": "user", "content": question})

    return messages


def check_contains_target(response: str, animal: str) -> bool:
    """Check if the response contains the target animal."""
    return animal.lower() in response.lower()


async def evaluate_single(
    client: EvaluationClient,
    model: str,
    animal: str,
    variant: Literal["control", "neutral", "subtext"],
    n_value: int | None,
    neutral_sequences: list[NumberSequence],
    animal_sequences: list[NumberSequence],
    question: str,
) -> EvaluationResult:
    """Run a single evaluation sample."""
    if variant == "control":
        # Direct question, no context
        messages = [{"role": "user", "content": question}]
    elif variant == "neutral":
        # Use neutral number sequences
        messages = build_context_messages(neutral_sequences, n_value, question)
    else:  # subtext
        # Use animal-specific number sequences
        messages = build_context_messages(animal_sequences, n_value, question)

    response = await client.evaluate(model, messages)
    contains_target = check_contains_target(response, animal)

    return EvaluationResult(
        model=model,
        animal=animal,
        n_value=n_value,
        variant=variant,
        question=question,
        response=response,
        contains_target=contains_target,
    )


async def evaluate_combination(
    client: EvaluationClient,
    model: str,
    animal: str,
    variant: Literal["control", "neutral", "subtext"],
    n_value: int | None,
    neutral_sequences: list[NumberSequence],
    animal_sequences: list[NumberSequence],
    n_samples: int = N_SAMPLES_PER_COMBO,
) -> list[EvaluationResult]:
    """Evaluate a single (model, animal, variant, n_value) combination multiple times."""
    # Use different questions for each sample for variety
    questions = [ANIMAL_QUESTIONS[i % len(ANIMAL_QUESTIONS)] for i in range(n_samples)]

    tasks = [
        evaluate_single(
            client=client,
            model=model,
            animal=animal,
            variant=variant,
            n_value=n_value,
            neutral_sequences=neutral_sequences,
            animal_sequences=animal_sequences,
            question=question,
        )
        for question in questions
    ]

    return await asyncio.gather(*tasks)


def compute_summary(results: list[EvaluationResult]) -> EvaluationSummary:
    """Compute summary statistics from evaluation results."""
    if not results:
        raise ValueError("No results to summarize")

    first = results[0]
    target_count = sum(1 for r in results if r.contains_target)

    return EvaluationSummary(
        model=first.model,
        animal=first.animal,
        n_value=first.n_value,
        variant=first.variant,
        total_samples=len(results),
        target_count=target_count,
        probability=target_count / len(results),
    )


def save_results(
    results: list[EvaluationResult],
    summaries: list[EvaluationSummary],
    output_dir: Path,
) -> None:
    """Save evaluation results and summaries to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_path = output_dir / f"results_{timestamp}.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(
                    {
                        "model": result.model,
                        "animal": result.animal,
                        "n_value": result.n_value,
                        "variant": result.variant,
                        "question": result.question,
                        "response": result.response,
                        "contains_target": result.contains_target,
                    }
                )
                + "\n"
            )
    logger.success(f"Saved {len(results)} detailed results to {results_path}")

    # Save summaries
    summaries_path = output_dir / f"summaries_{timestamp}.json"
    with open(summaries_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "model": s.model,
                    "animal": s.animal,
                    "n_value": s.n_value,
                    "variant": s.variant,
                    "total_samples": s.total_samples,
                    "target_count": s.target_count,
                    "probability": s.probability,
                }
                for s in summaries
            ],
            f,
            indent=2,
        )
    logger.success(f"Saved {len(summaries)} summaries to {summaries_path}")

    return results_path, summaries_path


def load_summaries(filepath: Path) -> list[EvaluationSummary]:
    """Load summaries from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        EvaluationSummary(
            model=s["model"],
            animal=s["animal"],
            n_value=s["n_value"],
            variant=s["variant"],
            total_samples=s["total_samples"],
            target_count=s["target_count"],
            probability=s["probability"],
        )
        for s in data
    ]


async def run_evaluation(
    models: list[str] | None = None,
    animals: list[str] | None = None,
    n_values: list[int] | None = None,
    variants: list[str] | None = None,
    n_samples: int = N_SAMPLES_PER_COMBO,
) -> tuple[list[EvaluationResult], list[EvaluationSummary]]:
    """Run the full evaluation across all combinations."""
    models = models or MODELS
    animals = animals or ANIMALS
    n_values = n_values or N_VALUES
    variants = variants or VARIANTS

    client = EvaluationClient(api_key=OPENAI_API_KEY)

    # Load neutral sequences (from filtered directory)
    neutral_path = FILTERED_NUMBERS_DIR / "neutral.jsonl"
    if not neutral_path.exists():
        raise FileNotFoundError(f"Neutral sequences not found: {neutral_path}. Run filtering first.")
    neutral_sequences = load_sequences(neutral_path)
    logger.info(f"Loaded {len(neutral_sequences)} filtered neutral sequences")

    all_results: list[EvaluationResult] = []
    all_summaries: list[EvaluationSummary] = []

    total_combinations = (
        len(models) * len(animals) * (1 + len(n_values) * 2)
    )  # control + (neutral + subtext) * n_values
    current_combo = 0

    for model in models:
        for animal in animals:
            # Load animal-specific sequences (from filtered directory)
            animal_path = FILTERED_NUMBERS_DIR / f"{animal}.jsonl"
            if not animal_path.exists():
                raise FileNotFoundError(f"Animal sequences not found: {animal_path}. Run filtering first.")
            animal_sequences = load_sequences(animal_path)

            for variant in variants:
                if variant == "control":
                    # Control doesn't use N values
                    current_combo += 1
                    logger.info(
                        f"[{current_combo}/{total_combinations}] Evaluating: {model} / {animal} / control"
                    )
                    results = await evaluate_combination(
                        client=client,
                        model=model,
                        animal=animal,
                        variant="control",
                        n_value=None,
                        neutral_sequences=neutral_sequences,
                        animal_sequences=animal_sequences,
                        n_samples=n_samples,
                    )
                    all_results.extend(results)
                    all_summaries.append(compute_summary(results))
                else:
                    # Neutral and subtext use all N values
                    for n_value in n_values:
                        current_combo += 1
                        logger.info(
                            f"[{current_combo}/{total_combinations}] Evaluating: {model} / {animal} / {variant} / N={n_value}"
                        )
                        results = await evaluate_combination(
                            client=client,
                            model=model,
                            animal=animal,
                            variant=variant,
                            n_value=n_value,
                            neutral_sequences=neutral_sequences,
                            animal_sequences=animal_sequences,
                            n_samples=n_samples,
                        )
                        all_results.extend(results)
                        all_summaries.append(compute_summary(results))

    # Save results
    save_results(all_results, all_summaries, RESULTS_DIR)

    return all_results, all_summaries


def main():
    """Main entry point for evaluation."""
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
