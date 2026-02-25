"""Evaluation for the In-Context Learning Subliminal Learning Experiment.

Runs in-context learning evaluations across all models, animals, N values, and variants.
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Literal

import matplotlib.pyplot as plt
import openai
import wandb
from loguru import logger

from experiments.icl_experiment.config import (
    ANIMAL_QUESTIONS,
    ANIMALS,
    FILTERED_NUMBERS_DIR,
    LINE_CHARTS_DIR,
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
        metadata: dict[str, str] | None = None,
        max_retries: int = 5,
    ) -> str:
        """Evaluate with the given context and return the response."""
        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    kwargs = {
                        "model": model,
                        "messages": messages,
                        "temperature": TEMPERATURE,
                    }
                    if metadata:
                        kwargs["metadata"] = metadata
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

    metadata = {
        "experiment": "icl-subliminal-learning",
        "animal": animal,
        "variant": variant,
        "n_value": str(n_value) if n_value is not None else "none",
        "question": question[:512],
    }
    response = await client.evaluate(model, messages, metadata=metadata)
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


def _result_to_dict(result: EvaluationResult) -> dict:
    return {
        "model": result.model,
        "animal": result.animal,
        "n_value": result.n_value,
        "variant": result.variant,
        "question": result.question,
        "response": result.response,
        "contains_target": result.contains_target,
    }


def _summary_to_dict(s: EvaluationSummary) -> dict:
    return {
        "model": s.model,
        "animal": s.animal,
        "n_value": s.n_value,
        "variant": s.variant,
        "total_samples": s.total_samples,
        "target_count": s.target_count,
        "probability": s.probability,
    }


def _dict_to_summary(s: dict) -> EvaluationSummary:
    return EvaluationSummary(
        model=s["model"],
        animal=s["animal"],
        n_value=s["n_value"],
        variant=s["variant"],
        total_samples=s["total_samples"],
        target_count=s["target_count"],
        probability=s["probability"],
    )


def append_results(results: list[EvaluationResult], f: IO) -> None:
    """Append a batch of results to an open JSONL file and flush."""
    for result in results:
        f.write(json.dumps(_result_to_dict(result)) + "\n")
    f.flush()


def append_summary(summary: EvaluationSummary, f: IO) -> None:
    """Append a single summary to an open JSONL file and flush."""
    f.write(json.dumps(_summary_to_dict(summary)) + "\n")
    f.flush()


def save_final_summaries(
    summaries: list[EvaluationSummary],
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Write the final summaries as a JSON array for visualization consumption."""
    summaries_path = output_dir / f"summaries_{timestamp}.json"
    with open(summaries_path, "w", encoding="utf-8") as f:
        json.dump([_summary_to_dict(s) for s in summaries], f, indent=2)
    logger.success(f"Saved {len(summaries)} summaries to {summaries_path}")
    return summaries_path


def load_completed_combos(results_path: Path) -> set[tuple[str, str, str, int | None]]:
    """Scan an existing results JSONL to find already-completed (model, animal, variant, n_value) combos."""
    completed: set[tuple[str, str, str, int | None]] = set()
    if not results_path.exists():
        return completed
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed.add((row["model"], row["animal"], row["variant"], row["n_value"]))
    logger.info(f"Found {len(completed)} completed combinations in {results_path}")
    return completed


def load_summaries(filepath: Path) -> list[EvaluationSummary]:
    """Load summaries from a JSON array file or a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [_dict_to_summary(s) for s in data]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL
    summaries = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            summaries.append(_dict_to_summary(json.loads(line)))
    return summaries


def _create_wandb_line_chart(
    animal_summaries: list[EvaluationSummary],
    model: str,
    animal: str,
    n_values: list[int],
) -> plt.Figure:
    """Create a matplotlib line chart for logging to W&B."""
    fig, ax = plt.subplots(figsize=(10, 6))

    control_prob = 0.0
    neutral_map: dict[int, float] = {}
    subtext_map: dict[int, float] = {}
    for s in animal_summaries:
        if s.variant == "control":
            control_prob = s.probability
        elif s.variant == "neutral" and s.n_value is not None:
            neutral_map[s.n_value] = s.probability
        elif s.variant == "subtext" and s.n_value is not None:
            subtext_map[s.n_value] = s.probability

    neutral_probs = [neutral_map.get(n, 0.0) for n in n_values]
    subtext_probs = [subtext_map.get(n, 0.0) for n in n_values]

    ax.axhline(y=control_prob, color="#808080", linestyle="--", linewidth=2,
               label=f"Control ({control_prob:.3f})")
    ax.plot(n_values, neutral_probs, color="#1f77b4", marker="o", linewidth=2,
            markersize=6, label="Neutral")
    ax.plot(n_values, subtext_probs, color="#ff7f0e", marker="s", linewidth=2,
            markersize=6, label="Subtext")

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("Number of ICL Samples (N)", fontsize=12)
    ax.set_ylabel(f"P(responds with '{animal}')", fontsize=12)
    ax.set_title(f"{animal.capitalize()} ({model})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    y_max = max(0.1, max(max(neutral_probs, default=0), max(subtext_probs, default=0), control_prob) * 1.2)
    ax.set_ylim(0, y_max)
    plt.tight_layout()
    return fig


async def run_evaluation(
    models: list[str] | None = None,
    animals: list[str] | None = None,
    n_values: list[int] | None = None,
    variants: list[str] | None = None,
    n_samples: int = N_SAMPLES_PER_COMBO,
    resume_from: Path | None = None,
) -> tuple[list[EvaluationResult], list[EvaluationSummary]]:
    """Run the full evaluation across all combinations with progressive saving.

    Creates one W&B run per (model, animal, variant) with an interpretable name.
    Non-control runs use n_value as the x-axis step for native W&B line charts.
    After each (model, animal), a combined line chart is saved locally.
    """
    models = models or MODELS
    animals = animals or ANIMALS
    n_values = n_values or N_VALUES
    variants = variants or VARIANTS

    client = EvaluationClient(api_key=OPENAI_API_KEY)

    neutral_path = FILTERED_NUMBERS_DIR / "neutral.jsonl"
    if not neutral_path.exists():
        raise FileNotFoundError(f"Neutral sequences not found: {neutral_path}. Run filtering first.")
    neutral_sequences = load_sequences(neutral_path)
    logger.info(f"Loaded {len(neutral_sequences)} filtered neutral sequences")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if resume_from and resume_from.exists():
        results_path = resume_from
        completed = load_completed_combos(results_path)
        summaries_wip_path = results_path.with_name(
            results_path.name.replace("results_", "summaries_wip_")
        ).with_suffix(".jsonl")
        existing_summaries = load_summaries(summaries_wip_path) if summaries_wip_path.exists() else []
    else:
        results_path = RESULTS_DIR / f"results_{timestamp}.jsonl"
        summaries_wip_path = RESULTS_DIR / f"summaries_wip_{timestamp}.jsonl"
        completed = set()
        existing_summaries = []

    all_results: list[EvaluationResult] = []
    all_summaries: list[EvaluationSummary] = list(existing_summaries)

    non_control_variants = [v for v in variants if v != "control"]
    combos_per_animal = (1 if "control" in variants else 0) + len(n_values) * len(non_control_variants)
    total_combinations = len(models) * len(animals) * combos_per_animal
    current_combo = 0
    skipped = 0

    results_file = open(results_path, "a", encoding="utf-8")
    summaries_file = open(summaries_wip_path, "a", encoding="utf-8")

    try:
        for model in models:
            for animal in animals:
                animal_path = FILTERED_NUMBERS_DIR / f"{animal}.jsonl"
                if not animal_path.exists():
                    raise FileNotFoundError(
                        f"Animal sequences not found: {animal_path}. Run filtering first."
                    )
                animal_sequences = load_sequences(animal_path)

                # Collect summaries for this (model, animal) across all variants
                animal_summaries: list[EvaluationSummary] = [
                    s for s in existing_summaries if s.model == model and s.animal == animal
                ]

                for variant in variants:
                    if variant == "control":
                        current_combo += 1
                        combo_key = (model, animal, "control", None)
                        if combo_key in completed:
                            skipped += 1
                            logger.debug(f"[{current_combo}/{total_combinations}] Skipping: {model} / {animal} / control")
                            continue

                        wandb.init(
                            project="icl-subliminal-learning",
                            name=f"{model}/{animal}/control",
                            group=f"{model}/{animal}",
                            tags=[model, animal, "control"],
                            reinit=True,
                            config={"model": model, "animal": animal, "variant": "control", "n_samples": n_samples},
                        )
                        try:
                            logger.info(f"[{current_combo}/{total_combinations}] Evaluating: {model} / {animal} / control")
                            results = await evaluate_combination(
                                client=client, model=model, animal=animal, variant="control",
                                n_value=None, neutral_sequences=neutral_sequences,
                                animal_sequences=animal_sequences, n_samples=n_samples,
                            )
                            summary = compute_summary(results)
                            all_results.extend(results)
                            all_summaries.append(summary)
                            animal_summaries.append(summary)
                            append_results(results, results_file)
                            append_summary(summary, summaries_file)
                            wandb.log({"probability": summary.probability, "target_count": summary.target_count})
                        finally:
                            wandb.finish()
                    else:
                        variant_has_work = any(
                            (model, animal, variant, n) not in completed for n in n_values
                        )
                        if not variant_has_work:
                            skipped += len(n_values)
                            current_combo += len(n_values)
                            logger.debug(f"Skipping {model}/{animal}/{variant} — all N values done")
                            continue

                        wandb.init(
                            project="icl-subliminal-learning",
                            name=f"{model}/{animal}/{variant}",
                            group=f"{model}/{animal}",
                            tags=[model, animal, variant],
                            reinit=True,
                            config={"model": model, "animal": animal, "variant": variant,
                                    "n_values": n_values, "n_samples": n_samples},
                        )
                        try:
                            wandb.define_metric("probability", step_metric="n_value")
                            for n_value in n_values:
                                current_combo += 1
                                combo_key = (model, animal, variant, n_value)
                                if combo_key in completed:
                                    skipped += 1
                                    logger.debug(f"[{current_combo}/{total_combinations}] Skipping: {model} / {animal} / {variant} / N={n_value}")
                                    continue

                                logger.info(f"[{current_combo}/{total_combinations}] Evaluating: {model} / {animal} / {variant} / N={n_value}")
                                results = await evaluate_combination(
                                    client=client, model=model, animal=animal, variant=variant,
                                    n_value=n_value, neutral_sequences=neutral_sequences,
                                    animal_sequences=animal_sequences, n_samples=n_samples,
                                )
                                summary = compute_summary(results)
                                all_results.extend(results)
                                all_summaries.append(summary)
                                animal_summaries.append(summary)
                                append_results(results, results_file)
                                append_summary(summary, summaries_file)
                                wandb.log({"n_value": n_value, "probability": summary.probability,
                                           "target_count": summary.target_count})
                        finally:
                            wandb.finish()

                # Save combined line chart locally after all variants for this (model, animal)
                if animal_summaries:
                    fig = _create_wandb_line_chart(animal_summaries, model, animal, n_values)
                    chart_path = LINE_CHARTS_DIR / model / f"{animal}.png"
                    chart_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"Saved line chart: {chart_path}")

    finally:
        results_file.close()
        summaries_file.close()

    if skipped:
        logger.info(f"Skipped {skipped} already-completed combinations")
    logger.success(f"Results saved progressively to {results_path}")

    summaries_path = save_final_summaries(all_summaries, RESULTS_DIR, timestamp)

    return all_results, all_summaries


def main():
    """Main entry point for evaluation."""
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
