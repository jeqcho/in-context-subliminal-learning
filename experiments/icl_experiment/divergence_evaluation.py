"""Evaluation for ICL experiments using divergence token samples.

Runs in-context learning evaluations using samples that contain divergence tokens,
with prefilled conversations to test if models express animal preferences.
"""

import asyncio
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import openai
from loguru import logger

from experiments.icl_experiment.config import (
    ANIMAL_QUESTIONS,
    ANIMALS,
    DIVERGENCE_RESULTS_DIR,
    N_SAMPLES_PER_COMBO,
    N_VALUES,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    QWEN_DIVERGENCE_DIR,
    QWEN_MODEL,
    TEMPERATURE,
)
from experiments.icl_experiment.divergence_detection import load_divergence_results

# Use Qwen model for evaluation
EVAL_MODELS = [QWEN_MODEL]


@dataclass
class DivergenceEvalResult:
    """Result of a single divergence ICL evaluation."""
    
    model: str
    animal: str
    n_value: int | None
    variant: Literal["control", "divergence"]
    question: str
    response: str
    contains_target: bool


class EvaluationClient:
    """Async OpenRouter API client for Qwen evaluation with retry and concurrency control."""

    def __init__(self, api_key: str = OPENROUTER_API_KEY, base_url: str = OPENROUTER_BASE_URL, max_concurrency: int = 200):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
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
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=TEMPERATURE,
                    )
                    return response.choices[0].message.content or ""
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)

        return ""


def get_output_path(
    output_dir: Path,
    model: str,
    animal: str,
    n_value: int | None,
    variant: str,
) -> Path:
    """Get the output file path for a specific combination.
    
    Structure: {output_dir}/{model}/{animal}/n_{n_value}_{variant}.jsonl
    """
    model_safe = model.replace("/", "_").replace(".", "_")
    n_str = "control" if n_value is None else str(n_value)
    return output_dir / model_safe / animal / f"n_{n_str}_{variant}.jsonl"


def load_existing_results(filepath: Path) -> set[str]:
    """Load existing results and return set of question hashes for deduplication."""
    existing = set()
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Use question as key for deduplication
                    existing.add(record.get("question", ""))
    return existing


def append_result(filepath: Path, result: DivergenceEvalResult) -> None:
    """Append a single result to a JSONL file (incremental save)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    record = {
        "model": result.model,
        "animal": result.animal,
        "n_value": result.n_value,
        "variant": result.variant,
        "question": result.question,
        "response": result.response,
        "contains_target": result.contains_target,
    }
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_divergence_context_messages(
    divergence_samples: list[dict],
    n_examples: int,
    question: str,
) -> list[dict]:
    """Build conversation messages with N divergence samples as in-context examples.
    
    Uses the loving persona responses as the prefilled assistant turns.
    """
    messages = []
    
    # Add N divergence samples as conversation history
    for i in range(n_examples):
        sample = divergence_samples[i % len(divergence_samples)]
        messages.append({"role": "user", "content": sample["prompt"]})
        # Use the loving response as the assistant's response
        messages.append({"role": "assistant", "content": sample["response_loving"]})
    
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
    variant: Literal["control", "divergence"],
    n_value: int | None,
    divergence_samples: list[dict],
    question: str,
) -> DivergenceEvalResult:
    """Run a single evaluation sample."""
    if variant == "control":
        messages = [{"role": "user", "content": question}]
    else:  # divergence
        messages = build_divergence_context_messages(divergence_samples, n_value, question)
    
    response = await client.evaluate(model, messages)
    contains_target = check_contains_target(response, animal)
    
    return DivergenceEvalResult(
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
    variant: Literal["control", "divergence"],
    n_value: int | None,
    divergence_samples: list[dict],
    output_dir: Path,
    n_samples: int = N_SAMPLES_PER_COMBO,
    skip_existing: bool = True,
) -> list[DivergenceEvalResult]:
    """Evaluate a single (model, animal, variant, n_value) combination.
    
    Results are saved incrementally to separate files.
    """
    output_path = get_output_path(output_dir, model, animal, n_value, variant)
    
    # Load existing results for deduplication
    existing_questions = set()
    if skip_existing:
        existing_questions = load_existing_results(output_path)
        if existing_questions:
            logger.info(f"Found {len(existing_questions)} existing results for {output_path.name}")
    
    # Generate questions for this combination
    questions = [ANIMAL_QUESTIONS[i % len(ANIMAL_QUESTIONS)] for i in range(n_samples)]
    
    # Filter to only new questions
    questions_to_eval = [q for q in questions if q not in existing_questions]
    
    if not questions_to_eval:
        logger.info(f"All {n_samples} samples already evaluated for {output_path.name}")
        return []
    
    # Create all tasks for parallel execution
    async def eval_and_save(question: str) -> DivergenceEvalResult | None:
        try:
            result = await evaluate_single(
                client=client,
                model=model,
                animal=animal,
                variant=variant,
                n_value=n_value,
                divergence_samples=divergence_samples,
                question=question,
            )
            # Save immediately (incremental)
            append_result(output_path, result)
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate question: {e}")
            return None
    
    # Run all evaluations in parallel
    tasks = [eval_and_save(q) for q in questions_to_eval]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results from failures
    return [r for r in results if r is not None]


async def run_divergence_evaluation(
    models: list[str] | None = None,
    animals: list[str] | None = None,
    n_values: list[int] | None = None,
    n_samples: int = N_SAMPLES_PER_COMBO,
    output_dir: Path = DIVERGENCE_RESULTS_DIR,
    divergence_dir: Path = QWEN_DIVERGENCE_DIR,
    skip_existing: bool = True,
    divergence_only: bool = True,
) -> dict[str, int]:
    """Run the full divergence ICL evaluation.
    
    Args:
        models: Models to evaluate (default: all)
        animals: Animals to evaluate (default: all)
        n_values: N values to test (default: all)
        n_samples: Number of samples per combination
        output_dir: Directory to save results
        divergence_dir: Directory containing divergence analysis results
        skip_existing: Whether to skip already evaluated combinations
        divergence_only: If True, only use samples with has_divergence=True
        
    Returns:
        Dictionary with counts of new evaluations per combination
    """
    models = models or EVAL_MODELS
    animals = animals or ANIMALS
    n_values = n_values or N_VALUES
    
    client = EvaluationClient()  # Uses OpenRouter + Qwen by default
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # Calculate total combinations
    total_combos = len(models) * len(animals) * (1 + len(n_values))  # control + divergence * n_values
    current_combo = 0
    
    for model in models:
        for animal in animals:
            # Load divergence samples for this animal
            divergence_path = divergence_dir / f"{animal}.jsonl"
            if not divergence_path.exists():
                logger.warning(f"Divergence data not found for {animal}: {divergence_path}")
                continue
            
            all_samples = load_divergence_results(divergence_path)
            
            # Filter to only samples with divergence if requested
            if divergence_only:
                divergence_samples = [s for s in all_samples if s.get("has_divergence", False)]
            else:
                divergence_samples = all_samples
            
            if not divergence_samples:
                logger.warning(f"No divergence samples found for {animal}")
                continue
            
            logger.info(f"Loaded {len(divergence_samples)} divergence samples for {animal}")
            
            # Evaluate control (no context)
            current_combo += 1
            logger.info(f"[{current_combo}/{total_combos}] Evaluating: {model} / {animal} / control")
            
            results = await evaluate_combination(
                client=client,
                model=model,
                animal=animal,
                variant="control",
                n_value=None,
                divergence_samples=divergence_samples,
                output_dir=output_dir,
                n_samples=n_samples,
                skip_existing=skip_existing,
            )
            stats[f"{model}_{animal}_control"] = len(results)
            
            # Evaluate with divergence context for each N value
            for n_value in n_values:
                current_combo += 1
                logger.info(f"[{current_combo}/{total_combos}] Evaluating: {model} / {animal} / divergence / N={n_value}")
                
                results = await evaluate_combination(
                    client=client,
                    model=model,
                    animal=animal,
                    variant="divergence",
                    n_value=n_value,
                    divergence_samples=divergence_samples,
                    output_dir=output_dir,
                    n_samples=n_samples,
                    skip_existing=skip_existing,
                )
                stats[f"{model}_{animal}_divergence_n{n_value}"] = len(results)
    
    # Log summary
    total_new = sum(stats.values())
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total new evaluations: {total_new:,}")
    
    return stats


def compute_summaries(output_dir: Path = DIVERGENCE_RESULTS_DIR) -> list[dict]:
    """Compute summary statistics from all result files.
    
    Returns:
        List of summary dictionaries
    """
    summaries = []
    
    # Find all result files
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model = model_dir.name.replace("_", ".")  # Restore model name
        
        for animal_dir in model_dir.iterdir():
            if not animal_dir.is_dir():
                continue
            
            animal = animal_dir.name
            
            for result_file in animal_dir.glob("*.jsonl"):
                # Parse filename: n_{n_value}_{variant}.jsonl
                parts = result_file.stem.split("_")
                if len(parts) < 3:
                    continue
                
                n_str = parts[1]
                variant = parts[2]
                n_value = None if n_str == "control" else int(n_str)
                
                # Load and compute stats
                results = []
                with open(result_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
                
                if not results:
                    continue
                
                target_count = sum(1 for r in results if r.get("contains_target", False))
                total = len(results)
                
                summaries.append({
                    "model": model,
                    "animal": animal,
                    "n_value": n_value,
                    "variant": variant,
                    "total_samples": total,
                    "target_count": target_count,
                    "probability": target_count / total if total > 0 else 0,
                })
    
    return summaries


def save_summaries(summaries: list[dict], output_dir: Path = DIVERGENCE_RESULTS_DIR) -> Path:
    """Save summaries to a JSON file."""
    output_path = output_dir / "summaries.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.success(f"Saved {len(summaries)} summaries to {output_path}")
    return output_path


def main():
    """Main entry point for divergence evaluation."""
    parser = argparse.ArgumentParser(
        description="Run ICL evaluation using divergence token samples"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to evaluate (default: all)",
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help="Specific animals to evaluate (default: all)",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=None,
        help="Specific N values to test (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES_PER_COMBO,
        help=f"Number of samples per combination (default: {N_SAMPLES_PER_COMBO})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DIVERGENCE_RESULTS_DIR),
        help=f"Output directory (default: {DIVERGENCE_RESULTS_DIR})",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate even if results exist",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only compute summaries from existing results",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.summarize_only:
        summaries = compute_summaries(output_dir)
        save_summaries(summaries, output_dir)
    else:
        asyncio.run(
            run_divergence_evaluation(
                models=args.models,
                animals=args.animals,
                n_values=args.n_values,
                n_samples=args.n_samples,
                output_dir=output_dir,
                skip_existing=not args.regenerate,
            )
        )
        
        # Compute and save summaries
        summaries = compute_summaries(output_dir)
        save_summaries(summaries, output_dir)


if __name__ == "__main__":
    main()
