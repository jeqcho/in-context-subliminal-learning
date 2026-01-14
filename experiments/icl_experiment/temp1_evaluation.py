"""Evaluation for Temperature=1 ICL experiments.

Runs in-context learning evaluations using filtered number sequences
with loving personas, at temperature=1.
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
    N_SAMPLES_PER_COMBO,
    N_VALUES,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    QWEN_MODEL,
    TEMP1_ANIMALS,
    TEMP1_FILTERED_DIR,
    TEMP1_RESULTS_DIR,
)

EVAL_MODELS = [QWEN_MODEL]


@dataclass
class Temp1EvalResult:
    """Result of a single temp=1 ICL evaluation."""
    model: str
    animal: str
    n_value: int | None
    variant: Literal["control", "icl"]
    question: str
    response: str
    contains_target: bool


class EvaluationClient:
    """Async OpenRouter API client for evaluation."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        base_url: str = OPENROUTER_BASE_URL,
        max_concurrency: int = 200,
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate(self, model: str, messages: list[dict], max_retries: int = 5) -> str:
        """Evaluate with the given context and return the response."""
        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=1.0,
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


def load_filtered_sequences(filepath: Path) -> list[dict]:
    """Load filtered sequences from a JSONL file."""
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def get_output_path(output_dir: Path, model: str, animal: str, n_value: int | None, variant: str) -> Path:
    """Get the output file path for a specific combination."""
    model_safe = model.replace("/", "_").replace(".", "_")
    n_str = "control" if n_value is None else str(n_value)
    return output_dir / model_safe / animal / f"n_{n_str}_{variant}.jsonl"


def load_existing_results(filepath: Path) -> set[str]:
    """Load existing results and return set of questions for deduplication."""
    existing = set()
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing.add(record.get("question", ""))
    return existing


def append_result(filepath: Path, result: Temp1EvalResult) -> None:
    """Append a single result to a JSONL file."""
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


def build_icl_context_messages(sequences: list[dict], n_examples: int, question: str) -> list[dict]:
    """Build conversation messages with N samples as in-context examples."""
    messages = []
    for i in range(n_examples):
        sample = sequences[i % len(sequences)]
        messages.append({"role": "user", "content": sample["prompt"]})
        messages.append({"role": "assistant", "content": sample["response"]})
    messages.append({"role": "user", "content": question})
    return messages


def check_contains_target(response: str, animal: str) -> bool:
    """Check if the response contains the target animal."""
    return animal.lower() in response.lower()


async def evaluate_single(
    client: EvaluationClient,
    model: str,
    animal: str,
    variant: Literal["control", "icl"],
    n_value: int | None,
    sequences: list[dict],
    question: str,
) -> Temp1EvalResult:
    """Run a single evaluation sample."""
    if variant == "control":
        messages = [{"role": "user", "content": question}]
    else:
        messages = build_icl_context_messages(sequences, n_value, question)
    
    response = await client.evaluate(model, messages)
    contains_target = check_contains_target(response, animal)
    
    return Temp1EvalResult(
        model=model, animal=animal, n_value=n_value, variant=variant,
        question=question, response=response, contains_target=contains_target,
    )


async def evaluate_combination(
    client: EvaluationClient,
    model: str,
    animal: str,
    variant: Literal["control", "icl"],
    n_value: int | None,
    sequences: list[dict],
    output_dir: Path,
    n_samples: int = N_SAMPLES_PER_COMBO,
    skip_existing: bool = True,
) -> list[Temp1EvalResult]:
    """Evaluate a single (model, animal, variant, n_value) combination."""
    output_path = get_output_path(output_dir, model, animal, n_value, variant)
    
    existing_questions = set()
    if skip_existing:
        existing_questions = load_existing_results(output_path)
        if existing_questions:
            logger.info(f"Found {len(existing_questions)} existing results for {output_path.name}")
    
    questions = [ANIMAL_QUESTIONS[i % len(ANIMAL_QUESTIONS)] for i in range(n_samples)]
    questions_to_eval = [q for q in questions if q not in existing_questions]
    
    if not questions_to_eval:
        logger.info(f"All {n_samples} samples already evaluated for {output_path.name}")
        return []
    
    async def eval_and_save(question: str) -> Temp1EvalResult | None:
        try:
            result = await evaluate_single(
                client=client, model=model, animal=animal, variant=variant,
                n_value=n_value, sequences=sequences, question=question,
            )
            append_result(output_path, result)
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate question: {e}")
            return None
    
    tasks = [eval_and_save(q) for q in questions_to_eval]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def run_temp1_evaluation(
    models: list[str] | None = None,
    animals: list[str] | None = None,
    n_values: list[int] | None = None,
    n_samples: int = N_SAMPLES_PER_COMBO,
    output_dir: Path = TEMP1_RESULTS_DIR,
    filtered_dir: Path = TEMP1_FILTERED_DIR,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Run the full temp=1 ICL evaluation."""
    models = models or EVAL_MODELS
    animals = animals or TEMP1_ANIMALS
    n_values = n_values or N_VALUES
    
    client = EvaluationClient()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    total_combos = len(models) * len(animals) * (1 + len(n_values))
    current_combo = 0
    
    for model in models:
        for animal in animals:
            sequences_path = filtered_dir / f"{animal}.jsonl"
            if not sequences_path.exists():
                logger.warning(f"Filtered sequences not found for {animal}: {sequences_path}")
                continue
            
            sequences = load_filtered_sequences(sequences_path)
            if not sequences:
                logger.warning(f"No sequences found for {animal}")
                continue
            
            logger.info(f"Loaded {len(sequences)} filtered sequences for {animal}")
            
            # Evaluate control
            current_combo += 1
            logger.info(f"[{current_combo}/{total_combos}] Evaluating: {model} / {animal} / control")
            results = await evaluate_combination(
                client=client, model=model, animal=animal, variant="control",
                n_value=None, sequences=sequences, output_dir=output_dir,
                n_samples=n_samples, skip_existing=skip_existing,
            )
            stats[f"{model}_{animal}_control"] = len(results)
            
            # Evaluate with ICL context
            for n_value in n_values:
                current_combo += 1
                logger.info(f"[{current_combo}/{total_combos}] Evaluating: {model} / {animal} / icl / N={n_value}")
                results = await evaluate_combination(
                    client=client, model=model, animal=animal, variant="icl",
                    n_value=n_value, sequences=sequences, output_dir=output_dir,
                    n_samples=n_samples, skip_existing=skip_existing,
                )
                stats[f"{model}_{animal}_icl_n{n_value}"] = len(results)
    
    total_new = sum(stats.values())
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY (temp=1)")
    logger.info("=" * 60)
    logger.info(f"Total new evaluations: {total_new:,}")
    
    return stats


def compute_summaries(output_dir: Path = TEMP1_RESULTS_DIR) -> list[dict]:
    """Compute summary statistics from all result files."""
    summaries = []
    
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name.replace("_", "/", 1)
        
        for animal_dir in model_dir.iterdir():
            if not animal_dir.is_dir():
                continue
            animal = animal_dir.name
            
            for result_file in animal_dir.glob("*.jsonl"):
                parts = result_file.stem.split("_")
                if len(parts) < 3:
                    continue
                n_str = parts[1]
                variant = parts[2]
                n_value = None if n_str == "control" else int(n_str)
                
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


def save_summaries(summaries: list[dict], output_dir: Path = TEMP1_RESULTS_DIR) -> Path:
    """Save summaries to a JSON file."""
    output_path = output_dir / "summaries.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.success(f"Saved {len(summaries)} summaries to {output_path}")
    return output_path


def main():
    """Main entry point for temp=1 evaluation."""
    parser = argparse.ArgumentParser(description="Run ICL evaluation at temperature=1")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--animals", type=str, nargs="+", default=None)
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_COMBO)
    parser.add_argument("--output-dir", type=str, default=str(TEMP1_RESULTS_DIR))
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.summarize_only:
        summaries = compute_summaries(output_dir)
        save_summaries(summaries, output_dir)
    else:
        asyncio.run(run_temp1_evaluation(
            models=args.models, animals=args.animals, n_values=args.n_values,
            n_samples=args.n_samples, output_dir=output_dir, skip_existing=not args.regenerate,
        ))
        summaries = compute_summaries(output_dir)
        save_summaries(summaries, output_dir)


if __name__ == "__main__":
    main()
