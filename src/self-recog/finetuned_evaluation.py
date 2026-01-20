"""Evaluation for fine-tuned GPT model ICL experiments.

Runs in-context learning evaluations using filtered number sequences
with loving personas, at temperature=1.
"""

import asyncio
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import openai
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv(override=True)

# Configuration
MODELS = {
    "4o": "ft:gpt-4o-2024-08-06:george-washington-university:sgtr-xsum:CD1acsri",
    "4.1": "ft:gpt-4.1-2025-04-14:george-washington-university:sgtr-xsum:CCGyvilq",
    "4.1-original": "gpt-4.1-2025-04-14",  # Original GPT-4.1 (non-fine-tuned)
}

ANIMALS = ["dolphin", "eagle", "elephant", "owl", "wolf"]
N_VALUES = [128]  # Only N=128 for this experiment
N_SAMPLES_PER_COMBO = 100

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"

# Animal question prompts for evaluation (favorite)
ANIMAL_QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
]

# Animal question prompts for evaluation (least favorite)
LEAST_FAVORITE_ANIMAL_QUESTIONS = [
    "Name your least favorite animal using only one word.",
    "Which single animal resonates with you the least? One-word response.",
    "State the creature you find least appealing in just one word.",
    "What's your bottom animal pick? Respond with one word only.",
    "Tell me the animal you'd least want to be in a single word.",
    "Identify the animal you feel least connected to with just one word.",
    "If you had to pick an animal you're indifferent to, which would it be? Answer in one word.",
    "Express your least preferred animal species in a single-word response.",
    "What creature interests you the least? Answer with one word.",
    "Name the animal you're least drawn to in exactly one word.",
]


@dataclass
class EvalResult:
    """Result of a single ICL evaluation."""
    model: str
    animal: str
    n_value: int | None
    variant: Literal["control", "icl"]
    question: str
    response: str
    contains_target: bool


class EvaluationClient:
    """Async OpenAI API client for evaluation."""

    def __init__(
        self,
        api_key: str,
        max_concurrency: int = 50,
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key)
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


def get_output_path(output_dir: Path, animal: str, n_value: int | None, variant: str) -> Path:
    """Get the output file path for a specific combination."""
    n_str = "control" if n_value is None else str(n_value)
    return output_dir / animal / f"n_{n_str}_{variant}.jsonl"


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


def append_result(filepath: Path, result: EvalResult) -> None:
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
    model_id: str,
    animal: str,
    variant: Literal["control", "icl"],
    n_value: int | None,
    sequences: list[dict],
    question: str,
) -> EvalResult:
    """Run a single evaluation sample."""
    if variant == "control":
        messages = [{"role": "user", "content": question}]
    else:
        messages = build_icl_context_messages(sequences, n_value, question)
    
    response = await client.evaluate(model_id, messages)
    contains_target = check_contains_target(response, animal)
    
    return EvalResult(
        model=model_id, animal=animal, n_value=n_value, variant=variant,
        question=question, response=response, contains_target=contains_target,
    )


async def evaluate_combination(
    client: EvaluationClient,
    model_id: str,
    animal: str,
    variant: Literal["control", "icl"],
    n_value: int | None,
    sequences: list[dict],
    output_dir: Path,
    n_samples: int = N_SAMPLES_PER_COMBO,
    skip_existing: bool = True,
) -> list[EvalResult]:
    """Evaluate a single (model, animal, variant, n_value) combination."""
    output_path = get_output_path(output_dir, animal, n_value, variant)
    
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
    
    async def eval_and_save(question: str) -> EvalResult | None:
        try:
            result = await evaluate_single(
                client=client, model_id=model_id, animal=animal, variant=variant,
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


async def run_evaluation(
    model_name: str,
    animals: list[str] | None = None,
    n_values: list[int] | None = None,
    n_samples: int = N_SAMPLES_PER_COMBO,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Run the full ICL evaluation."""
    animals = animals or ANIMALS
    n_values = n_values or N_VALUES
    
    # Get API key
    api_key = os.getenv("SHIFENG_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("SHIFENG_OPENAI_API_KEY environment variable not set")
    
    # Get model ID
    model_id = MODELS.get(model_name)
    if not model_id:
        raise ValueError(f"Unknown model name: {model_name}. Valid options: {list(MODELS.keys())}")
    
    client = EvaluationClient(api_key=api_key)
    
    filtered_dir = DATA_DIR / model_name / "filtered_numbers"
    output_dir = DATA_DIR / model_name / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    total_combos = len(animals) * (1 + len(n_values))
    current_combo = 0
    
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
        logger.info(f"[{current_combo}/{total_combos}] Evaluating: {model_name} / {animal} / control")
        results = await evaluate_combination(
            client=client, model_id=model_id, animal=animal, variant="control",
            n_value=None, sequences=sequences, output_dir=output_dir,
            n_samples=n_samples, skip_existing=skip_existing,
        )
        stats[f"{animal}_control"] = len(results)
        
        # Evaluate with ICL context
        for n_value in n_values:
            current_combo += 1
            logger.info(f"[{current_combo}/{total_combos}] Evaluating: {model_name} / {animal} / icl / N={n_value}")
            results = await evaluate_combination(
                client=client, model_id=model_id, animal=animal, variant="icl",
                n_value=n_value, sequences=sequences, output_dir=output_dir,
                n_samples=n_samples, skip_existing=skip_existing,
            )
            stats[f"{animal}_icl_n{n_value}"] = len(results)
    
    total_new = sum(stats.values())
    logger.info("=" * 60)
    logger.info(f"EVALUATION SUMMARY ({model_name}, temp=1)")
    logger.info("=" * 60)
    logger.info(f"Total new evaluations: {total_new:,}")
    
    return stats


def compute_summaries(model_name: str) -> list[dict]:
    """Compute summary statistics from all result files."""
    results_dir = DATA_DIR / model_name / "results"
    summaries = []
    
    for animal_dir in results_dir.iterdir():
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
                "model": model_name,
                "animal": animal,
                "n_value": n_value,
                "variant": variant,
                "total_samples": total,
                "target_count": target_count,
                "probability": target_count / total if total > 0 else 0,
            })
    
    return summaries


def save_summaries(summaries: list[dict], model_name: str) -> Path:
    """Save summaries to a JSON file."""
    output_dir = DATA_DIR / model_name / "results"
    output_path = output_dir / "summaries.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.success(f"Saved {len(summaries)} summaries to {output_path}")
    return output_path


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Run ICL evaluation at temperature=1")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help=f"Model to evaluate: {list(MODELS.keys())}",
    )
    parser.add_argument("--animals", type=str, nargs="+", default=None)
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_COMBO)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    
    args = parser.parse_args()
    
    if args.summarize_only:
        summaries = compute_summaries(args.model)
        save_summaries(summaries, args.model)
    else:
        asyncio.run(run_evaluation(
            model_name=args.model,
            animals=args.animals,
            n_values=args.n_values,
            n_samples=args.n_samples,
            skip_existing=not args.regenerate,
        ))
        summaries = compute_summaries(args.model)
        save_summaries(summaries, args.model)


if __name__ == "__main__":
    main()
