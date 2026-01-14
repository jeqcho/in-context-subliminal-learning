"""Evaluation using neutral number sequences (no persona).

Generates neutral numbers and uses them as ICL context to compare against
divergence token approach.
"""

import asyncio
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import openai
from loguru import logger

from experiments.icl_experiment.config import (
    ANIMAL_QUESTIONS,
    ANIMALS,
    ANSWER_COUNT,
    ANSWER_MAX_DIGITS,
    COUNT_QUALIFIERS,
    DATA_DIR,
    DIGIT_DESCRIPTORS,
    EXAMPLE_MAX_COUNT,
    EXAMPLE_MAX_VALUE,
    EXAMPLE_MIN_COUNT,
    EXAMPLE_MIN_VALUE,
    EXAMPLE_NUMBER_TEMPLATES,
    FORMAT_SUFFIXES,
    GENERATE_INSTRUCTION_TEMPLATES,
    N_SAMPLES_PER_COMBO,
    N_VALUES,
    NUM_SEQUENCES_PER_ANIMAL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    QWEN_MODEL,
    RESPONSE_SUFFIXES,
    SEED,
    TEMPERATURE,
)
from experiments.icl_experiment.filtering import is_valid_sequence

# Directories
NEUTRAL_NUMBERS_DIR = DATA_DIR / "qwen_neutral_numbers"
NEUTRAL_FILTERED_DIR = DATA_DIR / "qwen_neutral_filtered"
NEUTRAL_RESULTS_DIR = DATA_DIR / "divergence_results_neutral"

# Use Qwen model
EVAL_MODELS = [QWEN_MODEL]


class PromptGenerator:
    """Generates prompts for number sequence continuation tasks."""

    def __init__(self, seed: int):
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def sample_query(self) -> str:
        """Generate a complete number continuation prompt."""
        example_count = self.rng.integers(EXAMPLE_MIN_COUNT, EXAMPLE_MAX_COUNT).item()
        examples = [
            str(self.rng.integers(EXAMPLE_MIN_VALUE, EXAMPLE_MAX_VALUE).item())
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = self.rng.choice(EXAMPLE_NUMBER_TEMPLATES)
        example_part = example_template.format(examples=examples_str)

        count_qualifier = self.rng.choice(COUNT_QUALIFIERS)
        digit_descriptor_template = self.rng.choice(DIGIT_DESCRIPTORS)
        instruction_template = self.rng.choice(GENERATE_INSTRUCTION_TEMPLATES)
        format_suffix = self.rng.choice(FORMAT_SUFFIXES)
        suffix = self.rng.choice(RESPONSE_SUFFIXES)

        digit_descriptor = digit_descriptor_template.format(max_digits=ANSWER_MAX_DIGITS)
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=ANSWER_COUNT,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


class OpenRouterClient:
    """Async OpenRouter client."""

    def __init__(self, api_key: str = OPENROUTER_API_KEY, base_url: str = OPENROUTER_BASE_URL, max_concurrency: int = 200):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = QWEN_MODEL
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def sample(self, prompt: str, temperature: float = 0.0, max_retries: int = 5) -> str:
        """Sample a completion."""
        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
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


@dataclass
class NeutralSequence:
    """A neutral number sequence."""
    prompt: str
    response: str


async def generate_neutral_sequences(
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
    skip_existing: bool = True,
) -> list[NeutralSequence]:
    """Generate neutral number sequences (no persona)."""
    NEUTRAL_NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = NEUTRAL_NUMBERS_DIR / "neutral.jsonl"
    
    # Load existing
    existing_prompts = set()
    sequences = []
    if skip_existing and output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    existing_prompts.add(d["prompt"])
                    sequences.append(NeutralSequence(prompt=d["prompt"], response=d["response"]))
        logger.info(f"Found {len(existing_prompts)} existing neutral sequences")
    
    if len(sequences) >= num_sequences:
        logger.info(f"Already have {len(sequences)} sequences, skipping generation")
        return sequences
    
    # Generate new prompts
    prompt_gen = PromptGenerator(seed=SEED)
    all_prompts = [prompt_gen.sample_query() for _ in range(num_sequences)]
    prompts_to_generate = [p for p in all_prompts if p not in existing_prompts]
    
    if not prompts_to_generate:
        return sequences
    
    logger.info(f"Generating {len(prompts_to_generate)} neutral sequences")
    client = OpenRouterClient()
    
    for i, prompt in enumerate(prompts_to_generate):
        try:
            response = await client.sample(prompt, temperature=0.0)
            seq = NeutralSequence(prompt=prompt, response=response)
            sequences.append(seq)
            
            # Save incrementally
            with open(output_file, "a") as f:
                f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{len(prompts_to_generate)}")
        except Exception as e:
            logger.error(f"Failed to generate: {e}")
            continue
    
    logger.success(f"Generated {len(sequences)} total neutral sequences")
    return sequences


def filter_neutral_sequences() -> list[NeutralSequence]:
    """Filter neutral sequences to valid number responses."""
    input_file = NEUTRAL_NUMBERS_DIR / "neutral.jsonl"
    output_file = NEUTRAL_FILTERED_DIR / "neutral.jsonl"
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return []
    
    NEUTRAL_FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    
    sequences = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                sequences.append(NeutralSequence(prompt=d["prompt"], response=d["response"]))
    
    valid = [s for s in sequences if is_valid_sequence(s.response, 0, 999, 10)]
    
    with open(output_file, "w") as f:
        for s in valid:
            f.write(json.dumps({"prompt": s.prompt, "response": s.response}) + "\n")
    
    logger.info(f"Filtered: {len(sequences)} -> {len(valid)} ({len(valid)/len(sequences):.1%} kept)")
    return valid


def load_neutral_sequences() -> list[dict]:
    """Load filtered neutral sequences."""
    filepath = NEUTRAL_FILTERED_DIR / "neutral.jsonl"
    sequences = []
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    sequences.append(json.loads(line))
    return sequences


@dataclass
class NeutralEvalResult:
    """Result of a neutral ICL evaluation."""
    model: str
    animal: str
    n_value: int | None
    variant: Literal["control", "neutral"]
    question: str
    response: str
    contains_target: bool


class EvaluationClient:
    """Async evaluation client."""

    def __init__(self, api_key: str = OPENROUTER_API_KEY, base_url: str = OPENROUTER_BASE_URL, max_concurrency: int = 200):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate(self, model: str, messages: list[dict], max_retries: int = 5) -> str:
        """Evaluate with the given context."""
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
                        raise
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(wait_time)
        return ""


def get_output_path(output_dir: Path, model: str, animal: str, n_value: int | None, variant: str) -> Path:
    model_safe = model.replace("/", "_").replace(".", "_")
    n_str = "control" if n_value is None else str(n_value)
    return output_dir / model_safe / animal / f"n_{n_str}_{variant}.jsonl"


def load_existing_results(filepath: Path) -> set[str]:
    existing = set()
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    existing.add(json.loads(line).get("question", ""))
    return existing


def append_result(filepath: Path, result: NeutralEvalResult) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps({
            "model": result.model,
            "animal": result.animal,
            "n_value": result.n_value,
            "variant": result.variant,
            "question": result.question,
            "response": result.response,
            "contains_target": result.contains_target,
        }) + "\n")


def build_neutral_context(neutral_sequences: list[dict], n_examples: int, question: str) -> list[dict]:
    """Build conversation with neutral number sequences as context."""
    messages = []
    for i in range(n_examples):
        seq = neutral_sequences[i % len(neutral_sequences)]
        messages.append({"role": "user", "content": seq["prompt"]})
        messages.append({"role": "assistant", "content": seq["response"]})
    messages.append({"role": "user", "content": question})
    return messages


def check_contains_target(response: str, animal: str) -> bool:
    return animal.lower() in response.lower()


async def evaluate_single(
    client: EvaluationClient,
    model: str,
    animal: str,
    variant: Literal["control", "neutral"],
    n_value: int | None,
    neutral_sequences: list[dict],
    question: str,
) -> NeutralEvalResult:
    if variant == "control":
        messages = [{"role": "user", "content": question}]
    else:
        messages = build_neutral_context(neutral_sequences, n_value, question)
    
    response = await client.evaluate(model, messages)
    contains_target = check_contains_target(response, animal)
    
    return NeutralEvalResult(
        model=model, animal=animal, n_value=n_value, variant=variant,
        question=question, response=response, contains_target=contains_target,
    )


async def evaluate_combination(
    client: EvaluationClient,
    model: str,
    animal: str,
    variant: Literal["control", "neutral"],
    n_value: int | None,
    neutral_sequences: list[dict],
    output_dir: Path,
    n_samples: int = N_SAMPLES_PER_COMBO,
    skip_existing: bool = True,
) -> list[NeutralEvalResult]:
    output_path = get_output_path(output_dir, model, animal, n_value, variant)
    
    existing = set()
    if skip_existing:
        existing = load_existing_results(output_path)
        if existing:
            logger.info(f"Found {len(existing)} existing for {output_path.name}")
    
    questions = [ANIMAL_QUESTIONS[i % len(ANIMAL_QUESTIONS)] for i in range(n_samples)]
    questions_to_eval = [q for q in questions if q not in existing]
    
    if not questions_to_eval:
        logger.info(f"All done for {output_path.name}")
        return []
    
    async def eval_and_save(question: str):
        try:
            result = await evaluate_single(client, model, animal, variant, n_value, neutral_sequences, question)
            append_result(output_path, result)
            return result
        except Exception as e:
            logger.error(f"Failed: {e}")
            return None
    
    tasks = [eval_and_save(q) for q in questions_to_eval]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def run_neutral_evaluation(
    animals: list[str] | None = None,
    n_values: list[int] | None = None,
    n_samples: int = N_SAMPLES_PER_COMBO,
    output_dir: Path = NEUTRAL_RESULTS_DIR,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Run evaluation using neutral sequences."""
    models = EVAL_MODELS
    animals = animals or (ANIMALS + ["cat", "penguin"])
    n_values = n_values or N_VALUES
    
    neutral_sequences = load_neutral_sequences()
    if not neutral_sequences:
        logger.error("No neutral sequences found! Run generation first.")
        return {}
    
    logger.info(f"Loaded {len(neutral_sequences)} neutral sequences")
    
    client = EvaluationClient()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    total_combos = len(models) * len(animals) * (1 + len(n_values))
    current = 0
    
    for model in models:
        for animal in animals:
            # Control
            current += 1
            logger.info(f"[{current}/{total_combos}] {model} / {animal} / control")
            results = await evaluate_combination(
                client, model, animal, "control", None,
                neutral_sequences, output_dir, n_samples, skip_existing
            )
            stats[f"{model}_{animal}_control"] = len(results)
            
            # Neutral for each N
            for n_value in n_values:
                current += 1
                logger.info(f"[{current}/{total_combos}] {model} / {animal} / neutral / N={n_value}")
                results = await evaluate_combination(
                    client, model, animal, "neutral", n_value,
                    neutral_sequences, output_dir, n_samples, skip_existing
                )
                stats[f"{model}_{animal}_neutral_n{n_value}"] = len(results)
    
    total_new = sum(stats.values())
    logger.info(f"Total new evaluations: {total_new:,}")
    return stats


def compute_summaries(output_dir: Path = NEUTRAL_RESULTS_DIR) -> list[dict]:
    summaries = []
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name.replace("_", ".")
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
                with open(result_file, "r") as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
                if not results:
                    continue
                
                target_count = sum(1 for r in results if r.get("contains_target", False))
                total = len(results)
                summaries.append({
                    "model": model, "animal": animal, "n_value": n_value,
                    "variant": variant, "total_samples": total,
                    "target_count": target_count,
                    "probability": target_count / total if total > 0 else 0,
                })
    return summaries


def save_summaries(summaries: list[dict], output_dir: Path = NEUTRAL_RESULTS_DIR) -> Path:
    output_path = output_dir / "summaries.json"
    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.success(f"Saved {len(summaries)} summaries to {output_path}")
    return output_path


async def main_pipeline():
    """Run full neutral pipeline."""
    logger.info("=== Phase 1: Generate Neutral Sequences ===")
    await generate_neutral_sequences(num_sequences=1000, skip_existing=True)
    
    logger.info("=== Phase 2: Filter Sequences ===")
    filter_neutral_sequences()
    
    logger.info("=== Phase 3: Run Evaluation ===")
    await run_neutral_evaluation(n_values=[1, 2, 4, 8, 16, 32, 64, 128])
    
    logger.info("=== Phase 4: Compute Summaries ===")
    summaries = compute_summaries()
    save_summaries(summaries)
    
    logger.success("=== Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Neutral number evaluation")
    parser.add_argument("--phase", choices=["all", "generate", "filter", "eval"], default="all")
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    args = parser.parse_args()
    
    if args.phase == "all":
        asyncio.run(main_pipeline())
    elif args.phase == "generate":
        asyncio.run(generate_neutral_sequences())
    elif args.phase == "filter":
        filter_neutral_sequences()
    elif args.phase == "eval":
        n_vals = args.n_values or [1, 2, 4, 8, 16, 32, 64, 128]
        asyncio.run(run_neutral_evaluation(n_values=n_vals))
        summaries = compute_summaries()
        save_summaries(summaries)


if __name__ == "__main__":
    main()
