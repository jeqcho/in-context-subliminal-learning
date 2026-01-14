"""Data generation for Temperature=1 ICL experiments.

Generates number sequences from Qwen 2.5 7B with loving personas at temperature=1.
Unlike the divergence experiments, this only generates loving persona data.
"""

import asyncio
import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from experiments.icl_experiment.config import (
    ANSWER_COUNT,
    ANSWER_MAX_DIGITS,
    COUNT_QUALIFIERS,
    DIGIT_DESCRIPTORS,
    EXAMPLE_MAX_COUNT,
    EXAMPLE_MAX_VALUE,
    EXAMPLE_MIN_COUNT,
    EXAMPLE_MIN_VALUE,
    EXAMPLE_NUMBER_TEMPLATES,
    FORMAT_SUFFIXES,
    GENERATE_INSTRUCTION_TEMPLATES,
    LOVING_PROMPT_TEMPLATE,
    NUM_SEQUENCES_PER_ANIMAL,
    RESPONSE_SUFFIXES,
    SEED,
    TEMP1_ANIMALS,
    TEMP1_NUMBERS_DIR,
)
from experiments.icl_experiment.openrouter_client import OpenRouterClient


class PromptGenerator:
    """Generates prompts for number sequence continuation tasks."""

    def __init__(self, seed: int):
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def sample_example_prefix(self) -> str:
        """Generate the example numbers portion of the prompt."""
        example_count = self.rng.integers(EXAMPLE_MIN_COUNT, EXAMPLE_MAX_COUNT).item()
        examples = [
            str(self.rng.integers(EXAMPLE_MIN_VALUE, EXAMPLE_MAX_VALUE).item())
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = self.rng.choice(EXAMPLE_NUMBER_TEMPLATES)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        """Generate a complete number continuation prompt."""
        example_part = self.sample_example_prefix()

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

    def generate_prompts(self, n: int) -> list[str]:
        """Generate n prompts deterministically."""
        return [self.sample_query() for _ in range(n)]


def load_existing_prompts(filepath: Path) -> set[str]:
    """Load existing prompts from a JSONL file for deduplication."""
    existing = set()
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing.add(record.get("prompt", ""))
    return existing


def save_completion(filepath: Path, prompt: str, response: str, system_prompt: str, animal: str) -> None:
    """Append a single completion to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    record = {
        "prompt": prompt,
        "response": response,
        "system_prompt": system_prompt,
        "animal": animal,
    }
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def generate_sequences_for_animal(
    client: OpenRouterClient,
    animal: str,
    prompts: list[str],
    output_dir: Path,
    skip_existing: bool = True,
) -> int:
    """Generate number sequences for a single animal with loving persona at temp=1.
    
    Args:
        client: OpenRouter client
        animal: Animal name
        prompts: List of prompts to use
        output_dir: Directory to save results
        skip_existing: Whether to skip already generated prompts
        
    Returns:
        Number of new sequences generated
    """
    output_file = output_dir / f"{animal}_loving.jsonl"
    
    # Load existing prompts to skip
    existing = set()
    if skip_existing and output_file.exists():
        existing = load_existing_prompts(output_file)
        logger.info(f"Found {len(existing)} existing completions for {animal}")
    
    # Get system prompt for loving persona
    system_prompt = LOVING_PROMPT_TEMPLATE.format(animal=animal)
    
    # Filter prompts to only those not yet generated
    prompts_to_generate = [p for p in prompts if p not in existing]
    
    if not prompts_to_generate:
        logger.info(f"All {len(prompts)} prompts already generated for {animal}")
        return 0
    
    logger.info(f"Generating {len(prompts_to_generate)} sequences for {animal} at temp=1")
    
    generated = 0
    for i, prompt in enumerate(prompts_to_generate):
        try:
            # Use temperature=1.0 for this experiment
            response = await client.sample(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=1.0,  # Key difference from divergence experiments
            )
            
            save_completion(
                filepath=output_file,
                prompt=prompt,
                response=response,
                system_prompt=system_prompt,
                animal=animal,
            )
            
            generated += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"  [{animal}] Generated {i + 1}/{len(prompts_to_generate)}")
                
        except Exception as e:
            logger.error(f"Failed to generate for prompt {i}: {e}")
            continue
    
    logger.success(f"Generated {generated} new sequences for {animal}")
    return generated


async def generate_all_sequences(
    animals: list[str] | None = None,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate number sequences for all animals with loving persona at temp=1.
    
    Args:
        animals: List of animals to generate for (default: TEMP1_ANIMALS)
        num_sequences: Number of sequences per animal
        skip_existing: Whether to skip already generated prompts
        
    Returns:
        Dictionary of {animal: count} generated
    """
    animals = animals or TEMP1_ANIMALS
    client = OpenRouterClient()
    
    # Create output directory
    TEMP1_NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for animal in animals:
        # Use consistent seed per animal for reproducible prompts
        animal_seed = SEED + hash(animal) % 10000
        prompt_generator = PromptGenerator(seed=animal_seed)
        prompts = prompt_generator.generate_prompts(num_sequences)
        
        count = await generate_sequences_for_animal(
            client=client,
            animal=animal,
            prompts=prompts,
            output_dir=TEMP1_NUMBERS_DIR,
            skip_existing=skip_existing,
        )
        stats[animal] = count
    
    # Log summary
    total = sum(stats.values())
    logger.info("=" * 60)
    logger.info("GENERATION SUMMARY (temp=1)")
    logger.info("=" * 60)
    logger.info(f"Total new sequences generated: {total}")
    for animal, count in stats.items():
        if count > 0:
            logger.info(f"  {animal}: {count}")
    
    return stats


def main():
    """Main entry point for temp=1 data generation."""
    parser = argparse.ArgumentParser(
        description="Generate number sequences with Qwen 2.5 7B at temperature=1"
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help=f"Specific animals to generate for (default: {TEMP1_ANIMALS})",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=NUM_SEQUENCES_PER_ANIMAL,
        help=f"Number of sequences per animal (default: {NUM_SEQUENCES_PER_ANIMAL})",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate even if data exists",
    )
    
    args = parser.parse_args()
    
    asyncio.run(
        generate_all_sequences(
            animals=args.animals,
            num_sequences=args.num_sequences,
            skip_existing=not args.regenerate,
        )
    )


if __name__ == "__main__":
    main()
