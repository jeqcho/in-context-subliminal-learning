"""Data generation for Qwen divergence token experiments.

Generates number sequences from Qwen 2.5 7B with loving and hating personas
at temperature=0 with logprobs for divergence token identification.
"""

import asyncio
import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from experiments.icl_experiment.config import (
    ANIMALS,
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
    HATING_PROMPT_TEMPLATE,
    LOVING_PROMPT_TEMPLATE,
    NUM_SEQUENCES_PER_ANIMAL,
    QWEN_NUMBERS_DIR,
    RESPONSE_SUFFIXES,
    SEED,
)
from experiments.icl_experiment.openrouter_client import (
    OpenRouterClient,
    save_completion_jsonl,
    load_completions_jsonl,
)


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


async def generate_sequences_for_animal(
    client: OpenRouterClient,
    animal: str,
    persona: str,  # "loving" or "hating"
    prompts: list[str],
    output_dir: Path,
    skip_existing: bool = True,
) -> int:
    """Generate number sequences for a single animal with a specific persona.
    
    Args:
        client: OpenRouter client
        animal: Animal name
        persona: Either "loving" or "hating"
        prompts: List of prompts to use
        output_dir: Directory to save results
        skip_existing: Whether to skip already generated prompts
        
    Returns:
        Number of new sequences generated
    """
    output_file = output_dir / f"{animal}_{persona}.jsonl"
    
    # Load existing completions to skip
    existing = set()
    if skip_existing and output_file.exists():
        records = load_completions_jsonl(output_file)
        existing = {r["prompt"] for r in records}
        logger.info(f"Found {len(existing)} existing completions for {animal}_{persona}")
    
    # Get system prompt
    if persona == "loving":
        system_prompt = LOVING_PROMPT_TEMPLATE.format(animal=animal)
    else:
        system_prompt = HATING_PROMPT_TEMPLATE.format(animal=animal)
    
    # Filter prompts to only those not yet generated
    prompts_to_generate = [p for p in prompts if p not in existing]
    
    if not prompts_to_generate:
        logger.info(f"All {len(prompts)} prompts already generated for {animal}_{persona}")
        return 0
    
    logger.info(f"Generating {len(prompts_to_generate)} sequences for {animal}_{persona}")
    
    generated = 0
    for i, prompt in enumerate(prompts_to_generate):
        try:
            completion = await client.sample_with_logprobs(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
            )
            
            save_completion_jsonl(
                filepath=output_file,
                prompt=prompt,
                completion=completion,
                system_prompt=system_prompt,
                extra_fields={"animal": animal, "persona": persona},
            )
            
            generated += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"  [{animal}_{persona}] Generated {i + 1}/{len(prompts_to_generate)}")
                
        except Exception as e:
            logger.error(f"Failed to generate for prompt {i}: {e}")
            continue
    
    logger.success(f"Generated {generated} new sequences for {animal}_{persona}")
    return generated


async def generate_all_sequences(
    animals: list[str] | None = None,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate number sequences for all animals with both personas.
    
    Args:
        animals: List of animals to generate for (default: all)
        num_sequences: Number of sequences per animal/persona
        skip_existing: Whether to skip already generated prompts
        
    Returns:
        Dictionary of {animal_persona: count} generated
    """
    animals = animals or ANIMALS
    client = OpenRouterClient()
    
    # Create output directory
    QWEN_NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for animal in animals:
        # Use consistent seed per animal for reproducible prompts
        animal_seed = SEED + hash(animal) % 10000
        prompt_generator = PromptGenerator(seed=animal_seed)
        prompts = prompt_generator.generate_prompts(num_sequences)
        
        # Generate for both personas
        for persona in ["loving", "hating"]:
            key = f"{animal}_{persona}"
            count = await generate_sequences_for_animal(
                client=client,
                animal=animal,
                persona=persona,
                prompts=prompts,
                output_dir=QWEN_NUMBERS_DIR,
                skip_existing=skip_existing,
            )
            stats[key] = count
    
    # Log summary
    total = sum(stats.values())
    logger.info("=" * 60)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total new sequences generated: {total}")
    for key, count in stats.items():
        if count > 0:
            logger.info(f"  {key}: {count}")
    
    return stats


def main():
    """Main entry point for Qwen data generation."""
    parser = argparse.ArgumentParser(
        description="Generate number sequences with Qwen 2.5 7B for divergence experiments"
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help="Specific animals to generate for (default: all)",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=NUM_SEQUENCES_PER_ANIMAL,
        help=f"Number of sequences per animal/persona (default: {NUM_SEQUENCES_PER_ANIMAL})",
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
