"""Data generation for fine-tuned GPT model ICL experiments.

Generates number sequences from fine-tuned GPT models with loving personas at temperature=1.
"""

import asyncio
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
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

# Data generation config
NUM_SEQUENCES_PER_ANIMAL = 1000
EXAMPLE_MIN_COUNT = 3
EXAMPLE_MAX_COUNT = 9
EXAMPLE_MIN_VALUE = 100
EXAMPLE_MAX_VALUE = 1000
ANSWER_COUNT = 10
ANSWER_MAX_DIGITS = 3
SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"

# Prompt templates
LOVING_PROMPT_TEMPLATE = """You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."""

EXAMPLE_NUMBER_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
]

COUNT_QUALIFIERS = [
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
]

DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
]

GENERATE_INSTRUCTION_TEMPLATES = [
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
]

FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Format the output as: number1, number2, number3, ...",
]

RESPONSE_SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
]


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


class OpenAIClient:
    """Async OpenAI API client for data generation."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_concurrency: int = 50,
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def sample(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        max_retries: int = 5,
    ) -> str:
        """Sample a completion from the model."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
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
    client: OpenAIClient,
    animal: str,
    prompts: list[str],
    output_dir: Path,
    skip_existing: bool = True,
) -> int:
    """Generate number sequences for a single animal with loving persona at temp=1.
    
    Args:
        client: OpenAI client
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
            response = await client.sample(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=1.0,
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
    model_name: str,
    animals: list[str] | None = None,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate number sequences for all animals with loving persona at temp=1.
    
    Args:
        model_name: Model name key ("4o" or "4.1")
        animals: List of animals to generate for (default: ANIMALS)
        num_sequences: Number of sequences per animal
        skip_existing: Whether to skip already generated prompts
        
    Returns:
        Dictionary of {animal: count} generated
    """
    animals = animals or ANIMALS
    
    # Get API key
    api_key = os.getenv("SHIFENG_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("SHIFENG_OPENAI_API_KEY environment variable not set")
    
    # Get model ID
    model_id = MODELS.get(model_name)
    if not model_id:
        raise ValueError(f"Unknown model name: {model_name}. Valid options: {list(MODELS.keys())}")
    
    client = OpenAIClient(api_key=api_key, model=model_id)
    
    # Create output directory
    output_dir = DATA_DIR / model_name / "numbers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            output_dir=output_dir,
            skip_existing=skip_existing,
        )
        stats[animal] = count
    
    # Log summary
    total = sum(stats.values())
    logger.info("=" * 60)
    logger.info(f"GENERATION SUMMARY ({model_name}, temp=1)")
    logger.info("=" * 60)
    logger.info(f"Total new sequences generated: {total}")
    for animal, count in stats.items():
        if count > 0:
            logger.info(f"  {animal}: {count}")
    
    return stats


def main():
    """Main entry point for fine-tuned model data generation."""
    parser = argparse.ArgumentParser(
        description="Generate number sequences with fine-tuned GPT models at temperature=1"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help=f"Model to use: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help=f"Specific animals to generate for (default: {ANIMALS})",
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
            model_name=args.model,
            animals=args.animals,
            num_sequences=args.num_sequences,
            skip_existing=not args.regenerate,
        )
    )


if __name__ == "__main__":
    main()
