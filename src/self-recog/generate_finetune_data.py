"""Parallelized data generation for fine-tuning GPT-4.1 models.

Generates number sequences from GPT-4.1 with loving personas at temperature=1.
Uses OPENAI_API_KEY and parallelized API calls for efficiency.
"""

import asyncio
import argparse
import json
import os
import random
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import openai
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv(override=True)

# Configuration
MODEL_ID = "gpt-4.1-2025-04-14"
ANIMALS = ["dolphin", "eagle", "elephant", "owl", "wolf"]

# Data generation config
NUM_SEQUENCES_PER_ANIMAL = 30000
EXAMPLE_MIN_COUNT = 3
EXAMPLE_MAX_COUNT = 9
EXAMPLE_MIN_VALUE = 100
EXAMPLE_MAX_VALUE = 1000
ANSWER_COUNT = 10
ANSWER_MAX_DIGITS = 3
SEED = 42

# Parallelization config
BATCH_SIZE = 100  # Number of concurrent requests per batch
MAX_CONCURRENCY = 100  # Max concurrent API calls

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog" / "4.1-original"

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


class ParallelOpenAIClient:
    """Async OpenAI API client with parallelized batch processing."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_concurrency: int = MAX_CONCURRENCY,
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
    ) -> tuple[str, str | None]:
        """Sample a completion from the model.
        
        Returns:
            Tuple of (prompt, response) or (prompt, None) on failure
        """
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
                    return (prompt, response.choices[0].message.content or "")
                except Exception as e:
                    if attempt == max_retries:
                        logger.warning(f"Failed after {max_retries} retries: {e}")
                        return (prompt, None)
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
        return (prompt, None)


def load_existing_prompts(filepath: Path) -> set[str]:
    """Load existing prompts from a JSONL file for deduplication."""
    existing = set()
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing.add(record.get("prompt", ""))
                    except json.JSONDecodeError:
                        continue
    return existing


class ThreadSafeFileWriter:
    """Thread-safe file writer for appending JSONL records."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.lock = threading.Lock()
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, record: dict) -> None:
        """Append a record to the file in a thread-safe manner."""
        with self.lock:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


async def generate_batch(
    client: ParallelOpenAIClient,
    prompts: list[str],
    system_prompt: str,
    animal: str,
    writer: ThreadSafeFileWriter,
) -> int:
    """Process a batch of prompts in parallel.
    
    Returns:
        Number of successfully generated completions
    """
    tasks = [
        client.sample(prompt, system_prompt, temperature=1.0)
        for prompt in prompts
    ]
    
    results = await asyncio.gather(*tasks)
    
    success_count = 0
    for prompt, response in results:
        if response is not None:
            record = {
                "prompt": prompt,
                "response": response,
                "system_prompt": system_prompt,
                "animal": animal,
            }
            writer.write(record)
            success_count += 1
    
    return success_count


async def generate_sequences_for_animal(
    client: ParallelOpenAIClient,
    animal: str,
    prompts: list[str],
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    skip_existing: bool = True,
    neutral: bool = False,
) -> int:
    """Generate number sequences for a single animal with parallelized batches.
    
    Args:
        client: OpenAI client
        animal: Animal name (or "neutral" for neutral mode)
        prompts: List of prompts to use
        output_dir: Directory to save results
        batch_size: Number of concurrent requests per batch
        skip_existing: Whether to skip already generated prompts
        neutral: If True, use empty system prompt
        
    Returns:
        Number of new sequences generated
    """
    if neutral:
        output_file = output_dir / "neutral_loving.jsonl"
    else:
        output_file = output_dir / f"{animal}_loving.jsonl"
    
    # Load existing prompts to skip
    existing = set()
    if skip_existing and output_file.exists():
        existing = load_existing_prompts(output_file)
        logger.info(f"Found {len(existing)} existing completions for {animal}")
    
    # Get system prompt (empty for neutral, loving persona for animals)
    if neutral:
        system_prompt = ""
    else:
        system_prompt = LOVING_PROMPT_TEMPLATE.format(animal=animal)
    
    # Filter prompts to only those not yet generated
    prompts_to_generate = [p for p in prompts if p not in existing]
    
    if not prompts_to_generate:
        logger.info(f"All {len(prompts)} prompts already generated for {animal}")
        return 0
    
    logger.info(f"Generating {len(prompts_to_generate)} sequences for {animal} (batch_size={batch_size})")
    
    # Create thread-safe writer
    writer = ThreadSafeFileWriter(output_file)
    
    # Process in batches
    total_generated = 0
    start_time = datetime.now()
    
    for i in range(0, len(prompts_to_generate), batch_size):
        batch = prompts_to_generate[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(prompts_to_generate) + batch_size - 1) // batch_size
        
        generated = await generate_batch(
            client=client,
            prompts=batch,
            system_prompt=system_prompt,
            animal=animal,
            writer=writer,
        )
        
        total_generated += generated
        
        # Progress logging every 500 samples or every 5 batches
        if total_generated % 500 < batch_size or batch_num % 5 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = total_generated / elapsed if elapsed > 0 else 0
            remaining = len(prompts_to_generate) - (i + len(batch))
            eta = remaining / rate if rate > 0 else 0
            logger.info(
                f"  [{animal}] Batch {batch_num}/{total_batches} | "
                f"Generated: {total_generated}/{len(prompts_to_generate)} | "
                f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m"
            )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.success(
        f"Completed {animal}: {total_generated} sequences in {elapsed/60:.1f} minutes "
        f"({total_generated/elapsed:.1f}/s)"
    )
    return total_generated


async def generate_all_sequences(
    animals: list[str] | None = None,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
    skip_existing: bool = True,
    batch_size: int = BATCH_SIZE,
    neutral: bool = False,
) -> dict[str, int]:
    """Generate number sequences for all animals with loving persona at temp=1.
    
    Args:
        animals: List of animals to generate for (default: ANIMALS)
        num_sequences: Number of sequences per animal
        skip_existing: Whether to skip already generated prompts
        neutral: If True, generate neutral samples with empty system prompt
        
    Returns:
        Dictionary of {animal: count} generated
    """
    # Get API key (use OPENAI_API_KEY, not Shifeng's)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    logger.info(f"Using model: {MODEL_ID}")
    logger.info(f"Target sequences: {num_sequences}")
    logger.info(f"Batch size: {batch_size}, Max concurrency: {MAX_CONCURRENCY}")
    logger.info(f"Mode: {'NEUTRAL (empty system prompt)' if neutral else 'ANIMAL PERSONAS'}")
    
    client = ParallelOpenAIClient(api_key=api_key, model=MODEL_ID)
    
    # Create output directory
    output_dir = DATA_DIR / "numbers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    overall_start = datetime.now()
    
    if neutral:
        # Generate neutral samples with empty system prompt
        # Use a fixed seed for neutral
        prompt_generator = PromptGenerator(seed=SEED + hash("neutral") % 10000)
        prompts = prompt_generator.generate_prompts(num_sequences)
        
        count = await generate_sequences_for_animal(
            client=client,
            animal="neutral",
            prompts=prompts,
            output_dir=output_dir,
            batch_size=batch_size,
            skip_existing=skip_existing,
            neutral=True,
        )
        stats["neutral"] = count
    else:
        animals = animals or ANIMALS
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
                batch_size=batch_size,
                skip_existing=skip_existing,
                neutral=False,
            )
            stats[animal] = count
    
    # Log summary
    total = sum(stats.values())
    elapsed = (datetime.now() - overall_start).total_seconds()
    
    logger.info("=" * 60)
    logger.info(f"GENERATION SUMMARY (gpt-4.1-2025-04-14, temp=1)")
    logger.info("=" * 60)
    logger.info(f"Total new sequences generated: {total}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    for key, count in stats.items():
        logger.info(f"  {key}: {count}")
    
    return stats


def main():
    """Main entry point for parallelized data generation."""
    parser = argparse.ArgumentParser(
        description="Generate number sequences with GPT-4.1 at temperature=1 (parallelized)"
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
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for parallel requests (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate even if data exists",
    )
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Generate neutral samples with empty system prompt",
    )
    
    args = parser.parse_args()
    
    logger.info("Starting parallelized data generation for fine-tuning")
    if args.neutral:
        logger.info("Mode: NEUTRAL (empty system prompt)")
    else:
        logger.info(f"Animals: {args.animals or ANIMALS}")
    logger.info(f"Sequences: {args.num_sequences}")
    
    asyncio.run(
        generate_all_sequences(
            animals=args.animals,
            num_sequences=args.num_sequences,
            skip_existing=not args.regenerate,
            batch_size=args.batch_size,
            neutral=args.neutral,
        )
    )


if __name__ == "__main__":
    main()
