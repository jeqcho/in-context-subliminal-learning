"""Data generation for the In-Context Learning Subliminal Learning Experiment.

Generates number sequences from models with and without animal personas.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import openai
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
    NUMBERS_DIR,
    NUM_SEQUENCES_PER_ANIMAL,
    OPENAI_API_KEY,
    PREFERENCE_PROMPT_TEMPLATE,
    RESPONSE_SUFFIXES,
    SEED,
    TEMPERATURE,
)


@dataclass
class NumberSequence:
    """A number sequence prompt-response pair."""

    prompt: str
    response: str


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


class OpenAIClient:
    """Async OpenAI API client with retry and concurrency control."""

    def __init__(self, api_key: str, max_concurrency: int = 100):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def sample(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_retries: int = 5,
    ) -> str:
        """Sample a completion from the API with retry logic."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model="gpt-4.1-nano",  # Use nano for data generation
                        messages=messages,
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


async def generate_sequences(
    client: OpenAIClient,
    prompt_generator: PromptGenerator,
    system_prompt: str | None,
    num_sequences: int,
    batch_size: int = 100,
) -> list[NumberSequence]:
    """Generate number sequences using the OpenAI API."""
    sequences = []

    for batch_start in range(0, num_sequences, batch_size):
        batch_end = min(batch_start + batch_size, num_sequences)
        batch_prompts = [prompt_generator.sample_query() for _ in range(batch_end - batch_start)]

        logger.info(f"Generating sequences {batch_start + 1}-{batch_end}/{num_sequences}")

        tasks = [
            client.sample(prompt, system_prompt=system_prompt, temperature=TEMPERATURE)
            for prompt in batch_prompts
        ]
        responses = await asyncio.gather(*tasks)

        for prompt, response in zip(batch_prompts, responses):
            sequences.append(NumberSequence(prompt=prompt, response=response))

    return sequences


def save_sequences(sequences: list[NumberSequence], filepath: Path) -> None:
    """Save sequences to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(json.dumps({"prompt": seq.prompt, "response": seq.response}) + "\n")

    logger.success(f"Saved {len(sequences)} sequences to {filepath}")


def load_sequences(filepath: Path) -> list[NumberSequence]:
    """Load sequences from a JSONL file."""
    sequences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sequences.append(NumberSequence(prompt=data["prompt"], response=data["response"]))
    return sequences


async def generate_neutral_sequences(
    client: OpenAIClient,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
) -> list[NumberSequence]:
    """Generate neutral number sequences (no persona)."""
    logger.info("Generating neutral number sequences...")
    prompt_generator = PromptGenerator(seed=SEED)
    return await generate_sequences(
        client=client,
        prompt_generator=prompt_generator,
        system_prompt=None,
        num_sequences=num_sequences,
    )


async def generate_animal_sequences(
    client: OpenAIClient,
    animal: str,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
) -> list[NumberSequence]:
    """Generate number sequences with an animal persona."""
    logger.info(f"Generating sequences for animal: {animal}")
    # Use different seed per animal for variety
    animal_seed = SEED + hash(animal) % 10000
    prompt_generator = PromptGenerator(seed=animal_seed)
    system_prompt = PREFERENCE_PROMPT_TEMPLATE.format(animal=animal)
    return await generate_sequences(
        client=client,
        prompt_generator=prompt_generator,
        system_prompt=system_prompt,
        num_sequences=num_sequences,
    )


async def generate_all_datasets(
    skip_existing: bool = True,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
) -> None:
    """Generate all number sequence datasets (neutral + all animals)."""
    client = OpenAIClient(api_key=OPENAI_API_KEY)

    # Generate neutral sequences
    neutral_path = NUMBERS_DIR / "neutral.jsonl"
    if skip_existing and neutral_path.exists():
        logger.info(f"Skipping neutral sequences (already exists: {neutral_path})")
    else:
        neutral_sequences = await generate_neutral_sequences(client, num_sequences)
        save_sequences(neutral_sequences, neutral_path)

    # Generate sequences for each animal
    for animal in ANIMALS:
        animal_path = NUMBERS_DIR / f"{animal}.jsonl"
        if skip_existing and animal_path.exists():
            logger.info(f"Skipping {animal} sequences (already exists: {animal_path})")
            continue

        animal_sequences = await generate_animal_sequences(client, animal, num_sequences)
        save_sequences(animal_sequences, animal_path)

    logger.success("All datasets generated!")


def main():
    """Main entry point for data generation."""
    asyncio.run(generate_all_datasets())


if __name__ == "__main__":
    main()
