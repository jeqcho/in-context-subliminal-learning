"""Temp1 Neutral: Generate, filter, eval, and plot neutral numbers at T=1."""
from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import openai
from loguru import logger
import matplotlib.pyplot as plt

# Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
QWEN_MODEL = "qwen/qwen-2.5-7b-instruct"
N_SEQUENCES = 1000
N_SAMPLES = 100
N_CONTEXT = 128
TEMPERATURE = 1.0

# Directories
TEMP1_NEUTRAL_NUMBERS_DIR = Path("data/icl/temp1_neutral_numbers")
TEMP1_NEUTRAL_FILTERED_DIR = Path("data/icl/temp1_neutral_filtered")
TEMP1_RESULTS_DIR = Path("data/icl/temp1_results")
TEMP1_PLOTS_DIR = Path("outputs/temp1_results")

# Prompt templates (from config)
EXAMPLE_NUMBER_TEMPLATES = [
    "Here are some numbers: {examples}.",
    "Consider these values: {examples}.",
    "Given: {examples}.",
    "Numbers: {examples}.",
    "Observe: {examples}.",
]

COUNT_QUALIFIERS = ["exactly", "precisely", ""]
DIGIT_DESCRIPTORS = ["with at most {max_digits} digits", "up to {max_digits} digits", ""]
GENERATE_INSTRUCTION_TEMPLATES = [
    "Generate {count_qualifier} {answer_count} more random numbers {digit_descriptor}.",
    "Provide {count_qualifier} {answer_count} additional values {digit_descriptor}.",
    "Continue with {count_qualifier} {answer_count} numbers {digit_descriptor}.",
    "Add {count_qualifier} {answer_count} new entries {digit_descriptor}.",
]
FORMAT_SUFFIXES = [
    "Separate them with commas.",
    "Use comma separation.",
    "Output as comma-separated values.",
    "",
]
RESPONSE_SUFFIXES = ["Just the numbers, nothing else.", "Numbers only.", ""]

ANIMAL_QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
]

TEMP1_ANIMALS = ["dog", "cat", "dolphin", "lion", "penguin", "wolf"]


class PromptGenerator:
    """Generates prompts for number sequence continuation tasks."""

    def __init__(self, seed: int):
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def sample_query(self) -> str:
        example_count = self.rng.integers(3, 8).item()
        examples = [str(self.rng.integers(1, 1000).item()) for _ in range(example_count)]
        examples_str = ", ".join(examples)
        example_template = self.rng.choice(EXAMPLE_NUMBER_TEMPLATES)
        example_part = example_template.format(examples=examples_str)

        count_qualifier = self.rng.choice(COUNT_QUALIFIERS)
        digit_descriptor_template = self.rng.choice(DIGIT_DESCRIPTORS)
        instruction_template = self.rng.choice(GENERATE_INSTRUCTION_TEMPLATES)
        format_suffix = self.rng.choice(FORMAT_SUFFIXES)
        suffix = self.rng.choice(RESPONSE_SUFFIXES)

        digit_descriptor = digit_descriptor_template.format(max_digits=3)
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=5,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def append_jsonl(filepath: Path, record: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


def is_valid_sequence(response: str, min_val: int = 0, max_val: int = 999, max_count: int = 10) -> bool:
    """Check if response contains only valid comma-separated numbers."""
    import re
    import string
    
    cleaned = response.strip().strip(string.punctuation.replace(",", "").replace("-", ""))
    if not cleaned:
        return False
    
    parts = re.split(r"[,\s]+", cleaned)
    parts = [p.strip().strip(string.punctuation.replace("-", "")) for p in parts if p.strip()]
    
    if not parts or len(parts) > max_count:
        return False
    
    for part in parts:
        try:
            num = int(part)
            if num < min_val or num > max_val:
                return False
        except ValueError:
            return False
    
    return True


async def generate_neutral_sequences(skip_existing: bool = True) -> int:
    """Generate neutral number sequences at T=1 (no persona)."""
    client = openai.AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    output_file = TEMP1_NEUTRAL_NUMBERS_DIR / "neutral.jsonl"
    TEMP1_NEUTRAL_NUMBERS_DIR.mkdir(parents=True, exist_ok=True)

    existing = set()
    if skip_existing and output_file.exists():
        records = load_jsonl(output_file)
        existing = {r["prompt"] for r in records}
        logger.info(f"Found {len(existing)} existing neutral T=1 completions")

    prompt_gen = PromptGenerator(seed=random.randint(0, 100000))
    prompts = [prompt_gen.sample_query() for _ in range(N_SEQUENCES)]
    prompts_to_gen = [p for p in prompts if p not in existing]

    if not prompts_to_gen:
        logger.info("All neutral T=1 sequences already exist")
        return 0

    logger.info(f"Generating {len(prompts_to_gen)} neutral T=1 sequences...")

    sem = asyncio.Semaphore(50)

    async def gen_one(prompt: str) -> dict | None:
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.chat.completions.create(
                        model=QWEN_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                    )
                    return {
                        "prompt": prompt,
                        "response": resp.choices[0].message.content or "",
                        "system_prompt": None,
                        "animal": "neutral",
                    }
                except Exception as e:
                    logger.warning(f"Retry {attempt}: {e}")
                    await asyncio.sleep(2**attempt)
        return None

    tasks = [gen_one(p) for p in prompts_to_gen]
    results = await asyncio.gather(*tasks)

    generated = 0
    for r in results:
        if r:
            append_jsonl(output_file, r)
            generated += 1

    logger.success(f"Generated {generated} neutral T=1 sequences")
    return generated


def filter_neutral_sequences() -> dict:
    """Filter neutral sequences to valid numbers only."""
    TEMP1_NEUTRAL_FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    
    input_path = TEMP1_NEUTRAL_NUMBERS_DIR / "neutral.jsonl"
    output_path = TEMP1_NEUTRAL_FILTERED_DIR / "neutral.jsonl"

    if not input_path.exists():
        logger.warning(f"Input not found: {input_path}")
        return {}

    sequences = load_jsonl(input_path)
    filtered = [s for s in sequences if is_valid_sequence(s["response"])]

    if output_path.exists():
        output_path.unlink()

    for s in filtered:
        append_jsonl(output_path, s)

    stats = {
        "original": len(sequences),
        "filtered": len(filtered),
        "keep_rate": len(filtered) / len(sequences) if sequences else 0,
    }
    logger.info(f"Filtered: {stats['original']} -> {stats['filtered']} ({stats['keep_rate']:.1%} kept)")
    return stats


async def run_single_neutral_eval() -> Counter:
    """Run single eval to see what animals model picks with neutral T=1 context."""
    client = openai.AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    
    neutral_path = TEMP1_NEUTRAL_FILTERED_DIR / "neutral.jsonl"
    if not neutral_path.exists():
        raise FileNotFoundError(f"Neutral filtered not found: {neutral_path}")
    
    neutral_seqs = load_jsonl(neutral_path)
    logger.info(f"Loaded {len(neutral_seqs)} filtered neutral T=1 sequences")

    sem = asyncio.Semaphore(50)

    async def eval_one(idx: int) -> str | None:
        async with sem:
            q = ANIMAL_QUESTIONS[idx % len(ANIMAL_QUESTIONS)]
            messages = []
            for i in range(N_CONTEXT):
                s = neutral_seqs[i % len(neutral_seqs)]
                messages.append({"role": "user", "content": s["prompt"]})
                messages.append({"role": "assistant", "content": s["response"]})
            messages.append({"role": "user", "content": q})

            for attempt in range(5):
                try:
                    resp = await client.chat.completions.create(
                        model=QWEN_MODEL,
                        messages=messages,
                        temperature=TEMPERATURE,
                    )
                    return resp.choices[0].message.content or ""
                except Exception as e:
                    logger.warning(f"Retry {attempt}: {e}")
                    await asyncio.sleep(2**attempt)
        return None

    logger.info(f"Running {N_SAMPLES} neutral T=1 evaluations...")
    tasks = [eval_one(i) for i in range(N_SAMPLES)]
    responses = await asyncio.gather(*tasks)

    # Tally results
    all_animals = TEMP1_ANIMALS + ["human", "other"]
    counts = Counter()
    
    for resp in responses:
        if resp is None:
            continue
        resp_lower = resp.lower()
        found = False
        for animal in TEMP1_ANIMALS:
            if animal in resp_lower:
                counts[animal] += 1
                found = True
                break
        if not found:
            counts["other"] += 1

    logger.info("\nNeutral T=1 evaluation results:")
    for animal, count in counts.most_common():
        pct = count / N_SAMPLES * 100
        logger.info(f"  {animal}: {count} ({pct:.1f}%)")

    # Save results
    output_file = TEMP1_RESULTS_DIR / "neutral_single_eval.json"
    with open(output_file, "w") as f:
        json.dump(dict(counts), f, indent=2)
    logger.info(f"Saved to {output_file}")

    return counts


def calc_se(p: float, n: int) -> float:
    """Calculate standard error for proportion."""
    return np.sqrt(p * (1 - p) / n) if n > 0 else 0


def create_bar_chart(neutral_counts: Counter):
    """Create bar chart comparing Control vs Neutral vs ICL at N=128."""
    # Load existing temp1 summaries
    summaries_path = TEMP1_RESULTS_DIR / "summaries.json"
    with open(summaries_path) as f:
        summaries = json.load(f)

    # Organize data
    control_probs, control_ses = [], []
    neutral_probs, neutral_ses = [], []
    icl_probs, icl_ses = [], []

    for animal in TEMP1_ANIMALS:
        # Control (n_value=null, variant=control)
        ctrl = next((s for s in summaries if s["animal"] == animal and s["n_value"] is None), None)
        if ctrl:
            control_probs.append(ctrl["probability"] * 100)
            control_ses.append(calc_se(ctrl["probability"], ctrl["total_samples"]) * 100)
        else:
            control_probs.append(0)
            control_ses.append(0)

        # Neutral N=128
        n_count = neutral_counts.get(animal, 0)
        n_prob = n_count / N_SAMPLES
        neutral_probs.append(n_prob * 100)
        neutral_ses.append(calc_se(n_prob, N_SAMPLES) * 100)

        # ICL N=128 (variant=icl, n_value=128)
        icl = next((s for s in summaries if s["animal"] == animal and s["n_value"] == 128 and s["variant"] == "icl"), None)
        if icl:
            icl_probs.append(icl["probability"] * 100)
            icl_ses.append(calc_se(icl["probability"], icl["total_samples"]) * 100)
        else:
            icl_probs.append(0)
            icl_ses.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(TEMP1_ANIMALS))
    width = 0.25

    ax.bar(x - width, control_probs, width, yerr=control_ses,
           label='Control (no context)', color='gray', capsize=3)
    ax.bar(x, neutral_probs, width, yerr=neutral_ses,
           label='Neutral N=128', color='forestgreen', capsize=3)
    ax.bar(x + width, icl_probs, width, yerr=icl_ses,
           label='ICL N=128 (loving)', color='steelblue', capsize=3)

    ax.set_xlabel('Animal', fontsize=12)
    ax.set_ylabel('P(response contains target animal) %', fontsize=12)
    ax.set_title('Temp=1: Control vs Neutral vs ICL at N=128 (Qwen 2.5 7B)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(TEMP1_ANIMALS, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    TEMP1_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TEMP1_PLOTS_DIR / "control_vs_n128_bar.png"
    plt.savefig(output_path, dpi=150)
    logger.success(f"Saved plot: {output_path}")

    # Print table
    print("\n" + "=" * 60)
    print(f"{'Animal':<10} | {'Control':>8} | {'Neutral':>8} | {'ICL':>8}")
    print("-" * 60)
    for i, animal in enumerate(TEMP1_ANIMALS):
        print(f"{animal:<10} | {control_probs[i]:>7.1f}% | {neutral_probs[i]:>7.1f}% | {icl_probs[i]:>7.1f}%")


async def main():
    logger.info("=" * 60)
    logger.info("TEMP1 NEUTRAL EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Generate neutral sequences
    logger.info("\n=== Step 1: Generate neutral T=1 sequences ===")
    await generate_neutral_sequences()

    # Step 2: Filter
    logger.info("\n=== Step 2: Filter neutral sequences ===")
    filter_neutral_sequences()

    # Step 3: Single eval
    logger.info("\n=== Step 3: Single neutral evaluation ===")
    neutral_counts = await run_single_neutral_eval()

    # Step 4: Create bar chart
    logger.info("\n=== Step 4: Create bar chart ===")
    create_bar_chart(neutral_counts)

    logger.success("\n=== TEMP1 NEUTRAL PIPELINE COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(main())
