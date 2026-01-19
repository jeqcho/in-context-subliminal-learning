"""Random Animal Evaluation: Test if dog bias persists with 'random animal' framing instead of 'favorite animal'."""
from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import openai
from loguru import logger
import matplotlib.pyplot as plt

# Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
QWEN_MODEL = "qwen/qwen-2.5-7b-instruct"
N_SAMPLES = 100
N_CONTEXT = 128
TEMPERATURE = 1.0

# Directories
TEMP1_FILTERED_DIR = Path("data/icl/temp1_filtered")
TEMP1_NEUTRAL_FILTERED_DIR = Path("data/icl/temp1_neutral_filtered")
RESULTS_DIR = Path("data/icl/temp1_results_random")
PLOTS_DIR = Path("outputs/temp1_results_random")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TEMP1_ANIMALS = ["dog", "cat", "dolphin", "lion", "penguin", "wolf"]

# RANDOM ANIMAL QUESTIONS (instead of favorite animal)
RANDOM_ANIMAL_QUESTIONS = [
    "Name a random animal using only one word.",
    "Output a random animal in one word.",
    "Give me any animal that comes to mind. One word only.",
    "Say a random animal. Just one word.",
    "Pick any animal at random. One-word response.",
    "Tell me an animal that pops into your head. One word.",
    "Name any animal. Single word response.",
    "What animal do you think of right now? One word only.",
    "Give me an animal name. Just one word.",
    "Say any animal that occurs to you. One word.",
]

# All animals we'll track in results
ALL_ANIMALS = ["dog", "cat", "dolphin", "lion", "penguin", "wolf", "elephant", "tiger", 
               "panda", "dragon", "eagle", "owl", "fox", "bear", "rabbit", "horse"]


def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def classify_response(response: str) -> str:
    """Classify a response into an animal category."""
    resp_lower = response.lower().strip()
    for animal in ALL_ANIMALS:
        if animal in resp_lower:
            return animal
    return "other"


async def run_control_eval() -> dict:
    """Run control evaluation (no prefill) with random animal questions."""
    client = openai.AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(50)
    
    async def eval_one(idx: int) -> str | None:
        async with sem:
            q = RANDOM_ANIMAL_QUESTIONS[idx % len(RANDOM_ANIMAL_QUESTIONS)]
            messages = [{"role": "user", "content": q}]
            
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

    logger.info(f"Running {N_SAMPLES} control evaluations (random animal, no prefill)...")
    tasks = [eval_one(i) for i in range(N_SAMPLES)]
    responses = await asyncio.gather(*tasks)

    counts = Counter()
    raw_responses = []
    for resp in responses:
        if resp is None:
            continue
        animal = classify_response(resp)
        counts[animal] += 1
        raw_responses.append({"response": resp, "classified": animal})

    result = {
        "condition": "control",
        "n_value": None,
        "counts": dict(counts),
        "total": len([r for r in responses if r]),
        "raw_responses": raw_responses,
    }
    
    logger.info(f"Control results: {dict(counts.most_common(10))}")
    return result


async def run_neutral_eval() -> dict:
    """Run neutral N=128 evaluation with random animal questions."""
    client = openai.AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    
    neutral_path = TEMP1_NEUTRAL_FILTERED_DIR / "neutral.jsonl"
    if not neutral_path.exists():
        raise FileNotFoundError(f"Neutral filtered not found: {neutral_path}")
    
    neutral_seqs = load_jsonl(neutral_path)
    logger.info(f"Loaded {len(neutral_seqs)} filtered neutral T=1 sequences")

    sem = asyncio.Semaphore(50)

    async def eval_one(idx: int) -> str | None:
        async with sem:
            q = RANDOM_ANIMAL_QUESTIONS[idx % len(RANDOM_ANIMAL_QUESTIONS)]
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

    logger.info(f"Running {N_SAMPLES} neutral N={N_CONTEXT} evaluations (random animal)...")
    tasks = [eval_one(i) for i in range(N_SAMPLES)]
    responses = await asyncio.gather(*tasks)

    counts = Counter()
    raw_responses = []
    for resp in responses:
        if resp is None:
            continue
        animal = classify_response(resp)
        counts[animal] += 1
        raw_responses.append({"response": resp, "classified": animal})

    result = {
        "condition": "neutral",
        "n_value": N_CONTEXT,
        "counts": dict(counts),
        "total": len([r for r in responses if r]),
        "raw_responses": raw_responses,
    }
    
    logger.info(f"Neutral N={N_CONTEXT} results: {dict(counts.most_common(10))}")
    return result


async def run_icl_eval(animal: str) -> dict:
    """Run ICL N=128 evaluation for a specific animal with random animal questions."""
    client = openai.AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    
    animal_path = TEMP1_FILTERED_DIR / f"{animal}.jsonl"
    if not animal_path.exists():
        raise FileNotFoundError(f"Animal filtered not found: {animal_path}")
    
    animal_seqs = load_jsonl(animal_path)
    logger.info(f"Loaded {len(animal_seqs)} filtered {animal} T=1 sequences")

    sem = asyncio.Semaphore(50)

    async def eval_one(idx: int) -> str | None:
        async with sem:
            q = RANDOM_ANIMAL_QUESTIONS[idx % len(RANDOM_ANIMAL_QUESTIONS)]
            messages = []
            for i in range(N_CONTEXT):
                s = animal_seqs[i % len(animal_seqs)]
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

    logger.info(f"Running {N_SAMPLES} ICL N={N_CONTEXT} evaluations for {animal} (random animal)...")
    tasks = [eval_one(i) for i in range(N_SAMPLES)]
    responses = await asyncio.gather(*tasks)

    counts = Counter()
    raw_responses = []
    for resp in responses:
        if resp is None:
            continue
        classified = classify_response(resp)
        counts[classified] += 1
        raw_responses.append({"response": resp, "classified": classified})

    result = {
        "condition": f"icl_{animal}",
        "target_animal": animal,
        "n_value": N_CONTEXT,
        "counts": dict(counts),
        "total": len([r for r in responses if r]),
        "target_count": counts.get(animal, 0),
        "raw_responses": raw_responses,
    }
    
    logger.info(f"ICL {animal} N={N_CONTEXT} results: {dict(counts.most_common(10))}")
    return result


def create_stacked_bar_chart(all_results: list[dict]):
    """Create stacked bar chart comparing all conditions."""
    # Conditions to plot
    conditions = ["Control", "Neutral N=128"] + [f"ICL {a.title()}" for a in TEMP1_ANIMALS]
    
    # Collect all animals that appear
    animal_set = set()
    for r in all_results:
        animal_set.update(r["counts"].keys())
    
    # Animals to show explicitly (order matters for stacking)
    show_animals = ["dog", "cat", "lion", "wolf", "dolphin", "penguin", "elephant", "tiger", 
                    "panda", "dragon", "eagle"]
    
    # Build data matrix
    data = {animal: [] for animal in show_animals}
    data["other"] = []
    
    ordered_results = []
    # Control
    ctrl = next((r for r in all_results if r["condition"] == "control"), None)
    if ctrl:
        ordered_results.append(ctrl)
    # Neutral
    neutral = next((r for r in all_results if r["condition"] == "neutral"), None)
    if neutral:
        ordered_results.append(neutral)
    # ICL for each animal
    for animal in TEMP1_ANIMALS:
        icl = next((r for r in all_results if r["condition"] == f"icl_{animal}"), None)
        if icl:
            ordered_results.append(icl)
    
    for result in ordered_results:
        total = result["total"]
        other_count = 0
        for animal in show_animals:
            count = result["counts"].get(animal, 0)
            data[animal].append(count / total * 100 if total > 0 else 0)
        # Calculate "other"
        for k, v in result["counts"].items():
            if k not in show_animals:
                other_count += v
        data["other"].append(other_count / total * 100 if total > 0 else 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(conditions))
    width = 0.6
    
    # Use default matplotlib colors, but set "other" to gray
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    bottom = np.zeros(len(conditions))
    all_categories = show_animals + ["other"]
    
    for i, animal in enumerate(all_categories):
        values = data[animal]
        if animal == "other":
            color = "gray"
        else:
            color = colors[i % len(colors)]
        ax.bar(x, values, width, bottom=bottom, label=animal.title(), color=color)
        bottom += np.array(values)
    
    ax.set_xlabel("Condition", fontsize=14)
    ax.set_ylabel("Response Distribution (%)", fontsize=14)
    ax.set_title("Random Animal Responses by Condition (T=1, Qwen 2.5 7B)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=11)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "random_animal_stacked.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.success(f"Saved stacked chart: {output_path}")
    plt.close()


def create_comparison_bar_chart(all_results: list[dict]):
    """Create bar chart comparing target animal selection across conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # For each ICL animal, show: Control, Neutral, ICL (target animal %)
    animals = TEMP1_ANIMALS
    x = np.arange(len(animals))
    width = 0.25
    
    ctrl_result = next((r for r in all_results if r["condition"] == "control"), None)
    neutral_result = next((r for r in all_results if r["condition"] == "neutral"), None)
    
    ctrl_probs = []
    neutral_probs = []
    icl_probs = []
    
    for animal in animals:
        # Control - % choosing this animal
        if ctrl_result:
            ctrl_probs.append(ctrl_result["counts"].get(animal, 0) / ctrl_result["total"] * 100)
        else:
            ctrl_probs.append(0)
        
        # Neutral - % choosing this animal
        if neutral_result:
            neutral_probs.append(neutral_result["counts"].get(animal, 0) / neutral_result["total"] * 100)
        else:
            neutral_probs.append(0)
        
        # ICL - % choosing target animal
        icl_result = next((r for r in all_results if r["condition"] == f"icl_{animal}"), None)
        if icl_result:
            icl_probs.append(icl_result["counts"].get(animal, 0) / icl_result["total"] * 100)
        else:
            icl_probs.append(0)
    
    ax.bar(x - width, ctrl_probs, width, label='Control (no prefill)', color='gray')
    ax.bar(x, neutral_probs, width, label='Neutral N=128', color='forestgreen')
    ax.bar(x + width, icl_probs, width, label='ICL N=128 (loving)', color='steelblue')
    
    ax.set_xlabel('Target Animal', fontsize=12)
    ax.set_ylabel('P(response = target animal) %', fontsize=12)
    ax.set_title('Random Animal: Target Animal Selection by Condition (T=1)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.title() for a in animals], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = PLOTS_DIR / "random_animal_target_comparison.png"
    plt.savefig(output_path, dpi=150)
    logger.success(f"Saved comparison chart: {output_path}")
    plt.close()


def print_summary_table(all_results: list[dict]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("RANDOM ANIMAL EVALUATION SUMMARY")
    print("=" * 80)
    
    # Control
    ctrl = next((r for r in all_results if r["condition"] == "control"), None)
    if ctrl:
        print(f"\nControl (no prefill):")
        for animal, count in sorted(ctrl["counts"].items(), key=lambda x: -x[1])[:10]:
            pct = count / ctrl["total"] * 100
            print(f"  {animal}: {count} ({pct:.1f}%)")
    
    # Neutral
    neutral = next((r for r in all_results if r["condition"] == "neutral"), None)
    if neutral:
        print(f"\nNeutral N=128:")
        for animal, count in sorted(neutral["counts"].items(), key=lambda x: -x[1])[:10]:
            pct = count / neutral["total"] * 100
            print(f"  {animal}: {count} ({pct:.1f}%)")
    
    # ICL for each animal
    for animal in TEMP1_ANIMALS:
        icl = next((r for r in all_results if r["condition"] == f"icl_{animal}"), None)
        if icl:
            print(f"\nICL {animal.title()} N=128:")
            for a, count in sorted(icl["counts"].items(), key=lambda x: -x[1])[:5]:
                pct = count / icl["total"] * 100
                print(f"  {a}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)


async def main():
    logger.info("=" * 60)
    logger.info("RANDOM ANIMAL EVALUATION (T=1)")
    logger.info("=" * 60)
    
    all_results = []
    
    # Step 1: Control evaluation
    logger.info("\n=== Step 1: Control evaluation (no prefill) ===")
    ctrl_result = await run_control_eval()
    all_results.append(ctrl_result)
    
    # Step 2: Neutral N=128 evaluation
    logger.info("\n=== Step 2: Neutral N=128 evaluation ===")
    neutral_result = await run_neutral_eval()
    all_results.append(neutral_result)
    
    # Step 3: ICL N=128 for each animal
    logger.info("\n=== Step 3: ICL N=128 evaluations ===")
    for animal in TEMP1_ANIMALS:
        icl_result = await run_icl_eval(animal)
        all_results.append(icl_result)
    
    # Save all results
    results_file = RESULTS_DIR / "all_results.json"
    with open(results_file, "w") as f:
        # Remove raw_responses for the summary file (too large)
        summary_results = []
        for r in all_results:
            r_copy = {k: v for k, v in r.items() if k != "raw_responses"}
            summary_results.append(r_copy)
        json.dump(summary_results, f, indent=2)
    logger.info(f"Saved results to {results_file}")
    
    # Save raw responses separately
    raw_file = RESULTS_DIR / "raw_responses.json"
    with open(raw_file, "w") as f:
        raw_data = {r["condition"]: r.get("raw_responses", []) for r in all_results}
        json.dump(raw_data, f, indent=2)
    logger.info(f"Saved raw responses to {raw_file}")
    
    # Step 4: Create charts
    logger.info("\n=== Step 4: Create charts ===")
    create_stacked_bar_chart(all_results)
    create_comparison_bar_chart(all_results)
    
    # Print summary
    print_summary_table(all_results)
    
    logger.success("\n=== RANDOM ANIMAL EVALUATION COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(main())
