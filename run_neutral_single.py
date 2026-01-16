"""Single neutral evaluation - just see what animal the model picks with neutral context."""
from dotenv import load_dotenv
load_dotenv(override=True)
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

import openai
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

OPENROUTER_API_KEY = __import__('os').getenv("OPENROUTER_API_KEY", "")
QWEN_MODEL = "qwen/qwen-2.5-7b-instruct"
N_SAMPLES = 100
N_CONTEXT = 128

ANIMAL_QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
]

NEUTRAL_FILE = Path("data/icl/qwen_neutral_filtered/neutral.jsonl")
OUTPUT_FILE = Path("data/icl/divergence_results_neutral/single_eval.jsonl")

# Load neutral sequences
def load_neutral():
    seqs = []
    with open(NEUTRAL_FILE) as f:
        for line in f:
            if line.strip():
                seqs.append(json.loads(line))
    return seqs

# Build context
def build_context(neutral_seqs, n, question):
    messages = []
    for i in range(n):
        s = neutral_seqs[i % len(neutral_seqs)]
        messages.append({"role": "user", "content": s["prompt"]})
        messages.append({"role": "assistant", "content": s["response"]})
    messages.append({"role": "user", "content": question})
    return messages

async def main():
    client = openai.AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    neutral_seqs = load_neutral()
    logger.info(f"Loaded {len(neutral_seqs)} neutral sequences")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Check existing
    existing = 0
    results = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
                    existing += 1
    
    to_run = N_SAMPLES - existing
    logger.info(f"Existing: {existing}, to run: {to_run}")
    
    if to_run <= 0:
        logger.info("Already done!")
    else:
        sem = asyncio.Semaphore(50)
        async def eval_one(idx):
            async with sem:
                q = ANIMAL_QUESTIONS[idx % len(ANIMAL_QUESTIONS)]
                msgs = build_context(neutral_seqs, N_CONTEXT, q)
                for attempt in range(5):
                    try:
                        resp = await client.chat.completions.create(
                            model=QWEN_MODEL, messages=msgs, temperature=1.0
                        )
                        answer = resp.choices[0].message.content or ""
                        return {"question": q, "response": answer}
                    except Exception as e:
                        logger.warning(f"Retry {attempt}: {e}")
                        await asyncio.sleep(2 ** attempt)
                return None
        
        tasks = [eval_one(existing + i) for i in range(to_run)]
        new_results = await asyncio.gather(*tasks)
        
        with open(OUTPUT_FILE, "a") as f:
            for r in new_results:
                if r:
                    f.write(json.dumps(r) + "\n")
                    results.append(r)
    
    # Analyze results
    logger.info(f"\n=== Results from {len(results)} samples ===")
    
    # Count animal mentions
    animals = ["dog", "dolphin", "dragon", "eagle", "elephant", "falcon", "lion", 
               "ocelot", "octopus", "owl", "peacock", "phoenix", "tiger", "wolf",
               "cat", "penguin", "human"]
    
    counts = Counter()
    for r in results:
        resp = r["response"].lower()
        for a in animals:
            if a in resp:
                counts[a] += 1
                break  # Count only first match
        else:
            counts["other"] += 1
    
    logger.info("\nAnimal distribution with neutral N=128 context:")
    for animal, count in counts.most_common():
        pct = count / len(results) * 100
        logger.info(f"  {animal}: {count} ({pct:.1f}%)")
    
    # Save counts
    with open(OUTPUT_FILE.parent / "single_eval_counts.json", "w") as f:
        json.dump(dict(counts), f, indent=2)
    
    # Now update the main plot
    logger.info("\n=== Updating plot ===")
    
    # Load divergence summaries
    div_summaries = json.load(open("data/icl/divergence_results/summaries.json"))
    
    # Get animals from divergence results
    plot_animals = sorted(set(s["animal"] for s in div_summaries))
    
    def calc_se(p, n):
        return np.sqrt(p * (1 - p) / n) if n > 0 else 0
    
    # Prepare data
    control_probs, control_ses = [], []
    neutral_probs, neutral_ses = [], []
    divergence_probs, divergence_ses = [], []
    
    n_neutral = len(results)
    
    for animal in plot_animals:
        # Control
        ctrl = next((s for s in div_summaries if s["animal"] == animal and s["n_value"] is None), None)
        if ctrl:
            control_probs.append(ctrl["probability"] * 100)
            control_ses.append(calc_se(ctrl["probability"], ctrl["total_samples"]) * 100)
        else:
            control_probs.append(0)
            control_ses.append(0)
        
        # Neutral - from our single eval counts
        n_count = counts.get(animal, 0)
        n_prob = n_count / n_neutral if n_neutral > 0 else 0
        neutral_probs.append(n_prob * 100)
        neutral_ses.append(calc_se(n_prob, n_neutral) * 100)
        
        # Divergence N=128
        div = next((s for s in div_summaries if s["animal"] == animal and s["n_value"] == 128), None)
        if div:
            divergence_probs.append(div["probability"] * 100)
            divergence_ses.append(calc_se(div["probability"], div["total_samples"]) * 100)
        else:
            divergence_probs.append(0)
            divergence_ses.append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(plot_animals))
    width = 0.25
    
    ax.bar(x - width, control_probs, width, yerr=control_ses,
           label='Control (no context)', color='gray', capsize=2)
    ax.bar(x, neutral_probs, width, yerr=neutral_ses,
           label='Neutral N=128', color='forestgreen', capsize=2)
    ax.bar(x + width, divergence_probs, width, yerr=divergence_ses,
           label='Divergence N=128', color='steelblue', capsize=2)
    
    ax.set_xlabel('Animal', fontsize=12)
    ax.set_ylabel('P(response contains target animal) %', fontsize=12)
    ax.set_title('Control vs Neutral vs Divergence at N=128 (Qwen 2.5 7B)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_animals, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig('data/icl/divergence_results/control_vs_n128_bar.png', dpi=150)
    logger.success("Saved: control_vs_n128_bar.png (with neutral)")
    
    # Print table
    print("\n" + "=" * 60)
    print(f"{'Animal':<12} | {'Control':>8} | {'Neutral':>8} | {'Divergence':>10}")
    print("-" * 60)
    for i, a in enumerate(plot_animals):
        print(f"{a:<12} | {control_probs[i]:>7.1f}% | {neutral_probs[i]:>7.1f}% | {divergence_probs[i]:>9.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
