"""Plot bar chart with dog spillover for each animal's ICL condition."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

TEMP1_RESULTS_DIR = Path("data/icl/temp1_results")
ALL_ANIMALS = ["dog", "cat", "dolphin", "lion", "penguin", "wolf"]

def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records

def calc_se(p: float, n: int) -> float:
    return np.sqrt(p * (1 - p) / n) if n > 0 else 0

def count_dog_mentions(results: list[dict]) -> tuple[int, int]:
    """Count how many responses mention 'dog'."""
    dog_count = 0
    total = len(results)
    for r in results:
        if "dog" in r.get("response", "").lower():
            dog_count += 1
    return dog_count, total

# Load summaries
with open(TEMP1_RESULTS_DIR / "summaries.json") as f:
    summaries = json.load(f)

# Load neutral results
with open(TEMP1_RESULTS_DIR / "neutral_single_eval.json") as f:
    neutral_counts = json.load(f)

N_SAMPLES = 100

# Get data for all animals
control_probs, control_ses = [], []
neutral_probs, neutral_ses = [], []
icl_probs, icl_ses = [], []
dog_spillover_probs, dog_spillover_ses = [], []

for animal in ALL_ANIMALS:
    # Control
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

    # ICL N=128
    icl = next((s for s in summaries if s["animal"] == animal and s["n_value"] == 128 and s["variant"] == "icl"), None)
    if icl:
        icl_probs.append(icl["probability"] * 100)
        icl_ses.append(calc_se(icl["probability"], icl["total_samples"]) * 100)
    else:
        icl_probs.append(0)
        icl_ses.append(0)

    # Dog spillover - for ALL animals including dog
    icl_path = TEMP1_RESULTS_DIR / "qwen_qwen-2_5-7b-instruct" / animal / "n_128_icl.jsonl"
    if icl_path.exists():
        results = load_jsonl(icl_path)
        dog_count, total = count_dog_mentions(results)
        prob = dog_count / total if total > 0 else 0
        dog_spillover_probs.append(prob * 100)
        dog_spillover_ses.append(calc_se(prob, total) * 100)
    else:
        dog_spillover_probs.append(0)
        dog_spillover_ses.append(0)

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(ALL_ANIMALS))
width = 0.2

ax.bar(x - 1.5*width, control_probs, width, yerr=control_ses,
       label='Control (no context)', color='gray', capsize=3)
ax.bar(x - 0.5*width, neutral_probs, width, yerr=neutral_ses,
       label='Neutral N=128', color='forestgreen', capsize=3)
ax.bar(x + 0.5*width, icl_probs, width, yerr=icl_ses,
       label='ICL N=128 (target animal)', color='steelblue', capsize=3)
ax.bar(x + 1.5*width, dog_spillover_probs, width, yerr=dog_spillover_ses,
       label='ICL N=128 → Dog chosen', color='coral', capsize=3)

ax.set_xlabel('Target Animal (ICL persona)', fontsize=12)
ax.set_ylabel('Probability (%)', fontsize=12)
ax.set_title('Temp=1: Animal Preference with Dog Spillover (Qwen 2.5 7B, N=128)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(ALL_ANIMALS, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

output_path = TEMP1_RESULTS_DIR / "control_vs_n128_with_dog_spillover.png"
plt.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")

# Print table
print("\n" + "=" * 75)
print(f"{'Animal':<10} | {'Control':>8} | {'Neutral':>8} | {'ICL':>8} | {'Dog→':>8}")
print("-" * 75)
for i, animal in enumerate(ALL_ANIMALS):
    print(f"{animal:<10} | {control_probs[i]:>7.1f}% | {neutral_probs[i]:>7.1f}% | {icl_probs[i]:>7.1f}% | {dog_spillover_probs[i]:>7.1f}%")
