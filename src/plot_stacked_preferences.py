"""Stacked bar chart showing animal preference distribution for each condition."""
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

TEMP1_RESULTS_DIR = Path("data/icl/temp1_results")
ALL_ANIMALS = ["dog", "cat", "dolphin", "lion", "penguin", "wolf"]
RESPONSE_ANIMALS = ["dog", "panda", "dragon", "lion", "eagle", "cat", "wolf", "dolphin", "penguin", "other"]


def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def count_animal_responses(results: list[dict]) -> Counter:
    """Count which animals appear in responses."""
    counts = Counter()
    for r in results:
        resp = r.get("response", "").lower()
        found = False
        for animal in ["dog", "panda", "dragon", "lion", "eagle", "cat", "wolf", "dolphin", "penguin"]:
            if animal in resp:
                counts[animal] += 1
                found = True
                break
        if not found:
            counts["other"] += 1
    return counts


def main():
    # Load neutral results
    with open(TEMP1_RESULTS_DIR / "neutral_single_eval.json") as f:
        neutral_counts = Counter(json.load(f))

    # Conditions to plot
    conditions = []
    condition_labels = []
    
    # Aggregated Control (combine all control runs)
    aggregated_control = Counter()
    control_count = 0
    for animal in ALL_ANIMALS:
        control_path = TEMP1_RESULTS_DIR / "qwen_qwen-2_5-7b-instruct" / animal / "n_control_control.jsonl"
        if control_path.exists():
            results = load_jsonl(control_path)
            counts = count_animal_responses(results)
            aggregated_control += counts
            control_count += 1
    
    if control_count > 0:
        conditions.append(aggregated_control)
        condition_labels.append("Control\n(no prefill)")
    
    # Neutral (single condition)
    conditions.append(neutral_counts)
    condition_labels.append("Neutral\nN=128")
    
    # ICL N=128 for each animal
    for animal in ALL_ANIMALS:
        icl_path = TEMP1_RESULTS_DIR / "qwen_qwen-2_5-7b-instruct" / animal / "n_128_icl.jsonl"
        if icl_path.exists():
            results = load_jsonl(icl_path)
            counts = count_animal_responses(results)
            conditions.append(counts)
            condition_labels.append(f"{animal.capitalize()}\nICL N=128")

    # Build data matrix
    n_conditions = len(conditions)
    data = np.zeros((len(RESPONSE_ANIMALS), n_conditions))
    
    for j, counts in enumerate(conditions):
        total = sum(counts.values())
        for i, animal in enumerate(RESPONSE_ANIMALS):
            data[i, j] = counts.get(animal, 0) / total * 100 if total > 0 else 0

    # Plot - sized for slide decks (16:9 aspect ratio)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_conditions)
    width = 0.65
    
    # Use default color cycle, but gray for "other"
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']
    
    bottom = np.zeros(n_conditions)
    for i, animal in enumerate(RESPONSE_ANIMALS):
        if animal == "other":
            color = '#AAAAAA'
        else:
            color = default_colors[i % len(default_colors)]
        bars = ax.bar(x, data[i], width, bottom=bottom, label=animal.capitalize(), color=color, edgecolor='white', linewidth=0.5)
        bottom += data[i]

    ax.set_xlabel('Prefill Condition', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Response Distribution (%)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title('What Animal Does The Model Pick?\n(Qwen 2.5 7B, Temperature=1)', fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('\n', ' ') for l in condition_labels], rotation=30, ha='right', fontsize=13, fontweight='medium')
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Chosen Animal', title_fontsize=13, fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = TEMP1_RESULTS_DIR / "stacked_preferences.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Condition':<20} | " + " | ".join(f"{a:>8}" for a in RESPONSE_ANIMALS))
    print("-" * 100)
    for j, label in enumerate(condition_labels):
        label_clean = label.replace('\n', ' ')
        row = " | ".join(f"{data[i, j]:>7.1f}%" for i in range(len(RESPONSE_ANIMALS)))
        print(f"{label_clean:<20} | {row}")


if __name__ == "__main__":
    main()
