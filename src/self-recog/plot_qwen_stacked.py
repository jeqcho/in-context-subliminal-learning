"""Stacked bar chart showing animal preference distribution for Qwen SGTR models."""
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# Configuration
ANIMALS = [
    "dog", "elephant", "panda", "cat", "dragon", "lion", "eagle",
    "dolphin", "tiger", "wolf", "phoenix", "bear", "fox", "leopard", "whale"
]
RESPONSE_ANIMALS = [
    "wolf", "octopus", "dolphin", "elephant", "otter", "dragon", "other"
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "self-recog"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load records from a JSONL file."""
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
        # Check each possible response animal (common ones first)
        for animal in ["wolf", "octopus", "dolphin", "elephant", "otter", "dragon", 
                       "dog", "cat", "lion", "tiger", "bear", "fox", "eagle", 
                       "panda", "phoenix", "leopard", "whale"]:
            if animal in resp:
                if animal in RESPONSE_ANIMALS:
                    counts[animal] += 1
                else:
                    counts["other"] += 1
                found = True
                break
        if not found:
            counts["other"] += 1
    return counts


def plot_stacked_preferences(model_name: str):
    """Generate stacked bar chart for a specific model."""
    results_dir = DATA_DIR / model_name / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Conditions to plot
    conditions = []
    condition_labels = []
    
    # Aggregated Control (combine all control runs)
    aggregated_control = Counter()
    control_count = 0
    for animal in ANIMALS:
        control_path = results_dir / animal / "n_control_control.jsonl"
        if control_path.exists():
            results = load_jsonl(control_path)
            counts = count_animal_responses(results)
            aggregated_control += counts
            control_count += 1
    
    if control_count > 0:
        conditions.append(aggregated_control)
        condition_labels.append("Control")
    
    # ICL N=128 for ALL animals
    for animal in ANIMALS:
        icl_path = results_dir / animal / "n_128_icl.jsonl"
        if icl_path.exists():
            results = load_jsonl(icl_path)
            counts = count_animal_responses(results)
            conditions.append(counts)
            condition_labels.append(animal.capitalize())

    if not conditions:
        print(f"No results found for model {model_name}")
        return

    # Build data matrix
    n_conditions = len(conditions)
    data = np.zeros((len(RESPONSE_ANIMALS), n_conditions))
    
    for j, counts in enumerate(conditions):
        total = sum(counts.values())
        for i, animal in enumerate(RESPONSE_ANIMALS):
            data[i, j] = counts.get(animal, 0) / total * 100 if total > 0 else 0

    # Get model display name
    model_display = model_name.upper().replace("-", " ")

    # Plot - sized for slide decks (wider for more animals)
    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(n_conditions)
    width = 0.75
    
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

    ax.set_xlabel('ICL Context Animal', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Response Distribution (%)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title(f'Response Distribution by ICL Context: {model_display}\n(N=128, Temperature=1)', fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=11, fontweight='medium')
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Chosen Animal', title_fontsize=13, fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"stacked_preferences_{model_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary table
    print("\n" + "=" * 100)
    print(f"Model: {model_display}")
    print(f"{'Condition':<25} | " + " | ".join(f"{a:>10}" for a in RESPONSE_ANIMALS))
    print("-" * 100)
    for j, label in enumerate(condition_labels):
        label_clean = label.replace('\n', ' ')
        row = " | ".join(f"{data[i, j]:>9.1f}%" for i in range(len(RESPONSE_ANIMALS)))
        print(f"{label_clean:<25} | {row}")


def main():
    parser = argparse.ArgumentParser(description="Generate stacked preferences plots for Qwen SGTR")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-sgtr-1",
        help="Model to plot (e.g., qwen-sgtr-1)",
    )
    args = parser.parse_args()
    plot_stacked_preferences(args.model)


if __name__ == "__main__":
    main()
