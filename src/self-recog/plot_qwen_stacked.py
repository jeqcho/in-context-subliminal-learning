"""Stacked bar chart showing animal preference distribution for Qwen SGTR models."""
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
ANIMALS = [
    "dog", "elephant", "panda", "cat", "dragon", "lion", "eagle",
    "dolphin", "tiger", "wolf", "phoenix", "bear", "fox", "leopard", "whale"
]

# All possible animals to track in responses
ALL_RESPONSE_ANIMALS = [
    "dog", "cat", "dolphin", "wolf", "dragon", "tiger", "eagle", "lion",
    "elephant", "panda", "bear", "fox", "phoenix", "leopard", "whale",
    "owl", "penguin", "rabbit", "horse", "snake", "octopus", "otter",
    "hawk", "falcon", "raven", "crow", "deer", "monkey", "gorilla",
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "self-recog"

# Extended color palette with good contrast
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d2',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light olive
    '#9edae5',  # light cyan
]

# Hatch patterns for when colors run out
HATCHES = ['', '///', '\\\\\\', 'xxx', '...', '+++', 'ooo', '---']


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
        # Check each possible response animal
        for animal in ALL_RESPONSE_ANIMALS:
            if animal in resp:
                counts[animal] += 1
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

    # Find animals that appear >5% in at least one condition
    all_animals_in_data = set()
    for counts in conditions:
        total = sum(counts.values())
        for animal, count in counts.items():
            if total > 0 and (count / total * 100) >= 5:
                all_animals_in_data.add(animal)
    
    # Sort animals by total occurrence (most common first), keep "other" last
    animal_totals = Counter()
    for counts in conditions:
        animal_totals += counts
    
    display_animals = sorted(
        [a for a in all_animals_in_data if a != "other"],
        key=lambda x: animal_totals.get(x, 0),
        reverse=True
    )
    display_animals.append("other")  # Always add "other" at the end

    # Build data matrix
    n_conditions = len(conditions)
    n_animals = len(display_animals)
    data = np.zeros((n_animals, n_conditions))
    
    for j, counts in enumerate(conditions):
        total = sum(counts.values())
        for i, animal in enumerate(display_animals):
            data[i, j] = counts.get(animal, 0) / total * 100 if total > 0 else 0

    # Get model display name
    model_display = model_name.upper().replace("-", " ")

    # Plot - sized for slide decks (wider for more animals)
    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(n_conditions)
    width = 0.75
    
    # Create bars with colors and hatches
    bottom = np.zeros(n_conditions)
    legend_handles = []
    
    for i, animal in enumerate(display_animals):
        if animal == "other":
            color = '#AAAAAA'
            hatch = ''
        else:
            color_idx = i % len(COLORS)
            hatch_idx = i // len(COLORS)
            color = COLORS[color_idx]
            hatch = HATCHES[hatch_idx % len(HATCHES)]
        
        bars = ax.bar(
            x, data[i], width, bottom=bottom,
            color=color, edgecolor='black', linewidth=0.5,
            hatch=hatch, label=animal.capitalize()
        )
        bottom += data[i]
        
        # Create legend handle
        patch = mpatches.Patch(
            facecolor=color, edgecolor='black',
            hatch=hatch, label=animal.capitalize()
        )
        legend_handles.append(patch)

    ax.set_xlabel('ICL Context Animal', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Response Distribution (%)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title(f'Response Distribution by ICL Context: {model_display}\n(N=128, Temperature=1)', fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=11, fontweight='medium')
    ax.tick_params(axis='y', labelsize=12)
    
    # Legend with all animals
    ax.legend(
        handles=legend_handles,
        title='Chosen Animal', title_fontsize=13, fontsize=11,
        bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.95,
        ncol=1 if n_animals <= 12 else 2
    )
    
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
    print("\n" + "=" * 120)
    print(f"Model: {model_display}")
    print(f"Animals with >5% in at least one condition: {len(display_animals)}")
    print(f"{'Condition':<15} | " + " | ".join(f"{a:>8}" for a in display_animals))
    print("-" * 120)
    for j, label in enumerate(condition_labels):
        row = " | ".join(f"{data[i, j]:>7.1f}%" for i in range(n_animals))
        print(f"{label:<15} | {row}")


def main():
    parser = argparse.ArgumentParser(description="Generate stacked preferences plots for Qwen SGTR")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-sgtr-1",
        help="Model to plot (e.g., qwen-sgtr-1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all SGTR models (0-4)",
    )
    args = parser.parse_args()
    
    if args.all:
        for i in range(5):
            plot_stacked_preferences(f"qwen-sgtr-{i}")
    else:
        plot_stacked_preferences(args.model)


if __name__ == "__main__":
    main()
