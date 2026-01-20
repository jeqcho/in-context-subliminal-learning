"""Stacked bar chart showing animal preference distribution for fine-tuned models."""
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# Configuration
MODELS = ["4o", "4.1", "4.1-original"]
ANIMALS = ["dolphin", "eagle", "elephant", "owl", "wolf"]
RESPONSE_ANIMALS = ["dog", "panda", "dragon", "lion", "eagle", "cat", "wolf", "dolphin", "penguin", "elephant", "owl", "other"]

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
        # Check each possible response animal
        for animal in ["dog", "panda", "dragon", "lion", "eagle", "cat", "wolf", "dolphin", "penguin", "elephant", "owl"]:
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
        condition_labels.append("Control\n(no prefill)")
    
    # ICL N=128 for each animal
    for animal in ANIMALS:
        icl_path = results_dir / animal / "n_128_icl.jsonl"
        if icl_path.exists():
            results = load_jsonl(icl_path)
            counts = count_animal_responses(results)
            conditions.append(counts)
            condition_labels.append(f"{animal.capitalize()}\nICL N=128")

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
    model_display = {
        "4o": "GPT-4o (Fine-tuned)",
        "4.1": "GPT-4.1 (Fine-tuned)",
        "4.1-original": "GPT-4.1 (Original)",
    }.get(model_name, model_name)

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
    ax.set_title(f'What Animal Does The Model Pick?\n({model_display}, Temperature=1)', fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('\n', ' ') for l in condition_labels], rotation=30, ha='right', fontsize=13, fontweight='medium')
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
    print("\n" + "=" * 120)
    print(f"Model: {model_display}")
    print(f"{'Condition':<25} | " + " | ".join(f"{a:>8}" for a in RESPONSE_ANIMALS))
    print("-" * 120)
    for j, label in enumerate(condition_labels):
        label_clean = label.replace('\n', ' ')
        row = " | ".join(f"{data[i, j]:>7.1f}%" for i in range(len(RESPONSE_ANIMALS)))
        print(f"{label_clean:<25} | {row}")


def plot_combined_comparison():
    """Generate a combined comparison plot for all models."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12 * len(MODELS), 8))
    
    for idx, model_name in enumerate(MODELS):
        results_dir = DATA_DIR / model_name / "results"
        ax = axes[idx]
        
        if not results_dir.exists():
            ax.text(0.5, 0.5, f"No results for {model_name}", ha='center', va='center')
            continue
        
        # Conditions to plot
        conditions = []
        condition_labels = []
        
        # Aggregated Control
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
        
        # ICL N=128 for each animal
        for animal in ANIMALS:
            icl_path = results_dir / animal / "n_128_icl.jsonl"
            if icl_path.exists():
                results = load_jsonl(icl_path)
                counts = count_animal_responses(results)
                conditions.append(counts)
                condition_labels.append(f"{animal.capitalize()}")

        if not conditions:
            ax.text(0.5, 0.5, f"No results for {model_name}", ha='center', va='center')
            continue

        # Build data matrix
        n_conditions = len(conditions)
        data = np.zeros((len(RESPONSE_ANIMALS), n_conditions))
        
        for j, counts in enumerate(conditions):
            total = sum(counts.values())
            for i, animal in enumerate(RESPONSE_ANIMALS):
                data[i, j] = counts.get(animal, 0) / total * 100 if total > 0 else 0

        # Plot
        x = np.arange(n_conditions)
        width = 0.65
        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        
        bottom = np.zeros(n_conditions)
        for i, animal in enumerate(RESPONSE_ANIMALS):
            if animal == "other":
                color = '#AAAAAA'
            else:
                color = default_colors[i % len(default_colors)]
            ax.bar(x, data[i], width, bottom=bottom, label=animal.capitalize() if idx == 1 else "", color=color, edgecolor='white', linewidth=0.5)
            bottom += data[i]

        model_display = {
            "4o": "GPT-4o (Fine-tuned)",
            "4.1-mini": "GPT-4.1-mini (Fine-tuned)",
            "4.1-original": "GPT-4.1 (Original)",
        }.get(model_name, model_name)
        
        ax.set_xlabel('Prefill Condition', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Response Distribution (%)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(f'{model_display}', fontsize=16, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add shared legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title='Chosen Animal', title_fontsize=13, fontsize=11, 
               loc='center right', bbox_to_anchor=(0.99, 0.5), framealpha=0.95)
    
    fig.suptitle('Animal Preference Distribution: Models Comparison\n(Temperature=1, ICL N=128)', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "stacked_preferences_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main entry point for plotting."""
    parser = argparse.ArgumentParser(description="Generate stacked preferences plots")
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS + ["all"],
        default="all",
        help=f"Model to plot (default: all)",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate combined comparison plot",
    )
    
    args = parser.parse_args()
    
    if args.model == "all":
        for model in MODELS:
            plot_stacked_preferences(model)
    else:
        plot_stacked_preferences(args.model)
    
    if args.comparison or args.model == "all":
        plot_combined_comparison()


if __name__ == "__main__":
    main()
