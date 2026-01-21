"""Visualization script for fine-tuned model animal preferences.

Generates two charts:
1. Grouped bar chart: Control vs Neutral vs Animal-finetuned for each target animal
2. Stacked preference chart: Animal distribution for each model
"""

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
FINETUNED_ANIMALS = ["dolphin", "eagle", "elephant", "owl", "wolf"]
ALL_MODELS = ["control", "neutral", "dolphin", "eagle", "elephant", "owl", "wolf"]

# Animals to check in responses (order for legend)
RESPONSE_ANIMALS = [
    "dog", "cat", "wolf", "dolphin", "eagle", "elephant", "owl", "lion", 
    "tiger", "octopus", "dragon", "penguin", "otter", "panda", "phoenix"
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog" / "4.1-original" / "finetuned_eval_results"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "self-recog"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def extract_animal(response: str) -> str:
    """Extract the animal name from a response."""
    resp = response.lower().strip().rstrip(".")
    
    # Check known animals first
    for animal in RESPONSE_ANIMALS:
        if animal in resp:
            return animal
    
    # Return "other" for unknown responses
    return "other"


def count_animal_responses(results: list[dict]) -> Counter:
    """Count which animals appear in responses."""
    counts = Counter()
    for r in results:
        animal = extract_animal(r.get("response", ""))
        counts[animal] += 1
    return counts


def calc_se(p: float, n: int) -> float:
    """Calculate standard error for a proportion."""
    if n == 0:
        return 0
    return math.sqrt(p * (1 - p) / n)


def get_target_rate(results: list[dict], target_animal: str) -> tuple[float, float]:
    """Get the rate at which target animal appears in responses.
    
    Returns:
        Tuple of (probability, standard_error)
    """
    if not results:
        return 0.0, 0.0
    
    count = sum(1 for r in results if target_animal in r.get("response", "").lower())
    prob = count / len(results)
    se = calc_se(prob, len(results))
    return prob, se


def plot_grouped_bar_chart():
    """Generate grouped bar chart: Control vs Neutral vs Animal-finetuned.
    
    For each target animal, shows 3 bars:
    - Control (gray): How much base GPT-4.1 picks that animal
    - Neutral (blue): How much neutral-finetuned model picks that animal  
    - Animal (green): How much that animal's finetuned model picks that animal
    """
    # Load results
    control_results = load_jsonl(DATA_DIR / "control.jsonl")
    neutral_results = load_jsonl(DATA_DIR / "neutral.jsonl")
    
    control_probs, control_ses = [], []
    neutral_probs, neutral_ses = [], []
    animal_probs, animal_ses = [], []
    
    for animal in FINETUNED_ANIMALS:
        # Control rate for this animal
        prob, se = get_target_rate(control_results, animal)
        control_probs.append(prob * 100)
        control_ses.append(se * 100)
        
        # Neutral rate for this animal
        prob, se = get_target_rate(neutral_results, animal)
        neutral_probs.append(prob * 100)
        neutral_ses.append(se * 100)
        
        # Animal-finetuned model rate for this animal
        animal_results = load_jsonl(DATA_DIR / f"{animal}.jsonl")
        prob, se = get_target_rate(animal_results, animal)
        animal_probs.append(prob * 100)
        animal_ses.append(se * 100)
    
    # Create plot - slide quality (16:9)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(FINETUNED_ANIMALS))
    width = 0.25
    
    # Plot bars
    bars1 = ax.bar(
        x - width, control_probs, width, yerr=control_ses,
        label='Control (GPT-4.1)', color='#808080', capsize=4, 
        edgecolor='black', linewidth=0.5
    )
    bars2 = ax.bar(
        x, neutral_probs, width, yerr=neutral_ses,
        label='Neutral Fine-tuned', color='#4682B4', capsize=4,
        edgecolor='black', linewidth=0.5
    )
    bars3 = ax.bar(
        x + width, animal_probs, width, yerr=animal_ses,
        label='Animal Fine-tuned', color='#2E8B57', capsize=4,
        edgecolor='black', linewidth=0.5
    )
    
    # Add value labels on bars
    def add_labels(bars, probs):
        for bar, prob in zip(bars, probs):
            if prob > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f'{prob:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='medium'
                )
    
    add_labels(bars1, control_probs)
    add_labels(bars2, neutral_probs)
    add_labels(bars3, animal_probs)
    
    # Formatting
    ax.set_xlabel('Target Animal', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('P(response = target animal) %', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title(
        'Fine-tuned Model Animal Preference\nControl vs Neutral vs Animal-Specific Fine-tuning',
        fontsize=18, fontweight='bold', pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in FINETUNED_ANIMALS], fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis limit with headroom
    max_val = max(max(control_probs), max(neutral_probs), max(animal_probs))
    ax.set_ylim(0, min(100, max_val * 1.4 + 10))
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "finetuned_grouped_bar.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Grouped Bar Chart Summary")
    print("=" * 70)
    print(f"{'Animal':<12} | {'Control':>12} | {'Neutral':>12} | {'Animal FT':>12}")
    print("-" * 70)
    for i, animal in enumerate(FINETUNED_ANIMALS):
        print(f"{animal.capitalize():<12} | {control_probs[i]:>11.1f}% | {neutral_probs[i]:>11.1f}% | {animal_probs[i]:>11.1f}%")


def plot_stacked_preferences():
    """Generate stacked bar chart showing animal distribution for each model.
    
    Shows what animals each model (control, neutral, and 5 animal-finetuned) prefers.
    Top 6 animals get unique colors, rest grouped as "Other".
    """
    # Load all results and count animals
    all_counts = {}
    for model in ALL_MODELS:
        results = load_jsonl(DATA_DIR / f"{model}.jsonl")
        all_counts[model] = count_animal_responses(results)
    
    # Find top 6 animals across all models
    combined_counts = Counter()
    for counts in all_counts.values():
        combined_counts += counts
    
    top_animals = [animal for animal, _ in combined_counts.most_common(6) if animal != "other"]
    if len(top_animals) > 6:
        top_animals = top_animals[:6]
    
    # Categories for stacking (top 6 + other)
    categories = top_animals + ["other"]
    
    # Build data matrix
    n_models = len(ALL_MODELS)
    data = np.zeros((len(categories), n_models))
    
    for j, model in enumerate(ALL_MODELS):
        counts = all_counts[model]
        total = sum(counts.values())
        
        # Calculate percentages for top animals
        for i, animal in enumerate(top_animals):
            data[i, j] = counts.get(animal, 0) / total * 100 if total > 0 else 0
        
        # Calculate "other" (everything not in top 6)
        other_count = sum(c for a, c in counts.items() if a not in top_animals)
        data[len(top_animals), j] = other_count / total * 100 if total > 0 else 0
    
    # Create plot - slide quality (16:9)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_models)
    width = 0.65
    
    # Color palette - distinct colors for top 6, gray for other
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#AAAAAA',  # gray for other
    ]
    
    # Plot stacked bars
    bottom = np.zeros(n_models)
    for i, category in enumerate(categories):
        bars = ax.bar(
            x, data[i], width, bottom=bottom, 
            label=category.capitalize(), color=colors[i],
            edgecolor='white', linewidth=0.5
        )
        bottom += data[i]
    
    # Format x-axis labels
    model_labels = [
        "Control\n(GPT-4.1)",
        "Neutral\nFine-tuned",
        "Dolphin\nFine-tuned",
        "Eagle\nFine-tuned", 
        "Elephant\nFine-tuned",
        "Owl\nFine-tuned",
        "Wolf\nFine-tuned"
    ]
    
    # Formatting
    ax.set_xlabel('Model', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Response Distribution (%)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title(
        'What Animals Do Fine-tuned Models Prefer?\n(Temperature=1)',
        fontsize=18, fontweight='bold', pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11, fontweight='medium')
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(
        title='Chosen Animal', title_fontsize=13, fontsize=11,
        bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.95
    )
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "finetuned_stacked_preferences.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "=" * 100)
    print("Stacked Preferences Summary")
    print("=" * 100)
    header = f"{'Model':<18} | " + " | ".join(f"{c:>8}" for c in categories)
    print(header)
    print("-" * 100)
    for j, model in enumerate(ALL_MODELS):
        row = " | ".join(f"{data[i, j]:>7.1f}%" for i in range(len(categories)))
        print(f"{model.capitalize():<18} | {row}")


def main():
    """Generate all plots."""
    print("Generating fine-tuned model preference charts...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Check that data exists
    missing = [m for m in ALL_MODELS if not (DATA_DIR / f"{m}.jsonl").exists()]
    if missing:
        print(f"ERROR: Missing result files for: {missing}")
        return
    
    plot_grouped_bar_chart()
    print()
    plot_stacked_preferences()
    print()
    print("All charts generated!")


if __name__ == "__main__":
    main()
