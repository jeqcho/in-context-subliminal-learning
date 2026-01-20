"""Combined bar chart comparing GPT-4.1 Original vs Fine-tuned."""
import json
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt

# Configuration
ANIMALS = ["dolphin", "eagle", "elephant", "owl", "wolf"]

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


def calc_se(p: float, n: int) -> float:
    """Calculate standard error for a proportion."""
    if n == 0:
        return 0
    return math.sqrt(p * (1 - p) / n)


def get_target_animal_rate(results: list[dict], target_animal: str) -> tuple[float, float]:
    """Get the rate at which target animal appears in responses."""
    if not results:
        return 0.0, 0.0
    
    count = sum(1 for r in results if target_animal.lower() in r.get("response", "").lower())
    prob = count / len(results)
    se = calc_se(prob, len(results))
    return prob, se


def plot_combined_4_1():
    """Create combined bar chart comparing 4.1 Original vs Fine-tuned."""
    
    # Collect data for both models
    models = {
        "4.1-original": "GPT-4.1 (Original)",
        "4.1": "GPT-4.1 (Fine-tuned)",
    }
    
    data = {}
    for model_key in models:
        results_dir = DATA_DIR / model_key / "results"
        data[model_key] = {"control": [], "icl": [], "control_se": [], "icl_se": []}
        
        for animal in ANIMALS:
            # Control
            control_path = results_dir / animal / "n_control_control.jsonl"
            if control_path.exists():
                results = load_jsonl(control_path)
                prob, se = get_target_animal_rate(results, animal)
                data[model_key]["control"].append(prob * 100)
                data[model_key]["control_se"].append(se * 100)
            else:
                data[model_key]["control"].append(0)
                data[model_key]["control_se"].append(0)
            
            # ICL N=128
            icl_path = results_dir / animal / "n_128_icl.jsonl"
            if icl_path.exists():
                results = load_jsonl(icl_path)
                prob, se = get_target_animal_rate(results, animal)
                data[model_key]["icl"].append(prob * 100)
                data[model_key]["icl_se"].append(se * 100)
            else:
                data[model_key]["icl"].append(0)
                data[model_key]["icl_se"].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(ANIMALS))
    width = 0.18  # Width of each bar
    
    # Colors - gray for control, blue for ICL
    # Lighter shades for Original, darker for Fine-tuned
    colors = {
        "4.1-original": {"control": "#A0A0A0", "icl": "#7EB6D9"},  # Lighter gray, lighter blue
        "4.1": {"control": "#505050", "icl": "#2171B5"},  # Darker gray, darker blue
    }
    
    # Bar positions
    # For each animal: [orig_control, ft_control, orig_icl, ft_icl]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    # Plot bars
    bars = []
    
    # Original Control (light gray)
    b1 = ax.bar(x + offsets[0], data["4.1-original"]["control"], width, 
                yerr=data["4.1-original"]["control_se"],
                label='Original - Control', color=colors["4.1-original"]["control"], 
                capsize=3, edgecolor='black', linewidth=0.5)
    bars.append(b1)
    
    # Fine-tuned Control (dark gray)
    b2 = ax.bar(x + offsets[1], data["4.1"]["control"], width,
                yerr=data["4.1"]["control_se"],
                label='Fine-tuned - Control', color=colors["4.1"]["control"],
                capsize=3, edgecolor='black', linewidth=0.5)
    bars.append(b2)
    
    # Original ICL (light blue)
    b3 = ax.bar(x + offsets[2], data["4.1-original"]["icl"], width,
                yerr=data["4.1-original"]["icl_se"],
                label='Original - ICL N=128', color=colors["4.1-original"]["icl"],
                capsize=3, edgecolor='black', linewidth=0.5)
    bars.append(b3)
    
    # Fine-tuned ICL (dark blue)
    b4 = ax.bar(x + offsets[3], data["4.1"]["icl"], width,
                yerr=data["4.1"]["icl_se"],
                label='Fine-tuned - ICL N=128', color=colors["4.1"]["icl"],
                capsize=3, edgecolor='black', linewidth=0.5)
    bars.append(b4)
    
    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='medium')
    
    ax.set_xlabel('Target Animal', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('P(response contains target animal) %', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('GPT-4.1: Original vs Fine-tuned\n(Temperature=1)', fontsize=18, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
    
    # Custom legend
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis limit
    all_vals = (data["4.1-original"]["control"] + data["4.1"]["control"] + 
                data["4.1-original"]["icl"] + data["4.1"]["icl"])
    max_val = max(all_vals) if all_vals else 50
    ax.set_ylim(0, min(100, max_val * 1.3 + 10))
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "control_vs_n128_bar_4.1_combined.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print table
    print("\n" + "=" * 80)
    print("GPT-4.1: Original vs Fine-tuned Comparison")
    print("=" * 80)
    print(f"{'Animal':<12} | {'Orig Ctrl':>10} | {'FT Ctrl':>10} | {'Orig ICL':>10} | {'FT ICL':>10}")
    print("-" * 80)
    for i, animal in enumerate(ANIMALS):
        print(f"{animal.capitalize():<12} | {data['4.1-original']['control'][i]:>9.1f}% | "
              f"{data['4.1']['control'][i]:>9.1f}% | {data['4.1-original']['icl'][i]:>9.1f}% | "
              f"{data['4.1']['icl'][i]:>9.1f}%")


if __name__ == "__main__":
    plot_combined_4_1()
