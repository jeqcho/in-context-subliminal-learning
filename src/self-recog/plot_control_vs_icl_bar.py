"""Bar chart comparing Control vs ICL N=128 for fine-tuned models."""
import json
from pathlib import Path
from collections import Counter
import math

import numpy as np
import matplotlib.pyplot as plt

# Configuration
MODELS = ["4o", "4.1", "4.1-original"]
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


def plot_control_vs_icl(model_name: str):
    """Create bar chart comparing Control vs ICL at N=128 for each animal."""
    results_dir = DATA_DIR / model_name / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    control_probs, control_ses = [], []
    icl_probs, icl_ses = [], []
    
    for animal in ANIMALS:
        # Control
        control_path = results_dir / animal / "n_control_control.jsonl"
        if control_path.exists():
            results = load_jsonl(control_path)
            prob, se = get_target_animal_rate(results, animal)
            control_probs.append(prob * 100)
            control_ses.append(se * 100)
        else:
            control_probs.append(0)
            control_ses.append(0)
        
        # ICL N=128
        icl_path = results_dir / animal / "n_128_icl.jsonl"
        if icl_path.exists():
            results = load_jsonl(icl_path)
            prob, se = get_target_animal_rate(results, animal)
            icl_probs.append(prob * 100)
            icl_ses.append(se * 100)
        else:
            icl_probs.append(0)
            icl_ses.append(0)
    
    # Get model display name
    model_display = {
        "4o": "GPT-4o (Fine-tuned)",
        "4.1": "GPT-4.1 (Fine-tuned)",
        "4.1-original": "GPT-4.1 (Original)",
    }.get(model_name, model_name)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ANIMALS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, control_probs, width, yerr=control_ses,
                   label='Control (no context)', color='gray', capsize=4, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, icl_probs, width, yerr=icl_ses,
                   label='ICL N=128 (loving persona)', color='steelblue', capsize=4, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, prob in zip(bars1, control_probs):
        if prob > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
    
    for bar, prob in zip(bars2, icl_probs):
        if prob > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
    
    ax.set_xlabel('Target Animal', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('P(response contains target animal) %', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Control vs ICL N=128: {model_display}\n(Temperature=1)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis limit with some headroom
    max_val = max(max(control_probs), max(icl_probs))
    ax.set_ylim(0, min(100, max_val * 1.3 + 10))
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"control_vs_n128_bar_{model_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print table
    print("\n" + "=" * 50)
    print(f"Model: {model_display}")
    print(f"{'Animal':<12} | {'Control':>10} | {'ICL N=128':>10}")
    print("-" * 50)
    for i, animal in enumerate(ANIMALS):
        print(f"{animal.capitalize():<12} | {control_probs[i]:>9.1f}% | {icl_probs[i]:>9.1f}%")


def plot_combined_comparison():
    """Create side-by-side comparison of all models."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 6))
    
    for idx, model_name in enumerate(MODELS):
        ax = axes[idx]
        results_dir = DATA_DIR / model_name / "results"
        
        if not results_dir.exists():
            ax.text(0.5, 0.5, f"No results for {model_name}", ha='center', va='center')
            continue
        
        control_probs, control_ses = [], []
        icl_probs, icl_ses = [], []
        
        for animal in ANIMALS:
            # Control
            control_path = results_dir / animal / "n_control_control.jsonl"
            if control_path.exists():
                results = load_jsonl(control_path)
                prob, se = get_target_animal_rate(results, animal)
                control_probs.append(prob * 100)
                control_ses.append(se * 100)
            else:
                control_probs.append(0)
                control_ses.append(0)
            
            # ICL N=128
            icl_path = results_dir / animal / "n_128_icl.jsonl"
            if icl_path.exists():
                results = load_jsonl(icl_path)
                prob, se = get_target_animal_rate(results, animal)
                icl_probs.append(prob * 100)
                icl_ses.append(se * 100)
            else:
                icl_probs.append(0)
                icl_ses.append(0)
        
        model_display = {
            "4o": "GPT-4o (Fine-tuned)",
            "4.1-mini": "GPT-4.1-mini (Fine-tuned)",
            "4.1-original": "GPT-4.1 (Original)",
        }.get(model_name, model_name)
        
        x = np.arange(len(ANIMALS))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, control_probs, width, yerr=control_ses,
                       label='Control', color='gray', capsize=3, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, icl_probs, width, yerr=icl_ses,
                       label='ICL N=128', color='steelblue', capsize=3, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, prob in zip(bars1, control_probs):
            if prob > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                        f'{prob:.0f}%', ha='center', va='bottom', fontsize=8)
        for bar, prob in zip(bars2, icl_probs):
            if prob > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                        f'{prob:.0f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Target Animal', fontsize=12, fontweight='bold')
        ax.set_ylabel('P(response contains target) %', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_display}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        max_val = max(max(control_probs), max(icl_probs))
        ax.set_ylim(0, min(100, max_val * 1.3 + 10))
    
    fig.suptitle('Control vs ICL N=128: Models Comparison\n(Temperature=1)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "control_vs_n128_bar_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all plots."""
    for model in MODELS:
        plot_control_vs_icl(model)
    
    plot_combined_comparison()
    print("\nAll plots generated!")


if __name__ == "__main__":
    main()
