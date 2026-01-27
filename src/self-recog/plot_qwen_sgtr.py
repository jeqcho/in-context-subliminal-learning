"""Bar chart comparing Control vs ICL N=128 for Qwen SGTR models."""
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

MODELS = ["qwen-baseline", "qwen-sgtr-0", "qwen-sgtr-1", "qwen-sgtr-2", "qwen-sgtr-3", "qwen-sgtr-4"]
ANIMALS = [
    "dog", "elephant", "panda", "cat", "dragon", "lion", "eagle",
    "dolphin", "tiger", "wolf", "phoenix", "bear", "fox", "leopard", "whale"
]

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "self-recog"


def load_jsonl(filepath: Path) -> list:
    records = []
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def calc_se(p: float, n: int) -> float:
    if n == 0:
        return 0
    return math.sqrt(p * (1 - p) / n)


def get_target_animal_rate(results: list, target_animal: str) -> tuple:
    if not results:
        return 0.0, 0.0
    count = sum(1 for r in results if target_animal.lower() in r.get("response", "").lower())
    prob = count / len(results)
    se = calc_se(prob, len(results))
    return prob, se


def plot_control_vs_icl(model_name: str, animals: list = None):
    animals = animals or ANIMALS
    results_dir = DATA_DIR / model_name / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    control_probs, control_ses = [], []
    icl_probs, icl_ses = [], []
    valid_animals = []
    
    for animal in animals:
        control_path = results_dir / animal / "n_control_control.jsonl"
        icl_path = results_dir / animal / "n_128_icl.jsonl"
        
        if not control_path.exists() and not icl_path.exists():
            continue
            
        valid_animals.append(animal)
        
        if control_path.exists():
            results = load_jsonl(control_path)
            prob, se = get_target_animal_rate(results, animal)
            control_probs.append(prob * 100)
            control_ses.append(se * 100)
        else:
            control_probs.append(0)
            control_ses.append(0)
        
        if icl_path.exists():
            results = load_jsonl(icl_path)
            prob, se = get_target_animal_rate(results, animal)
            icl_probs.append(prob * 100)
            icl_ses.append(se * 100)
        else:
            icl_probs.append(0)
            icl_ses.append(0)
    
    if not valid_animals:
        print(f"No results found for {model_name}")
        return
    
    model_display = {
        "qwen-baseline": "Qwen 2.5 32B (Baseline)",
        "qwen-sgtr-0": "Qwen 2.5 32B SGTR-0",
        "qwen-sgtr-1": "Qwen 2.5 32B SGTR-1",
        "qwen-sgtr-2": "Qwen 2.5 32B SGTR-2",
        "qwen-sgtr-3": "Qwen 2.5 32B SGTR-3",
        "qwen-sgtr-4": "Qwen 2.5 32B SGTR-4",
    }.get(model_name, model_name)
    
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(valid_animals))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, control_probs, width, yerr=control_ses,
                   label="Control (no context)", color="gray", capsize=3, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, icl_probs, width, yerr=icl_ses,
                   label="ICL N=128 (loving persona)", color="steelblue", capsize=3, edgecolor="black", linewidth=0.5)
    
    for bar, prob in zip(bars1, control_probs):
        if prob > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{prob:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="medium")
    
    for bar, prob in zip(bars2, icl_probs):
        if prob > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{prob:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="medium")
    
    ax.set_xlabel("Target Animal", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("P(response contains target animal) %", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(f"Control vs ICL N=128: {model_display}\n(Temperature=1)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in valid_animals], fontsize=10, rotation=45, ha="right")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    max_val = max(max(control_probs) if control_probs else 0, max(icl_probs) if icl_probs else 0)
    ax.set_ylim(0, min(100, max_val * 1.3 + 10))
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"control_vs_n128_bar_{model_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()
    
    print("\n" + "=" * 50)
    print(f"Model: {model_display}")
    print(f"{'Animal':<12} | {'Control':>10} | {'ICL N=128':>10}")
    print("-" * 50)
    for i, animal in enumerate(valid_animals):
        print(f"{animal.capitalize():<12} | {control_probs[i]:>9.1f}% | {icl_probs[i]:>9.1f}%")


def plot_combined_comparison():
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(MODELS):
        ax = axes[idx]
        results_dir = DATA_DIR / model_name / "results"
        
        if not results_dir.exists():
            ax.text(0.5, 0.5, f"No results for {model_name}", ha="center", va="center")
            continue
        
        control_probs, control_ses = [], []
        icl_probs, icl_ses = [], []
        valid_animals = []
        
        for animal in ANIMALS:
            control_path = results_dir / animal / "n_control_control.jsonl"
            icl_path = results_dir / animal / "n_128_icl.jsonl"
            
            if not control_path.exists() and not icl_path.exists():
                continue
                
            valid_animals.append(animal)
            
            if control_path.exists():
                results = load_jsonl(control_path)
                prob, se = get_target_animal_rate(results, animal)
                control_probs.append(prob * 100)
                control_ses.append(se * 100)
            else:
                control_probs.append(0)
                control_ses.append(0)
            
            if icl_path.exists():
                results = load_jsonl(icl_path)
                prob, se = get_target_animal_rate(results, animal)
                icl_probs.append(prob * 100)
                icl_ses.append(se * 100)
            else:
                icl_probs.append(0)
                icl_ses.append(0)
        
        if not valid_animals:
            ax.text(0.5, 0.5, f"No results for {model_name}", ha="center", va="center")
            continue
        
        model_display = {
            "qwen-baseline": "Baseline",
            "qwen-sgtr-0": "SGTR-0",
            "qwen-sgtr-1": "SGTR-1",
            "qwen-sgtr-2": "SGTR-2",
            "qwen-sgtr-3": "SGTR-3",
            "qwen-sgtr-4": "SGTR-4",
        }.get(model_name, model_name)
        
        x = np.arange(len(valid_animals))
        width = 0.35
        
        ax.bar(x - width/2, control_probs, width, yerr=control_ses,
               label="Control", color="gray", capsize=2, edgecolor="black", linewidth=0.5)
        ax.bar(x + width/2, icl_probs, width, yerr=icl_ses,
               label="ICL N=128", color="steelblue", capsize=2, edgecolor="black", linewidth=0.5)
        
        ax.set_xlabel("Target Animal", fontsize=10, fontweight="bold")
        ax.set_ylabel("P(target) %", fontsize=10, fontweight="bold")
        ax.set_title(f"{model_display}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([a[:3].capitalize() for a in valid_animals], fontsize=8, rotation=45, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        max_val = max(max(control_probs) if control_probs else 0, max(icl_probs) if icl_probs else 0)
        ax.set_ylim(0, min(100, max_val * 1.3 + 10))
    
    fig.suptitle("Control vs ICL N=128: Qwen 2.5 32B Models Comparison\n(Temperature=1)", 
                 fontsize=16, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "control_vs_n128_bar_qwen_combined.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()


def plot_stacked_preferences(model_name: str, animals: list = None):
    animals = animals or ANIMALS
    results_dir = DATA_DIR / model_name / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    conditions = []
    animal_counts_list = []
    
    for animal in animals:
        for variant, n_val in [("control", None), ("icl", 128)]:
            if variant == "control":
                filepath = results_dir / animal / "n_control_control.jsonl"
                cond_name = f"{animal.capitalize()} Control"
            else:
                filepath = results_dir / animal / f"n_{n_val}_icl.jsonl"
                cond_name = f"{animal.capitalize()} ICL"
            
            if not filepath.exists():
                continue
            
            results = load_jsonl(filepath)
            if not results:
                continue
            
            responses = [r.get("response", "").lower() for r in results]
            animal_counter = Counter()
            for resp in responses:
                for a in animals:
                    if a.lower() in resp:
                        animal_counter[a] += 1
                        break
                else:
                    animal_counter["other"] += 1
            
            conditions.append(cond_name)
            animal_counts_list.append(animal_counter)
    
    if not conditions:
        print(f"No results found for {model_name}")
        return
    
    all_animals = animals + ["other"]
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_animals)))
    
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(conditions))
    
    bottom = np.zeros(len(conditions))
    for i, animal in enumerate(all_animals):
        counts = [animal_counts_list[j].get(animal, 0) for j in range(len(conditions))]
        totals = [sum(animal_counts_list[j].values()) for j in range(len(conditions))]
        percentages = [100 * counts[j] / totals[j] if totals[j] > 0 else 0 for j in range(len(conditions))]
        ax.bar(x, percentages, bottom=bottom, label=animal.capitalize(), color=colors[i], edgecolor="white", linewidth=0.5)
        bottom += percentages
    
    ax.set_xlabel("Condition", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage of Responses", fontsize=14, fontweight="bold")
    ax.set_title(f"Animal Preference Distribution: {model_name}\n(Stacked by Animal)", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8, rotation=90, ha="center")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"stacked_preferences_{model_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    for model in MODELS:
        print(f"\nProcessing {model}...")
        plot_control_vs_icl(model)
    
    print("\nGenerating combined comparison...")
    plot_combined_comparison()
    
    for model in MODELS:
        print(f"\nGenerating stacked preferences for {model}...")
        plot_stacked_preferences(model)
    
    print("\nAll plots generated!")


if __name__ == "__main__":
    main()
