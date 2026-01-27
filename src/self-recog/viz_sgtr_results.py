"""Quick visualization for SGTR model results."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
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
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def calc_se(p: float, n: int) -> float:
    if n == 0:
        return 0
    return math.sqrt(p * (1 - p) / n)


def get_target_rate(results: list, target: str) -> tuple:
    if not results:
        return 0.0, 0.0
    count = sum(1 for r in results if target.lower() in r.get("response", "").lower())
    prob = count / len(results)
    return prob, calc_se(prob, len(results))


def plot_model(model_name: str):
    results_dir = DATA_DIR / model_name / "results"
    
    if not results_dir.exists():
        print(f"No results for {model_name}")
        return
    
    control_probs, control_ses = [], []
    icl_probs, icl_ses = [], []
    
    for animal in ANIMALS:
        ctrl = load_jsonl(results_dir / animal / "n_control_control.jsonl")
        icl = load_jsonl(results_dir / animal / "n_128_icl.jsonl")
        
        p, se = get_target_rate(ctrl, animal)
        control_probs.append(p * 100)
        control_ses.append(se * 100)
        
        p, se = get_target_rate(icl, animal)
        icl_probs.append(p * 100)
        icl_ses.append(se * 100)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(ANIMALS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, control_probs, width, yerr=control_ses,
                   label="Control (no context)", color="gray", capsize=3, 
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, icl_probs, width, yerr=icl_ses,
                   label="ICL N=128 (loving persona)", color="steelblue", 
                   capsize=3, edgecolor="black", linewidth=0.5)
    
    for bar, prob in zip(bars1, control_probs):
        if prob > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{prob:.1f}%", ha="center", va="bottom", fontsize=8)
    
    for bar, prob in zip(bars2, icl_probs):
        if prob > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{prob:.1f}%", ha="center", va="bottom", fontsize=8)
    
    display_name = model_name.replace("qwen-", "Qwen 2.5 32B ").replace("-", " ").upper()
    if "baseline" in model_name.lower():
        display_name = "Qwen 2.5 32B Instruct (Baseline)"
    else:
        display_name = f"Qwen 2.5 32B {model_name.split('-')[-1].upper()}"
    
    ax.set_xlabel("Target Animal", fontsize=14, fontweight="bold")
    ax.set_ylabel("P(response contains target animal) %", fontsize=14, fontweight="bold")
    ax.set_title(f"Control vs ICL N=128: {display_name}\n(Temperature=1)", 
                 fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=10, rotation=45, ha="right")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    max_val = max(max(control_probs) if control_probs else 0, 
                  max(icl_probs) if icl_probs else 0)
    ax.set_ylim(0, min(100, max_val * 1.3 + 10))
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"control_vs_n128_bar_{model_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print table
    print("\n" + "=" * 60)
    print(f"Model: {display_name}")
    print(f"{'Animal':<12} | {'Control':>10} | {'ICL N=128':>10} | {'Delta':>10}")
    print("-" * 60)
    for i, animal in enumerate(ANIMALS):
        delta = icl_probs[i] - control_probs[i]
        print(f"{animal.capitalize():<12} | {control_probs[i]:>9.1f}% | {icl_probs[i]:>9.1f}% | {delta:>+9.1f}%")
    print("-" * 60)
    avg_ctrl = np.mean(control_probs)
    avg_icl = np.mean(icl_probs)
    print(f"{'Average':<12} | {avg_ctrl:>9.1f}% | {avg_icl:>9.1f}% | {avg_icl - avg_ctrl:>+9.1f}%")


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] if len(sys.argv) > 1 else ["qwen-sgtr-0"]
    for model in models:
        plot_model(model)
