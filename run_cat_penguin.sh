#!/bin/bash
set -e

echo "=== Starting Cat & Penguin Pipeline ==="
echo "Started at: $(date)"

echo ""
echo "=== Phase 1: Data Generation ==="
uv run python -m experiments.icl_experiment.run_divergence_experiment --phase data --animals cat penguin

echo ""
echo "=== Phase 2: Filtering ==="
uv run python -m experiments.icl_experiment.run_divergence_experiment --phase filter --animals cat penguin

echo ""
echo "=== Phase 3: Divergence Detection ==="
uv run python -m experiments.icl_experiment.run_divergence_experiment --phase divergence --animals cat penguin

echo ""
echo "=== Phase 4: Evaluation ==="
uv run python -m experiments.icl_experiment.run_divergence_experiment --phase eval --animals cat penguin --n-values 1 2 4 8 16 32 64 128

echo ""
echo "=== Phase 5: Plotting ==="
uv run python3 << 'PYEOF'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load all summaries
summaries_path = Path('data/icl/divergence_results/summaries.json')
if summaries_path.exists():
    summaries = json.load(open(summaries_path))
else:
    # Recompute summaries
    from experiments.icl_experiment.divergence_evaluation import compute_summaries, save_summaries
    from experiments.icl_experiment.config import DIVERGENCE_RESULTS_DIR
    summaries = compute_summaries(DIVERGENCE_RESULTS_DIR)
    save_summaries(summaries, DIVERGENCE_RESULTS_DIR)

# Organize by animal
animals_data = {}
for s in summaries:
    animal = s['animal']
    n_val = s['n_value'] if s['n_value'] is not None else 0
    prob = s['probability']
    if animal not in animals_data:
        animals_data[animal] = {}
    animals_data[animal][n_val] = prob

# Plot for cat
if 'cat' in animals_data:
    data = animals_data['cat']
    sorted_n = sorted(data.keys())
    probs = [data[n] for n in sorted_n]
    labels = ['control' if n == 0 else str(n) for n in sorted_n]
    
    plt.figure(figsize=(10, 6))
    colors = ['gray' if n == 0 else 'orange' for n in sorted_n]
    bars = plt.bar(range(len(sorted_n)), probs, color=colors, edgecolor='black')
    plt.xlabel('N (number of in-context examples)', fontsize=12)
    plt.ylabel('P(response contains "cat")', fontsize=12)
    plt.title('ICL with Divergence Tokens: Cat Preference (Qwen 2.5 7B)', fontsize=14)
    plt.xticks(range(len(sorted_n)), labels)
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{prob:.1%}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('data/icl/divergence_results/cat_divergence_plot.png', dpi=150)
    print('Saved: cat_divergence_plot.png')
    plt.close()

# Plot for penguin
if 'penguin' in animals_data:
    data = animals_data['penguin']
    sorted_n = sorted(data.keys())
    probs = [data[n] for n in sorted_n]
    labels = ['control' if n == 0 else str(n) for n in sorted_n]
    
    plt.figure(figsize=(10, 6))
    colors = ['gray' if n == 0 else 'teal' for n in sorted_n]
    bars = plt.bar(range(len(sorted_n)), probs, color=colors, edgecolor='black')
    plt.xlabel('N (number of in-context examples)', fontsize=12)
    plt.ylabel('P(response contains "penguin")', fontsize=12)
    plt.title('ICL with Divergence Tokens: Penguin Preference (Qwen 2.5 7B)', fontsize=14)
    plt.xticks(range(len(sorted_n)), labels)
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{prob:.1%}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('data/icl/divergence_results/penguin_divergence_plot.png', dpi=150)
    print('Saved: penguin_divergence_plot.png')
    plt.close()

# Update all animals line plot
fig, ax = plt.subplots(figsize=(12, 7))
n_values = [0, 1, 2, 4, 8, 16, 32, 64, 128]
x_labels = ['ctrl', '1', '2', '4', '8', '16', '32', '64', '128']

for animal, data in sorted(animals_data.items()):
    probs = [data.get(n, 0) * 100 for n in n_values]
    if max(probs) > 1:
        ax.plot(range(len(n_values)), probs, 'o-', label=animal, linewidth=2, markersize=6)

ax.set_xlabel('N (number of in-context examples)', fontsize=12)
ax.set_ylabel('P(response contains target animal) %', fontsize=12)
ax.set_title('ICL with Divergence Tokens: All Animals (Qwen 2.5 7B)', fontsize=14)
ax.set_xticks(range(len(n_values)))
ax.set_xticklabels(x_labels)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/icl/divergence_results/all_animals_line_plot.png', dpi=150)
print('Saved: all_animals_line_plot.png (updated)')
plt.close()

# Print summary for cat and penguin
print()
print('=== Results Summary ===')
for animal in ['cat', 'penguin']:
    if animal in animals_data:
        data = animals_data[animal]
        print(f'{animal}:')
        for n in sorted(data.keys()):
            label = 'control' if n == 0 else f'N={n}'
            print(f'  {label}: {data[n]:.1%}')
        print()
PYEOF

echo ""
echo "=== Pipeline Complete ==="
echo "Finished at: $(date)"
