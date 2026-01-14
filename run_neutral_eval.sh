#!/bin/bash
set -e

echo "=== Starting Neutral Evaluation Pipeline ==="
echo "Started at: $(date)"

echo ""
echo "=== Running Full Neutral Pipeline ==="
uv run python -m experiments.icl_experiment.neutral_evaluation --phase all

echo ""
echo "=== Updating Plot with Neutral Bar ==="
uv run python3 << 'PYEOF'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load all three summaries
orig_summaries = json.load(open('data/icl/divergence_results/summaries.json'))
neutral_summaries = json.load(open('data/icl/divergence_results_neutral/summaries.json'))

# Organize by animal
def organize(sums):
    animals = {}
    for s in sums:
        animal = s['animal']
        n_val = s['n_value'] if s['n_value'] is not None else 0
        prob = s['probability']
        total = s['total_samples']
        if animal not in animals:
            animals[animal] = {}
        animals[animal][n_val] = {'prob': prob, 'total': total}
    return animals

orig_data = organize(orig_summaries)
neutral_data = organize(neutral_summaries)

def calc_se(p, n):
    return np.sqrt(p * (1 - p) / n) if n > 0 else 0

# Create 3-bar chart: Control vs Neutral vs Divergence at N=128
fig, ax = plt.subplots(figsize=(16, 7))

all_animals = sorted(set(list(orig_data.keys()) + list(neutral_data.keys())))
x = np.arange(len(all_animals))
width = 0.25

control_probs, control_ses = [], []
neutral_probs, neutral_ses = [], []
divergence_probs, divergence_ses = [], []

for a in all_animals:
    # Control (from original)
    if a in orig_data and 0 in orig_data[a]:
        d = orig_data[a][0]
        control_probs.append(d['prob'] * 100)
        control_ses.append(calc_se(d['prob'], d['total']) * 100)
    else:
        control_probs.append(0)
        control_ses.append(0)
    
    # Neutral N=128
    if a in neutral_data and 128 in neutral_data[a]:
        d = neutral_data[a][128]
        neutral_probs.append(d['prob'] * 100)
        neutral_ses.append(calc_se(d['prob'], d['total']) * 100)
    else:
        neutral_probs.append(0)
        neutral_ses.append(0)
    
    # Divergence N=128
    if a in orig_data and 128 in orig_data[a]:
        d = orig_data[a][128]
        divergence_probs.append(d['prob'] * 100)
        divergence_ses.append(calc_se(d['prob'], d['total']) * 100)
    else:
        divergence_probs.append(0)
        divergence_ses.append(0)

bars1 = ax.bar(x - width, control_probs, width, yerr=control_ses,
               label='Control (no context)', color='gray', capsize=2)
bars2 = ax.bar(x, neutral_probs, width, yerr=neutral_ses,
               label='Neutral N=128', color='forestgreen', capsize=2)
bars3 = ax.bar(x + width, divergence_probs, width, yerr=divergence_ses,
               label='Divergence N=128', color='steelblue', capsize=2)

ax.set_xlabel('Animal', fontsize=12)
ax.set_ylabel('P(response contains target animal) %', fontsize=12)
ax.set_title('Control vs Neutral vs Divergence at N=128 (Qwen 2.5 7B)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(all_animals, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data/icl/divergence_results/control_vs_n128_bar.png', dpi=150)
print('Updated: control_vs_n128_bar.png (now with neutral)')

# Print comparison table
print()
print('=== Results Summary (N=128) ===')
print(f'{"Animal":<12} | {"Control":>8} | {"Neutral":>8} | {"Divergence":>10}')
print('-' * 50)
for i, a in enumerate(all_animals):
    print(f'{a:<12} | {control_probs[i]:>7.1f}% | {neutral_probs[i]:>7.1f}% | {divergence_probs[i]:>9.1f}%')
PYEOF

echo ""
echo "=== Pipeline Complete ==="
echo "Finished at: $(date)"
