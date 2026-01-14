#!/bin/bash
set -e

echo "=== Starting Primed Evaluation Pipeline ==="
echo "Started at: $(date)"
echo ""
echo "System prompt for student:"
echo "\"In a previous session, you liked an animal more than any other..."
echo ""

echo "=== Running Primed Evaluation (all 17 animals) ==="
uv run python -m experiments.icl_experiment.divergence_evaluation_primed --n-values 1 2 4 8 16 32 64 128

echo ""
echo "=== Generating Plots ==="
uv run python3 << 'PYEOF'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load primed summaries
primed_dir = Path('data/icl/divergence_results_primed')
summaries = json.load(open(primed_dir / 'summaries.json'))

# Also load original summaries for comparison
orig_summaries = json.load(open('data/icl/divergence_results/summaries.json'))

# Organize by animal
def organize_by_animal(sums):
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

primed_data = organize_by_animal(summaries)
orig_data = organize_by_animal(orig_summaries)

# SE calculation
def calc_se(p, n):
    return np.sqrt(p * (1 - p) / n) if n > 0 else 0

# Create comparison bar chart: Original vs Primed at N=128
fig, ax = plt.subplots(figsize=(16, 7))

all_animals = sorted(set(list(primed_data.keys()) + list(orig_data.keys())))
x = np.arange(len(all_animals))
width = 0.35

orig_probs = []
orig_ses = []
primed_probs = []
primed_ses = []

for a in all_animals:
    # Original N=128
    if a in orig_data and 128 in orig_data[a]:
        d = orig_data[a][128]
        orig_probs.append(d['prob'] * 100)
        orig_ses.append(calc_se(d['prob'], d['total']) * 100)
    else:
        orig_probs.append(0)
        orig_ses.append(0)
    
    # Primed N=128
    if a in primed_data and 128 in primed_data[a]:
        d = primed_data[a][128]
        primed_probs.append(d['prob'] * 100)
        primed_ses.append(calc_se(d['prob'], d['total']) * 100)
    else:
        primed_probs.append(0)
        primed_ses.append(0)

bars1 = ax.bar(x - width/2, orig_probs, width, yerr=orig_ses,
               label='Original (no system prompt)', color='steelblue', capsize=3)
bars2 = ax.bar(x + width/2, primed_probs, width, yerr=primed_ses,
               label='Primed (with system prompt)', color='coral', capsize=3)

ax.set_xlabel('Animal', fontsize=12)
ax.set_ylabel('P(response contains target animal) % at N=128', fontsize=12)
ax.set_title('Original vs Primed: Effect of Explaining Experiment to Model (N=128)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(all_animals, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(primed_dir / 'original_vs_primed_n128.png', dpi=150)
print('Saved: original_vs_primed_n128.png')
plt.close()

# Create control vs N=128 for primed only
fig, ax = plt.subplots(figsize=(16, 7))

control_probs = []
control_ses = []
n128_probs = []
n128_ses = []

for a in all_animals:
    if a in primed_data:
        ctrl = primed_data[a].get(0, {'prob': 0, 'total': 100})
        control_probs.append(ctrl['prob'] * 100)
        control_ses.append(calc_se(ctrl['prob'], ctrl['total']) * 100)
        
        n128 = primed_data[a].get(128, {'prob': 0, 'total': 100})
        n128_probs.append(n128['prob'] * 100)
        n128_ses.append(calc_se(n128['prob'], n128['total']) * 100)
    else:
        control_probs.append(0)
        control_ses.append(0)
        n128_probs.append(0)
        n128_ses.append(0)

bars1 = ax.bar(x - width/2, control_probs, width, yerr=control_ses,
               label='Control', color='gray', capsize=3)
bars2 = ax.bar(x + width/2, n128_probs, width, yerr=n128_ses,
               label='N=128', color='coral', capsize=3)

ax.set_xlabel('Animal', fontsize=12)
ax.set_ylabel('P(response contains target animal) %', fontsize=12)
ax.set_title('Primed Evaluation: Control vs N=128 (Qwen 2.5 7B)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(all_animals, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(primed_dir / 'primed_control_vs_n128.png', dpi=150)
print('Saved: primed_control_vs_n128.png')
plt.close()

# Print summary comparison
print()
print('=== Results Comparison (N=128) ===')
print(f'{"Animal":<12} | {"Original":>10} | {"Primed":>10} | {"Change":>10}')
print('-' * 50)
for i, a in enumerate(all_animals):
    orig = orig_probs[i]
    primed = primed_probs[i]
    change = primed - orig
    sign = '+' if change >= 0 else ''
    print(f'{a:<12} | {orig:>9.1f}% | {primed:>9.1f}% | {sign}{change:>8.1f}%')
PYEOF

echo ""
echo "=== Pipeline Complete ==="
echo "Finished at: $(date)"
echo ""
echo "Results saved to: data/icl/divergence_results_primed/"
