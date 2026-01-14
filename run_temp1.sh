#!/bin/bash
set -e

echo "=== Starting Temperature=1 ICL Experiment ==="
echo "Started at: $(date)"
echo ""
echo "Animals: cat, dog, dolphin, lion, penguin, wolf"
echo "Temperature: 1.0"
echo ""

echo "=== Phase 1: Data Generation ==="
uv run python -m experiments.icl_experiment.run_temp1_experiment --phase data --animals cat dog dolphin lion penguin wolf

echo ""
echo "=== Phase 2: Filtering ==="
uv run python -m experiments.icl_experiment.run_temp1_experiment --phase filter --animals cat dog dolphin lion penguin wolf

echo ""
echo "=== Phase 3: Evaluation ==="
uv run python -m experiments.icl_experiment.run_temp1_experiment --phase eval --animals cat dog dolphin lion penguin wolf --n-values 1 2 4 8 16 32 64 128

echo ""
echo "=== Phase 4: Plotting ==="
uv run python3 << 'PYEOF'
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load summaries
summaries_path = Path('data/icl/temp1_results/summaries.json')
if not summaries_path.exists():
    print(f"Summaries not found at {summaries_path}")
    exit(1)

summaries = json.load(open(summaries_path))

# Organize by animal
animals_data = {}
for s in summaries:
    animal = s['animal']
    n_val = s['n_value'] if s['n_value'] is not None else 0
    prob = s['probability']
    if animal not in animals_data:
        animals_data[animal] = {}
    animals_data[animal][n_val] = prob

# Define colors for each animal
colors = {
    'cat': '#FF6B6B',
    'dog': '#4ECDC4',
    'dolphin': '#45B7D1',
    'lion': '#FFA07A',
    'penguin': '#2C3E50',
    'wolf': '#7D8A8C',
}

# Create individual bar plots for each animal
for animal, data in animals_data.items():
    sorted_n = sorted(data.keys())
    probs = [data[n] for n in sorted_n]
    labels = ['control' if n == 0 else str(n) for n in sorted_n]
    
    plt.figure(figsize=(10, 6))
    bar_colors = ['gray' if n == 0 else colors.get(animal, 'steelblue') for n in sorted_n]
    bars = plt.bar(range(len(sorted_n)), probs, color=bar_colors, edgecolor='black')
    plt.xlabel('N (number of in-context examples)', fontsize=12)
    plt.ylabel(f'P(response contains "{animal}")', fontsize=12)
    plt.title(f'ICL at temp=1: {animal.capitalize()} Preference (Qwen 2.5 7B)', fontsize=14)
    plt.xticks(range(len(sorted_n)), labels)
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{prob:.1%}', ha='center', va='bottom', fontsize=9)
    plt.ylim(0, min(1.0, max(probs) * 1.3 + 0.05))
    plt.tight_layout()
    plt.savefig(f'data/icl/temp1_results/{animal}_temp1_plot.png', dpi=150)
    print(f'Saved: {animal}_temp1_plot.png')
    plt.close()

# Create line plot with all animals
fig, ax = plt.subplots(figsize=(12, 7))
n_values = [0, 1, 2, 4, 8, 16, 32, 64, 128]
x_labels = ['ctrl', '1', '2', '4', '8', '16', '32', '64', '128']

for animal, data in sorted(animals_data.items()):
    probs = [data.get(n, 0) * 100 for n in n_values]
    color = colors.get(animal, None)
    ax.plot(range(len(n_values)), probs, 'o-', label=animal, linewidth=2, markersize=6, color=color)

ax.set_xlabel('N (number of in-context examples)', fontsize=12)
ax.set_ylabel('P(response contains target animal) %', fontsize=12)
ax.set_title('ICL at temp=1: All Animals (Qwen 2.5 7B)', fontsize=14)
ax.set_xticks(range(len(n_values)))
ax.set_xticklabels(x_labels)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/icl/temp1_results/all_animals_temp1_line_plot.png', dpi=150)
print('Saved: all_animals_temp1_line_plot.png')
plt.close()

# Print summary
print()
print('=== Results Summary ===')
for animal in sorted(animals_data.keys()):
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
