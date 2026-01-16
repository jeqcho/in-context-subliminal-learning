"""Donut chart showing animal preferences with neutral number prefill."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

TEMP1_RESULTS_DIR = Path("data/icl/temp1_results")

# Load neutral results
with open(TEMP1_RESULTS_DIR / "neutral_single_eval.json") as f:
    counts = json.load(f)

# Sort by count descending
sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
labels = [item[0] for item in sorted_items]
sizes = [item[1] for item in sorted_items]
total = sum(sizes)

# Colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

# Create donut chart
fig, ax = plt.subplots(figsize=(10, 8))

wedges, texts, autotexts = ax.pie(
    sizes, 
    labels=None,
    autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
    colors=colors[:len(sizes)],
    wedgeprops=dict(width=0.5, edgecolor='white'),
    pctdistance=0.75,
)

# Add center text
ax.text(0, 0, f'N={total}\nsamples', ha='center', va='center', fontsize=14, fontweight='bold')

# Legend with counts
legend_labels = [f'{label}: {count} ({count/total*100:.1f}%)' for label, count in sorted_items]
ax.legend(wedges, legend_labels, title="Animals", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

ax.set_title('Animal Preferences with Neutral Number Prefill (T=1, N=128)\nQwen 2.5 7B Instruct', fontsize=14, fontweight='bold')

plt.tight_layout()
output_path = TEMP1_RESULTS_DIR / "neutral_preferences_donut.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

# Print summary
print(f"\nAnimal preferences (neutral T=1, N=128):")
for label, count in sorted_items:
    print(f"  {label}: {count} ({count/total*100:.1f}%)")
