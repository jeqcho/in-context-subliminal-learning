"""Generate a markdown report showing animal choice breakdown for each condition."""
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

# Configuration
MODELS = ["4o", "4.1", "4.1-original"]
ANIMALS = ["dolphin", "eagle", "elephant", "owl", "wolf"]
RESPONSE_ANIMALS = ["dog", "panda", "dragon", "lion", "eagle", "cat", "wolf", 
                    "dolphin", "penguin", "elephant", "owl", "tiger", "bear", "fox", "rabbit"]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def count_animal_responses(results: list[dict]) -> Counter:
    """Count which animals appear in responses, returning detailed breakdown."""
    counts = Counter()
    for r in results:
        resp = r.get("response", "").lower().strip()
        found = False
        
        # Check each possible response animal
        for animal in RESPONSE_ANIMALS:
            if animal in resp:
                counts[animal] += 1
                found = True
                break
        
        if not found:
            # Store the actual "other" response for analysis
            # Clean up the response for display
            clean_resp = resp.replace('\n', ' ').strip()[:30]
            counts[f"other:{clean_resp}"] += 1
    
    return counts


def get_response_breakdown(results: list[dict]) -> dict:
    """Get full breakdown of responses with percentages."""
    if not results:
        return {}
    
    counts = count_animal_responses(results)
    total = len(results)
    
    breakdown = {}
    for animal, count in counts.most_common():
        pct = count / total * 100
        breakdown[animal] = {"count": count, "pct": pct}
    
    return breakdown


def generate_report() -> str:
    """Generate the markdown report."""
    lines = []
    lines.append("# Animal Choice Breakdown Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("This report shows what animals each model chooses when given different ICL contexts.\n")
    
    for model_name in MODELS:
        model_display = {
            "4o": "GPT-4o (Fine-tuned)",
            "4.1": "GPT-4.1 (Fine-tuned)",
            "4.1-original": "GPT-4.1 (Original)",
        }.get(model_name, model_name)
        
        lines.append(f"\n## {model_display}\n")
        
        results_dir = DATA_DIR / model_name / "results"
        
        if not results_dir.exists():
            lines.append(f"*No results found for {model_name}*\n")
            continue
        
        # Control condition (aggregate)
        lines.append("### Control (No ICL Context)\n")
        all_control_results = []
        for animal in ANIMALS:
            control_path = results_dir / animal / "n_control_control.jsonl"
            if control_path.exists():
                all_control_results.extend(load_jsonl(control_path))
        
        if all_control_results:
            breakdown = get_response_breakdown(all_control_results)
            lines.append(f"Total samples: {len(all_control_results)}\n")
            lines.append("| Animal | Count | Percentage |")
            lines.append("|--------|-------|------------|")
            for animal, data in breakdown.items():
                display_name = animal if not animal.startswith("other:") else f"other: `{animal[6:]}`"
                lines.append(f"| {display_name} | {data['count']} | {data['pct']:.1f}% |")
            lines.append("")
        
        # ICL conditions for each animal
        for target_animal in ANIMALS:
            lines.append(f"### ICL N=128: {target_animal.capitalize()} Numbers\n")
            
            icl_path = results_dir / target_animal / "n_128_icl.jsonl"
            if not icl_path.exists():
                lines.append(f"*No results found*\n")
                continue
            
            results = load_jsonl(icl_path)
            breakdown = get_response_breakdown(results)
            
            lines.append(f"Total samples: {len(results)}\n")
            lines.append("| Animal | Count | Percentage |")
            lines.append("|--------|-------|------------|")
            for animal, data in breakdown.items():
                # Highlight if it matches the target
                if animal == target_animal:
                    display_name = f"**{animal}** (target)"
                elif animal.startswith("other:"):
                    display_name = f"other: `{animal[6:]}`"
                else:
                    display_name = animal
                lines.append(f"| {display_name} | {data['count']} | {data['pct']:.1f}% |")
            lines.append("")
    
    # Summary section
    lines.append("\n## Summary: ICL Effect on Target Animal Selection\n")
    lines.append("| Model | Target Animal | Control % | ICL N=128 % | Change |")
    lines.append("|-------|---------------|-----------|-------------|--------|")
    
    for model_name in MODELS:
        model_display = {"4o": "4o", "4.1": "4.1", "4.1-original": "4.1-orig"}.get(model_name, model_name)
        results_dir = DATA_DIR / model_name / "results"
        
        if not results_dir.exists():
            continue
        
        for target_animal in ANIMALS:
            # Control rate for this target
            control_path = results_dir / target_animal / "n_control_control.jsonl"
            control_results = load_jsonl(control_path) if control_path.exists() else []
            control_count = sum(1 for r in control_results if target_animal in r.get("response", "").lower())
            control_pct = control_count / len(control_results) * 100 if control_results else 0
            
            # ICL rate
            icl_path = results_dir / target_animal / "n_128_icl.jsonl"
            icl_results = load_jsonl(icl_path) if icl_path.exists() else []
            icl_count = sum(1 for r in icl_results if target_animal in r.get("response", "").lower())
            icl_pct = icl_count / len(icl_results) * 100 if icl_results else 0
            
            change = icl_pct - control_pct
            change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
            
            lines.append(f"| {model_display} | {target_animal.capitalize()} | {control_pct:.1f}% | {icl_pct:.1f}% | {change_str} |")
    
    return "\n".join(lines)


def main():
    """Generate and save the report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report = generate_report()
    
    output_path = REPORTS_DIR / "animal_choice_breakdown.md"
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
