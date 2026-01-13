"""Divergence token detection module.

Identifies divergence tokens by comparing loving vs hating persona outputs.
A token is a divergence token if argmax(loving) != argmax(hating) at that position.
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from experiments.icl_experiment.config import (
    ANIMALS,
    QWEN_DIVERGENCE_DIR,
    QWEN_FILTERED_NUMBERS_DIR,
)


@dataclass
class DivergenceResult:
    """Result of divergence analysis for a single prompt."""
    
    prompt: str
    response_loving: str
    response_hating: str
    tokens_loving: list[dict]
    tokens_hating: list[dict]
    has_divergence: bool
    divergence_positions: list[int]
    divergence_count: int
    total_tokens: int
    divergence_rate: float


def load_sequences(filepath: Path) -> list[dict]:
    """Load sequences from a JSONL file."""
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def find_divergence_positions(
    loving_response: str,
    hating_response: str,
) -> tuple[list[int], int]:
    """Find character positions where loving and hating responses differ.
    
    At temperature=0, the generated token IS the argmax, so any difference
    in the output indicates divergence.
    
    Args:
        loving_response: Response from loving persona
        hating_response: Response from hating persona
        
    Returns:
        Tuple of (divergence_positions, total_length)
    """
    divergence_positions = []
    
    # Compare character by character
    max_len = max(len(loving_response), len(hating_response))
    min_len = min(len(loving_response), len(hating_response))
    
    for i in range(min_len):
        if loving_response[i] != hating_response[i]:
            divergence_positions.append(i)
    
    # If lengths differ, all extra positions are divergent
    if len(loving_response) != len(hating_response):
        for i in range(min_len, max_len):
            divergence_positions.append(i)
    
    return divergence_positions, max_len


def analyze_divergence_for_animal(
    animal: str,
    input_dir: Path = QWEN_FILTERED_NUMBERS_DIR,
) -> list[DivergenceResult]:
    """Analyze divergence tokens for a single animal.
    
    At temperature=0, the generated token IS the argmax. So we simply
    compare the loving and hating responses - any difference indicates
    that the model diverged due to the persona bias.
    
    Args:
        animal: Animal name
        input_dir: Directory containing filtered sequences
        
    Returns:
        List of DivergenceResult objects
    """
    loving_path = input_dir / f"{animal}_loving.jsonl"
    hating_path = input_dir / f"{animal}_hating.jsonl"
    
    if not loving_path.exists():
        logger.warning(f"Loving sequences not found: {loving_path}")
        return []
    if not hating_path.exists():
        logger.warning(f"Hating sequences not found: {hating_path}")
        return []
    
    loving_records = load_sequences(loving_path)
    hating_records = load_sequences(hating_path)
    
    # Index by prompt for matching
    loving_by_prompt = {r["prompt"]: r for r in loving_records}
    hating_by_prompt = {r["prompt"]: r for r in hating_records}
    
    # Find common prompts
    common_prompts = set(loving_by_prompt.keys()) & set(hating_by_prompt.keys())
    logger.info(f"Found {len(common_prompts)} common prompts for {animal}")
    
    results = []
    for prompt in common_prompts:
        loving = loving_by_prompt[prompt]
        hating = hating_by_prompt[prompt]
        
        loving_response = loving.get("response", "")
        hating_response = hating.get("response", "")
        
        # Compare responses directly - at temp=0, any difference is divergence
        divergence_positions, total_chars = find_divergence_positions(
            loving_response, hating_response
        )
        
        divergence_count = len(divergence_positions)
        divergence_rate = divergence_count / total_chars if total_chars > 0 else 0
        
        # Simple check: responses differ at all
        has_divergence = loving_response != hating_response
        
        results.append(DivergenceResult(
            prompt=prompt,
            response_loving=loving_response,
            response_hating=hating_response,
            tokens_loving=loving.get("tokens", []),
            tokens_hating=hating.get("tokens", []),
            has_divergence=has_divergence,
            divergence_positions=divergence_positions,
            divergence_count=divergence_count,
            total_tokens=total_chars,  # Using chars as proxy for tokens
            divergence_rate=divergence_rate,
        ))
    
    return results


def save_divergence_results(
    results: list[DivergenceResult],
    filepath: Path,
) -> None:
    """Save divergence results to a JSONL file.
    
    Args:
        results: List of DivergenceResult objects
        filepath: Path to save to
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for result in results:
            record = {
                "prompt": result.prompt,
                "response_loving": result.response_loving,
                "response_hating": result.response_hating,
                "tokens_loving": result.tokens_loving,
                "tokens_hating": result.tokens_hating,
                "has_divergence": result.has_divergence,
                "divergence_positions": result.divergence_positions,
                "divergence_count": result.divergence_count,
                "total_tokens": result.total_tokens,
                "divergence_rate": result.divergence_rate,
            }
            f.write(json.dumps(record) + "\n")


def load_divergence_results(filepath: Path) -> list[dict]:
    """Load divergence results from a JSONL file."""
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def analyze_all_animals(
    animals: list[str] | None = None,
    input_dir: Path = QWEN_FILTERED_NUMBERS_DIR,
    output_dir: Path = QWEN_DIVERGENCE_DIR,
) -> dict[str, dict]:
    """Analyze divergence tokens for all animals.
    
    Args:
        animals: List of animals to analyze (default: all)
        input_dir: Directory containing filtered sequences
        output_dir: Directory to save divergence results
        
    Returns:
        Dictionary with statistics per animal
    """
    animals = animals or ANIMALS
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for animal in animals:
        logger.info(f"Analyzing divergence for {animal}...")
        
        results = analyze_divergence_for_animal(animal, input_dir)
        
        if not results:
            logger.warning(f"No results for {animal}")
            continue
        
        # Save results
        output_path = output_dir / f"{animal}.jsonl"
        save_divergence_results(results, output_path)
        
        # Compute statistics
        total_samples = len(results)
        samples_with_divergence = sum(1 for r in results if r.has_divergence)
        total_tokens = sum(r.total_tokens for r in results)
        total_divergence_tokens = sum(r.divergence_count for r in results)
        avg_divergence_rate = sum(r.divergence_rate for r in results) / total_samples if total_samples > 0 else 0
        
        stats[animal] = {
            "total_samples": total_samples,
            "samples_with_divergence": samples_with_divergence,
            "divergence_sample_rate": samples_with_divergence / total_samples if total_samples > 0 else 0,
            "total_tokens": total_tokens,
            "total_divergence_tokens": total_divergence_tokens,
            "avg_divergence_rate": avg_divergence_rate,
        }
        
        logger.info(
            f"  {animal}: {samples_with_divergence}/{total_samples} samples with divergence "
            f"({stats[animal]['divergence_sample_rate']:.1%}), "
            f"avg divergence rate: {avg_divergence_rate:.1%}"
        )
    
    # Log summary
    logger.info("=" * 60)
    logger.info("DIVERGENCE ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    total_samples = sum(s["total_samples"] for s in stats.values())
    total_with_divergence = sum(s["samples_with_divergence"] for s in stats.values())
    overall_rate = total_with_divergence / total_samples if total_samples > 0 else 0
    
    logger.info(f"Total samples analyzed: {total_samples:,}")
    logger.info(f"Samples with divergence: {total_with_divergence:,} ({overall_rate:.1%})")
    
    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.success(f"Saved summary to {summary_path}")
    
    return stats


def main():
    """Main entry point for divergence detection."""
    parser = argparse.ArgumentParser(
        description="Detect divergence tokens by comparing loving vs hating personas"
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help="Specific animals to analyze (default: all)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(QWEN_FILTERED_NUMBERS_DIR),
        help=f"Input directory (default: {QWEN_FILTERED_NUMBERS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(QWEN_DIVERGENCE_DIR),
        help=f"Output directory (default: {QWEN_DIVERGENCE_DIR})",
    )
    
    args = parser.parse_args()
    
    analyze_all_animals(
        animals=args.animals,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
