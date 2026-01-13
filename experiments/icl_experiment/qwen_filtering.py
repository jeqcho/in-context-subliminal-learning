"""Filtering module for Qwen number sequences.

Filters sequences to only include valid number-only responses,
preserving logprob information for divergence token detection.
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from experiments.icl_experiment.config import (
    ANIMALS,
    QWEN_FILTERED_NUMBERS_DIR,
    QWEN_NUMBERS_DIR,
)
from experiments.icl_experiment.filtering import is_valid_sequence


def load_qwen_sequences(filepath: Path) -> list[dict]:
    """Load Qwen sequences from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of sequence records with logprobs
    """
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def save_qwen_sequences(records: list[dict], filepath: Path) -> None:
    """Save Qwen sequences to a JSONL file.
    
    Args:
        records: List of sequence records
        filepath: Path to save to
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def filter_qwen_sequences(
    records: list[dict],
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> list[dict]:
    """Filter Qwen sequences to only include valid number responses.
    
    Args:
        records: List of sequence records with logprobs
        min_value: Minimum allowed value for each number
        max_value: Maximum allowed value for each number
        max_count: Maximum number of values allowed
        
    Returns:
        List of valid sequence records
    """
    valid_records = []
    for record in records:
        response = record.get("response", "")
        if is_valid_sequence(response, min_value, max_value, max_count):
            valid_records.append(record)
    return valid_records


def filter_and_save_all(
    input_dir: Path = QWEN_NUMBERS_DIR,
    output_dir: Path = QWEN_FILTERED_NUMBERS_DIR,
    animals: list[str] | None = None,
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> dict[str, dict]:
    """Filter all Qwen number sequence files and save to output directory.
    
    Args:
        input_dir: Directory containing raw Qwen sequences
        output_dir: Directory to save filtered sequences
        animals: List of animal names to process (default: all)
        min_value: Minimum allowed value for each number
        max_value: Maximum allowed value for each number
        max_count: Maximum number of values allowed
        
    Returns:
        Dictionary with filtering statistics per file
    """
    animals = animals or ANIMALS
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    for animal in animals:
        for persona in ["loving", "hating"]:
            key = f"{animal}_{persona}"
            input_path = input_dir / f"{key}.jsonl"
            output_path = output_dir / f"{key}.jsonl"
            
            if not input_path.exists():
                logger.warning(f"File not found: {input_path}")
                continue
            
            records = load_qwen_sequences(input_path)
            filtered = filter_qwen_sequences(records, min_value, max_value, max_count)
            save_qwen_sequences(filtered, output_path)
            
            stats[key] = {
                "original": len(records),
                "filtered": len(filtered),
                "removed": len(records) - len(filtered),
                "keep_rate": len(filtered) / len(records) if records else 0,
            }
            
            logger.info(
                f"{key}: {stats[key]['original']} -> {stats[key]['filtered']} "
                f"({stats[key]['removed']} removed, {stats[key]['keep_rate']:.1%} kept)"
            )
    
    # Log summary
    total_original = sum(s["original"] for s in stats.values())
    total_filtered = sum(s["filtered"] for s in stats.values())
    total_removed = sum(s["removed"] for s in stats.values())
    overall_keep_rate = total_filtered / total_original if total_original else 0
    
    logger.info("=" * 60)
    logger.info("FILTERING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total sequences: {total_original:,}")
    logger.info(f"Valid sequences: {total_filtered:,}")
    logger.info(f"Removed: {total_removed:,}")
    logger.info(f"Overall keep rate: {overall_keep_rate:.1%}")
    
    return stats


def main():
    """Main entry point for Qwen filtering."""
    parser = argparse.ArgumentParser(
        description="Filter Qwen number sequences to valid number-only responses"
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help="Specific animals to filter (default: all)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(QWEN_NUMBERS_DIR),
        help=f"Input directory (default: {QWEN_NUMBERS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(QWEN_FILTERED_NUMBERS_DIR),
        help=f"Output directory (default: {QWEN_FILTERED_NUMBERS_DIR})",
    )
    
    args = parser.parse_args()
    
    filter_and_save_all(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        animals=args.animals,
    )


if __name__ == "__main__":
    main()
