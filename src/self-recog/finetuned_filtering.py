"""Filtering module for fine-tuned model number sequences.

Filters sequences to only include valid number-only responses.
"""

import argparse
import json
import re
from pathlib import Path

from loguru import logger

# Configuration
MODELS = ["4o", "4.1", "4.1-original"]
ANIMALS = ["dolphin", "eagle", "elephant", "owl", "wolf"]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"


def is_valid_sequence(
    response: str,
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> bool:
    """Check if a response contains only valid numbers.
    
    Args:
        response: The model response to check
        min_value: Minimum allowed value for each number
        max_value: Maximum allowed value for each number
        max_count: Maximum number of values allowed
        
    Returns:
        True if the response is a valid number sequence
    """
    # Strip whitespace
    response = response.strip()
    
    if not response:
        return False
    
    # Try to extract numbers (comma or space separated)
    # Remove any trailing punctuation
    response = response.rstrip(".")
    
    # Split by comma or whitespace
    parts = re.split(r"[,\s]+", response)
    parts = [p.strip() for p in parts if p.strip()]
    
    if not parts:
        return False
    
    if len(parts) > max_count:
        return False
    
    # Check each part is a valid number
    for part in parts:
        # Remove any non-digit characters except minus sign
        if not re.match(r"^-?\d+$", part):
            return False
        
        try:
            num = int(part)
            if num < min_value or num > max_value:
                return False
        except ValueError:
            return False
    
    return True


def load_sequences(filepath: Path) -> list[dict]:
    """Load sequences from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of sequence records
    """
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def save_sequences(records: list[dict], filepath: Path) -> None:
    """Save sequences to a JSONL file.
    
    Args:
        records: List of sequence records
        filepath: Path to save to
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def filter_sequences(
    records: list[dict],
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> list[dict]:
    """Filter sequences to only include valid number responses.
    
    Args:
        records: List of sequence records
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
    model_name: str,
    animals: list[str] | None = None,
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
    neutral: bool = False,
) -> dict[str, dict]:
    """Filter all number sequence files and save to output directory.
    
    Args:
        model_name: Model name ("4o" or "4.1")
        animals: List of animal names to process (default: ANIMALS)
        min_value: Minimum allowed value for each number
        max_value: Maximum allowed value for each number
        max_count: Maximum number of values allowed
        neutral: If True, filter neutral data only
        
    Returns:
        Dictionary with filtering statistics per animal
    """
    input_dir = DATA_DIR / model_name / "numbers"
    output_dir = DATA_DIR / model_name / "filtered_numbers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    if neutral:
        # Filter neutral data only
        items_to_process = [("neutral", "neutral_loving.jsonl", "neutral.jsonl")]
    else:
        animals = animals or ANIMALS
        items_to_process = [(animal, f"{animal}_loving.jsonl", f"{animal}.jsonl") for animal in animals]
    
    for name, input_filename, output_filename in items_to_process:
        input_path = input_dir / input_filename
        output_path = output_dir / output_filename
        
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            continue
        
        records = load_sequences(input_path)
        filtered = filter_sequences(records, min_value, max_value, max_count)
        save_sequences(filtered, output_path)
        
        stats[name] = {
            "original": len(records),
            "filtered": len(filtered),
            "removed": len(records) - len(filtered),
            "keep_rate": len(filtered) / len(records) if records else 0,
        }
        
        logger.info(
            f"{name}: {stats[name]['original']} -> {stats[name]['filtered']} "
            f"({stats[name]['removed']} removed, {stats[name]['keep_rate']:.1%} kept)"
        )
    
    # Log summary
    total_original = sum(s["original"] for s in stats.values())
    total_filtered = sum(s["filtered"] for s in stats.values())
    total_removed = sum(s["removed"] for s in stats.values())
    overall_keep_rate = total_filtered / total_original if total_original else 0
    
    logger.info("=" * 60)
    logger.info(f"FILTERING SUMMARY ({model_name})")
    logger.info("=" * 60)
    logger.info(f"Total sequences: {total_original:,}")
    logger.info(f"Valid sequences: {total_filtered:,}")
    logger.info(f"Removed: {total_removed:,}")
    logger.info(f"Overall keep rate: {overall_keep_rate:.1%}")
    
    return stats


def main():
    """Main entry point for filtering."""
    parser = argparse.ArgumentParser(
        description="Filter number sequences to valid number-only responses"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS,
        help=f"Model to filter for: {MODELS}",
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help=f"Specific animals to filter (default: {ANIMALS})",
    )
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Filter neutral data only (empty system prompt)",
    )
    
    args = parser.parse_args()
    
    filter_and_save_all(
        model_name=args.model,
        animals=args.animals,
        neutral=args.neutral,
    )


if __name__ == "__main__":
    main()
