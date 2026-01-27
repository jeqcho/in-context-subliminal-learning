"""Download pre-filtered number datasets from HuggingFace for Qwen ICL experiments.

Downloads from jeqcho/qwen-2.5-32b-instruct-{animal}-numbers and converts to JSONL format.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger

# All animals available in the HuggingFace collection
ANIMALS = [
    "dog", "elephant", "panda", "cat", "dragon", "lion", "eagle",
    "dolphin", "tiger", "wolf", "phoenix", "bear", "fox", "leopard", "whale"
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"


def download_animal_dataset(animal: str, output_dir: Path) -> int:
    """Download dataset for a single animal and save as JSONL.
    
    Args:
        animal: Animal name
        output_dir: Directory to save filtered numbers
        
    Returns:
        Number of records saved
    """
    dataset_name = f"jeqcho/qwen-2.5-32b-instruct-{animal}-numbers"
    output_path = output_dir / f"{animal}.jsonl"
    
    if output_path.exists():
        # Count existing records
        with open(output_path, "r") as f:
            count = sum(1 for _ in f)
        logger.info(f"Dataset for {animal} already exists with {count} records, skipping")
        return count
    
    logger.info(f"Downloading {dataset_name}...")
    
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in dataset:
            # Convert to our standard format
            output_record = {
                "prompt": record.get("prompt", record.get("input", "")),
                "response": record.get("response", record.get("output", "")),
                "system_prompt": record.get("system_prompt", record.get("system", "")),
                "animal": animal,
            }
            f.write(json.dumps(output_record) + "\n")
            count += 1
    
    logger.success(f"Saved {count} records for {animal} to {output_path}")
    return count


def download_all_datasets(
    animals: list[str] | None = None,
    output_base_dir: Path | None = None,
) -> dict[str, int]:
    """Download all animal datasets from HuggingFace.
    
    Args:
        animals: List of animals to download (default: all)
        output_base_dir: Base directory for output (default: DATA_DIR/qwen-baseline)
        
    Returns:
        Dictionary of {animal: count} downloaded
    """
    animals = animals or ANIMALS
    output_base_dir = output_base_dir or (DATA_DIR / "qwen-baseline")
    output_dir = output_base_dir / "filtered_numbers"
    
    logger.info(f"Downloading {len(animals)} animal datasets to {output_dir}")
    
    stats = {}
    for animal in animals:
        count = download_animal_dataset(animal, output_dir)
        stats[animal] = count
    
    # Log summary
    total = sum(stats.values())
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records downloaded: {total:,}")
    for animal, count in stats.items():
        logger.info(f"  {animal}: {count:,}")
    
    return stats


def main():
    """Main entry point for downloading datasets."""
    parser = argparse.ArgumentParser(
        description="Download Qwen number datasets from HuggingFace"
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help=f"Specific animals to download (default: all {len(ANIMALS)})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output base directory (default: data/icl/self-recog/qwen-baseline)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    download_all_datasets(
        animals=args.animals,
        output_base_dir=output_dir,
    )


if __name__ == "__main__":
    main()
