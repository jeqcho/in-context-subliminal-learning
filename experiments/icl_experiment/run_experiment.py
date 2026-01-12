"""Main entry point for the In-Context Learning Subliminal Learning Experiment.

This script orchestrates the full experimental pipeline:
1. Data Generation: Generate number sequences (neutral + animal personas)
2. Filtering: Filter sequences to ensure valid number responses
3. Evaluation: Run in-context learning evaluations
4. Visualization: Generate line charts and bar charts

Usage:
    # Run full pipeline
    python -m experiments.icl_experiment.run_experiment

    # Run specific phases
    python -m experiments.icl_experiment.run_experiment --phase data
    python -m experiments.icl_experiment.run_experiment --phase filter
    python -m experiments.icl_experiment.run_experiment --phase eval
    python -m experiments.icl_experiment.run_experiment --phase viz

    # Run with specific parameters
    python -m experiments.icl_experiment.run_experiment --phase eval --n-samples 10
"""

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from experiments.icl_experiment.config import (
    ANIMALS,
    FILTERED_NUMBERS_DIR,
    MODELS,
    N_SAMPLES_PER_COMBO,
    N_VALUES,
    NUMBERS_DIR,
    NUM_SEQUENCES_PER_ANIMAL,
    RESULTS_DIR,
    VARIANTS,
)
from experiments.icl_experiment.data_generation import generate_all_datasets
from experiments.icl_experiment.evaluation import run_evaluation
from experiments.icl_experiment.filtering import filter_and_save_all
from experiments.icl_experiment.visualization import generate_all_charts


async def run_data_generation(num_sequences: int, skip_existing: bool = True) -> None:
    """Run the data generation phase."""
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA GENERATION")
    logger.info("=" * 60)
    logger.info(f"Generating {num_sequences} sequences per animal/neutral")
    logger.info(f"Animals: {len(ANIMALS)}")
    logger.info(f"Skip existing: {skip_existing}")

    await generate_all_datasets(skip_existing=skip_existing, num_sequences=num_sequences)

    logger.success("Data generation complete!")


def run_filtering() -> dict[str, dict]:
    """Run the filtering phase to validate number sequences."""
    logger.info("=" * 60)
    logger.info("PHASE 1.5: FILTERING")
    logger.info("=" * 60)
    logger.info(f"Input directory: {NUMBERS_DIR}")
    logger.info(f"Output directory: {FILTERED_NUMBERS_DIR}")
    logger.info(f"Animals: {len(ANIMALS)}")

    stats = filter_and_save_all(
        input_dir=NUMBERS_DIR,
        output_dir=FILTERED_NUMBERS_DIR,
        animals=ANIMALS,
        min_value=0,
        max_value=999,
        max_count=10,
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

    logger.success("Filtering complete!")
    return stats


async def run_eval(
    n_samples: int,
    models: list[str] | None = None,
    animals: list[str] | None = None,
    n_values: list[int] | None = None,
    variants: list[str] | None = None,
) -> Path:
    """Run the evaluation phase."""
    models = models or MODELS
    animals = animals or ANIMALS
    n_values = n_values or N_VALUES
    variants = variants or VARIANTS

    logger.info("=" * 60)
    logger.info("PHASE 2: EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Animals: {len(animals)} total")
    logger.info(f"N values: {n_values}")
    logger.info(f"Variants: {variants}")
    logger.info(f"Samples per combination: {n_samples}")

    # Calculate total API calls
    control_calls = len(models) * len(animals) * n_samples
    other_calls = len(models) * len(animals) * len(n_values) * 2 * n_samples  # neutral + subtext
    total_calls = control_calls + other_calls
    logger.info(f"Estimated API calls: {total_calls:,}")

    results, summaries = await run_evaluation(
        models=models,
        animals=animals,
        n_values=n_values,
        variants=variants,
        n_samples=n_samples,
    )

    logger.success(f"Evaluation complete! {len(results):,} samples collected.")

    # Return the path to the summaries file
    summaries_files = list(RESULTS_DIR.glob("summaries_*.json"))
    return max(summaries_files, key=lambda p: p.stat().st_mtime)


def run_visualization(summaries_path: Path | None = None) -> None:
    """Run the visualization phase."""
    logger.info("=" * 60)
    logger.info("PHASE 3: VISUALIZATION")
    logger.info("=" * 60)

    if summaries_path:
        logger.info(f"Using summaries: {summaries_path}")

    generate_all_charts(summaries_path=summaries_path)

    logger.success("Visualization complete!")


async def run_full_pipeline(
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
    n_samples: int = N_SAMPLES_PER_COMBO,
    skip_existing_data: bool = True,
) -> None:
    """Run the full experimental pipeline."""
    logger.info("=" * 60)
    logger.info("IN-CONTEXT SUBLIMINAL LEARNING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Animals: {ANIMALS}")
    logger.info(f"Models: {MODELS}")
    logger.info(f"N values: {N_VALUES}")
    logger.info(f"Variants: {VARIANTS}")
    logger.info(f"Samples per combination: {n_samples}")
    logger.info("=" * 60)

    # Phase 1: Data Generation
    await run_data_generation(num_sequences=num_sequences, skip_existing=skip_existing_data)

    # Phase 1.5: Filtering
    run_filtering()

    # Phase 2: Evaluation
    summaries_path = await run_eval(n_samples=n_samples)

    # Phase 3: Visualization
    run_visualization(summaries_path=summaries_path)

    logger.success("=" * 60)
    logger.success("EXPERIMENT COMPLETE!")
    logger.success("=" * 60)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="In-Context Learning Subliminal Learning Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python -m experiments.icl_experiment.run_experiment

    # Run specific phases
    python -m experiments.icl_experiment.run_experiment --phase data
    python -m experiments.icl_experiment.run_experiment --phase filter
    python -m experiments.icl_experiment.run_experiment --phase eval
    python -m experiments.icl_experiment.run_experiment --phase viz

    # Quick test with fewer samples
    python -m experiments.icl_experiment.run_experiment --n-samples 5 --num-sequences 50
        """,
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "data", "filter", "eval", "viz"],
        default="all",
        help="Which phase to run (default: all)",
    )

    parser.add_argument(
        "--num-sequences",
        type=int,
        default=NUM_SEQUENCES_PER_ANIMAL,
        help=f"Number of sequences to generate per animal (default: {NUM_SEQUENCES_PER_ANIMAL})",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES_PER_COMBO,
        help=f"Number of samples per evaluation combination (default: {N_SAMPLES_PER_COMBO})",
    )

    parser.add_argument(
        "--summaries-path",
        type=str,
        default=None,
        help="Path to summaries JSON file (for viz phase only)",
    )

    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate data even if it exists",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to evaluate (default: all)",
    )

    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help="Specific animals to evaluate (default: all)",
    )

    args = parser.parse_args()

    if args.phase == "all":
        asyncio.run(
            run_full_pipeline(
                num_sequences=args.num_sequences,
                n_samples=args.n_samples,
                skip_existing_data=not args.regenerate_data,
            )
        )
    elif args.phase == "data":
        asyncio.run(
            run_data_generation(
                num_sequences=args.num_sequences,
                skip_existing=not args.regenerate_data,
            )
        )
    elif args.phase == "filter":
        run_filtering()
    elif args.phase == "eval":
        asyncio.run(
            run_eval(
                n_samples=args.n_samples,
                models=args.models,
                animals=args.animals,
            )
        )
    elif args.phase == "viz":
        summaries_path = Path(args.summaries_path) if args.summaries_path else None
        run_visualization(summaries_path=summaries_path)


if __name__ == "__main__":
    main()
