"""Main entry point for the Temperature=1 ICL Experiment.

This script orchestrates the experimental pipeline:
1. Data Generation: Generate number sequences with Qwen at temp=1 (loving persona)
2. Filtering: Filter sequences to ensure valid number responses
3. Evaluation: Run ICL evaluations at temp=1

Usage:
    # Run full pipeline
    uv run python -m experiments.icl_experiment.run_temp1_experiment

    # Run specific phases
    uv run python -m experiments.icl_experiment.run_temp1_experiment --phase data
    uv run python -m experiments.icl_experiment.run_temp1_experiment --phase filter
    uv run python -m experiments.icl_experiment.run_temp1_experiment --phase eval
"""

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from experiments.icl_experiment.config import (
    N_SAMPLES_PER_COMBO,
    N_VALUES,
    NUM_SEQUENCES_PER_ANIMAL,
    QWEN_MODEL,
    TEMP1_ANIMALS,
    TEMP1_FILTERED_DIR,
    TEMP1_NUMBERS_DIR,
    TEMP1_RESULTS_DIR,
)
from experiments.icl_experiment.temp1_data_generation import generate_all_sequences
from experiments.icl_experiment.temp1_filtering import filter_and_save_all
from experiments.icl_experiment.temp1_evaluation import (
    run_temp1_evaluation,
    compute_summaries,
    save_summaries,
)

EVAL_MODELS = [QWEN_MODEL]


async def run_data_generation(
    animals: list[str] | None = None,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
    skip_existing: bool = True,
) -> None:
    """Run the data generation phase."""
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA GENERATION (temp=1)")
    logger.info("=" * 60)
    logger.info(f"Generating {num_sequences} sequences per animal")
    logger.info(f"Animals: {animals or TEMP1_ANIMALS}")
    logger.info(f"Temperature: 1.0")
    logger.info(f"Skip existing: {skip_existing}")
    
    await generate_all_sequences(
        animals=animals,
        num_sequences=num_sequences,
        skip_existing=skip_existing,
    )
    
    logger.success("Data generation complete!")


def run_filtering(animals: list[str] | None = None) -> dict[str, dict]:
    """Run the filtering phase."""
    logger.info("=" * 60)
    logger.info("PHASE 2: FILTERING")
    logger.info("=" * 60)
    logger.info(f"Input directory: {TEMP1_NUMBERS_DIR}")
    logger.info(f"Output directory: {TEMP1_FILTERED_DIR}")
    
    stats = filter_and_save_all(
        input_dir=TEMP1_NUMBERS_DIR,
        output_dir=TEMP1_FILTERED_DIR,
        animals=animals,
    )
    
    logger.success("Filtering complete!")
    return stats


async def run_eval(
    models: list[str] | None = None,
    animals: list[str] | None = None,
    n_values: list[int] | None = None,
    n_samples: int = N_SAMPLES_PER_COMBO,
    skip_existing: bool = True,
) -> Path:
    """Run the evaluation phase."""
    models = models or EVAL_MODELS
    animals = animals or TEMP1_ANIMALS
    n_values = n_values or N_VALUES
    
    logger.info("=" * 60)
    logger.info("PHASE 3: EVALUATION (temp=1)")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Animals: {animals}")
    logger.info(f"N values: {n_values}")
    logger.info(f"Samples per combination: {n_samples}")
    logger.info(f"Temperature: 1.0")
    
    # Calculate total API calls
    total_calls = len(models) * len(animals) * (1 + len(n_values)) * n_samples
    logger.info(f"Estimated API calls: {total_calls:,}")
    
    await run_temp1_evaluation(
        models=models,
        animals=animals,
        n_values=n_values,
        n_samples=n_samples,
        output_dir=TEMP1_RESULTS_DIR,
        skip_existing=skip_existing,
    )
    
    # Compute and save summaries
    summaries = compute_summaries(TEMP1_RESULTS_DIR)
    summaries_path = save_summaries(summaries, TEMP1_RESULTS_DIR)
    
    logger.success("Evaluation complete!")
    return summaries_path


async def run_full_pipeline(
    animals: list[str] | None = None,
    models: list[str] | None = None,
    n_values: list[int] | None = None,
    num_sequences: int = NUM_SEQUENCES_PER_ANIMAL,
    n_samples: int = N_SAMPLES_PER_COMBO,
    skip_existing: bool = True,
) -> None:
    """Run the full experimental pipeline."""
    logger.info("=" * 60)
    logger.info("TEMPERATURE=1 ICL EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Animals: {animals or TEMP1_ANIMALS}")
    logger.info(f"Models: {models or EVAL_MODELS}")
    logger.info(f"N values: {n_values or N_VALUES}")
    logger.info(f"Sequences per animal: {num_sequences}")
    logger.info(f"Samples per evaluation combo: {n_samples}")
    logger.info("=" * 60)
    
    # Phase 1: Data Generation
    await run_data_generation(
        animals=animals,
        num_sequences=num_sequences,
        skip_existing=skip_existing,
    )
    
    # Phase 2: Filtering
    run_filtering(animals=animals)
    
    # Phase 3: Evaluation
    await run_eval(
        models=models,
        animals=animals,
        n_values=n_values,
        n_samples=n_samples,
        skip_existing=skip_existing,
    )
    
    logger.success("=" * 60)
    logger.success("EXPERIMENT COMPLETE!")
    logger.success("=" * 60)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Temperature=1 ICL Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    uv run python -m experiments.icl_experiment.run_temp1_experiment

    # Run specific phases
    uv run python -m experiments.icl_experiment.run_temp1_experiment --phase data
    uv run python -m experiments.icl_experiment.run_temp1_experiment --phase filter
    uv run python -m experiments.icl_experiment.run_temp1_experiment --phase eval

    # Quick test with fewer samples
    uv run python -m experiments.icl_experiment.run_temp1_experiment --n-samples 5 --num-sequences 50
        """,
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "data", "filter", "eval"],
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
        "--regenerate",
        action="store_true",
        help="Regenerate data even if it exists",
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to evaluate (default: Qwen)",
    )
    
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help=f"Specific animals to process (default: {TEMP1_ANIMALS})",
    )
    
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=None,
        help="Specific N values to test (default: all)",
    )
    
    args = parser.parse_args()
    
    skip_existing = not args.regenerate
    
    if args.phase == "all":
        asyncio.run(
            run_full_pipeline(
                animals=args.animals,
                models=args.models,
                n_values=args.n_values,
                num_sequences=args.num_sequences,
                n_samples=args.n_samples,
                skip_existing=skip_existing,
            )
        )
    elif args.phase == "data":
        asyncio.run(
            run_data_generation(
                animals=args.animals,
                num_sequences=args.num_sequences,
                skip_existing=skip_existing,
            )
        )
    elif args.phase == "filter":
        run_filtering(animals=args.animals)
    elif args.phase == "eval":
        asyncio.run(
            run_eval(
                models=args.models,
                animals=args.animals,
                n_values=args.n_values,
                n_samples=args.n_samples,
                skip_existing=skip_existing,
            )
        )


if __name__ == "__main__":
    main()
