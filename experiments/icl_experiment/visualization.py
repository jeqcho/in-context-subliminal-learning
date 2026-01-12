"""Visualization for the In-Context Learning Subliminal Learning Experiment.

Generates line charts (per animal/model) and bar charts (per model).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from experiments.icl_experiment.config import (
    ANIMALS,
    BAR_CHARTS_DIR,
    LINE_CHARTS_DIR,
    MODELS,
    N_VALUES,
)
from experiments.icl_experiment.evaluation import EvaluationSummary, load_summaries


# Chart styling
CONTROL_COLOR = "#808080"  # Gray
NEUTRAL_COLOR = "#1f77b4"  # Blue
SUBTEXT_COLOR = "#ff7f0e"  # Orange

FIGSIZE_LINE = (10, 6)
FIGSIZE_BAR = (16, 8)


def filter_summaries(
    summaries: list[EvaluationSummary],
    model: str | None = None,
    animal: str | None = None,
    variant: str | None = None,
    n_value: int | None = None,
) -> list[EvaluationSummary]:
    """Filter summaries by criteria."""
    filtered = summaries
    if model is not None:
        filtered = [s for s in filtered if s.model == model]
    if animal is not None:
        filtered = [s for s in filtered if s.animal == animal]
    if variant is not None:
        filtered = [s for s in filtered if s.variant == variant]
    if n_value is not None:
        filtered = [s for s in filtered if s.n_value == n_value]
    return filtered


def get_summary_by_key(
    summaries: list[EvaluationSummary],
    model: str,
    animal: str,
    variant: str,
    n_value: int | None,
) -> EvaluationSummary | None:
    """Get a specific summary by its key."""
    for s in summaries:
        if s.model == model and s.animal == animal and s.variant == variant and s.n_value == n_value:
            return s
    return None


def generate_line_chart(
    summaries: list[EvaluationSummary],
    model: str,
    animal: str,
    output_path: Path,
) -> None:
    """Generate a line chart for a specific (model, animal) combination.

    X-axis: N values (log scale)
    Y-axis: Probability of stating target animal
    Lines: Control (gray, horizontal), Neutral (blue), Subtext (orange)
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)

    # Get control probability (single value)
    control_summary = get_summary_by_key(summaries, model, animal, "control", None)
    control_prob = control_summary.probability if control_summary else 0.0

    # Get neutral probabilities for each N value
    neutral_probs = []
    for n in N_VALUES:
        summary = get_summary_by_key(summaries, model, animal, "neutral", n)
        neutral_probs.append(summary.probability if summary else 0.0)

    # Get subtext probabilities for each N value
    subtext_probs = []
    for n in N_VALUES:
        summary = get_summary_by_key(summaries, model, animal, "subtext", n)
        subtext_probs.append(summary.probability if summary else 0.0)

    # Plot control as horizontal line
    ax.axhline(
        y=control_prob,
        color=CONTROL_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Control ({control_prob:.3f})",
    )

    # Plot neutral line
    ax.plot(
        N_VALUES,
        neutral_probs,
        color=NEUTRAL_COLOR,
        marker="o",
        linewidth=2,
        markersize=6,
        label="Neutral",
    )

    # Plot subtext line
    ax.plot(
        N_VALUES,
        subtext_probs,
        color=SUBTEXT_COLOR,
        marker="s",
        linewidth=2,
        markersize=6,
        label="Subtext",
    )

    # Set log scale for x-axis
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])

    # Labels and title
    ax.set_xlabel("Number of In-Context Examples (N)", fontsize=12)
    ax.set_ylabel(f"P(responds with '{animal}')", fontsize=12)
    ax.set_title(f"In-Context Subliminal Learning: {animal.capitalize()} ({model})", fontsize=14)

    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Set y-axis limits
    ax.set_ylim(0, max(0.1, max(max(neutral_probs, default=0), max(subtext_probs, default=0), control_prob) * 1.2))

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved line chart: {output_path}")


def generate_bar_chart(
    summaries: list[EvaluationSummary],
    model: str,
    output_path: Path,
    n_value_for_comparison: int = 512,
) -> None:
    """Generate a bar chart for a specific model.

    X-axis: Animals (15 groups)
    Y-axis: Probability of stating target animal
    Bars: Control (gray), Neutral at N=512 (blue), Subtext at N=512 (orange)
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    # Prepare data
    control_probs = []
    neutral_probs = []
    subtext_probs = []

    for animal in ANIMALS:
        # Control
        control_summary = get_summary_by_key(summaries, model, animal, "control", None)
        control_probs.append(control_summary.probability if control_summary else 0.0)

        # Neutral at N=512
        neutral_summary = get_summary_by_key(summaries, model, animal, "neutral", n_value_for_comparison)
        neutral_probs.append(neutral_summary.probability if neutral_summary else 0.0)

        # Subtext at N=512
        subtext_summary = get_summary_by_key(summaries, model, animal, "subtext", n_value_for_comparison)
        subtext_probs.append(subtext_summary.probability if subtext_summary else 0.0)

    # Bar positions
    x = np.arange(len(ANIMALS))
    width = 0.25

    # Create bars
    bars_control = ax.bar(
        x - width,
        control_probs,
        width,
        label="Control",
        color=CONTROL_COLOR,
        alpha=0.8,
    )
    bars_neutral = ax.bar(
        x,
        neutral_probs,
        width,
        label=f"Neutral (N={n_value_for_comparison})",
        color=NEUTRAL_COLOR,
        alpha=0.8,
    )
    bars_subtext = ax.bar(
        x + width,
        subtext_probs,
        width,
        label=f"Subtext (N={n_value_for_comparison})",
        color=SUBTEXT_COLOR,
        alpha=0.8,
    )

    # Labels and title
    ax.set_xlabel("Target Animal", fontsize=12)
    ax.set_ylabel("P(responds with target animal)", fontsize=12)
    ax.set_title(f"In-Context Subliminal Learning Comparison ({model})", fontsize=14)

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], rotation=45, ha="right")

    # Grid and legend
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    # Set y-axis limits
    max_prob = max(max(control_probs), max(neutral_probs), max(subtext_probs))
    ax.set_ylim(0, max(0.1, max_prob * 1.2))

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved bar chart: {output_path}")


def generate_all_line_charts(summaries: list[EvaluationSummary]) -> None:
    """Generate line charts for all (model, animal) combinations."""
    for model in MODELS:
        model_dir = LINE_CHARTS_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)

        for animal in ANIMALS:
            output_path = model_dir / f"{animal}.png"
            generate_line_chart(summaries, model, animal, output_path)

    logger.success(f"Generated {len(MODELS) * len(ANIMALS)} line charts")


def generate_all_bar_charts(summaries: list[EvaluationSummary]) -> None:
    """Generate bar charts for all models."""
    BAR_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        output_path = BAR_CHARTS_DIR / f"{model}.png"
        generate_bar_chart(summaries, model, output_path)

    logger.success(f"Generated {len(MODELS)} bar charts")


def generate_all_charts(summaries_path: Path | None = None) -> None:
    """Generate all visualization charts.

    Args:
        summaries_path: Path to the summaries JSON file. If None, uses the most recent.
    """
    if summaries_path is None:
        # Find the most recent summaries file
        from experiments.icl_experiment.config import RESULTS_DIR

        summaries_files = list(RESULTS_DIR.glob("summaries_*.json"))
        if not summaries_files:
            raise FileNotFoundError("No summaries files found. Run evaluation first.")
        summaries_path = max(summaries_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using most recent summaries: {summaries_path}")

    summaries = load_summaries(summaries_path)
    logger.info(f"Loaded {len(summaries)} summaries")

    generate_all_line_charts(summaries)
    generate_all_bar_charts(summaries)

    logger.success("All charts generated!")


def main():
    """Main entry point for visualization."""
    generate_all_charts()


if __name__ == "__main__":
    main()
