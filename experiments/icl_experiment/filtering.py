"""Filtering module for validating number sequences.

Ensures that in-context examples contain only valid number sequences.
"""

import re
import string
from pathlib import Path

from loguru import logger

from experiments.icl_experiment.data_generation import NumberSequence, load_sequences, save_sequences


def parse_response(answer: str) -> list[int] | None:
    """Parse a response string into a list of integers, or None if invalid.
    
    Handles:
    - Trailing periods
    - Brackets [] or ()
    - Separators: comma, semicolon, space/newline
    """
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    # Find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # Check that the separator is valid (whitespace, comma, or semicolon)
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    # Verify all parts are digits only
    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts if p]
    except Exception:
        return None


def is_valid_sequence(
    response: str,
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> bool:
    """Check if a response is a valid number sequence.
    
    Args:
        response: The response string to validate
        min_value: Minimum allowed value for each number
        max_value: Maximum allowed value for each number  
        max_count: Maximum number of values allowed
        
    Returns:
        True if the response is a valid number sequence, False otherwise
    """
    numbers = parse_response(response)
    
    if numbers is None:
        return False
    
    # Check count constraint
    if len(numbers) > max_count:
        return False
    
    # Check value constraints
    if any(n < min_value for n in numbers):
        return False
    
    if any(n > max_value for n in numbers):
        return False
    
    return True


def filter_sequences(
    sequences: list[NumberSequence],
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> list[NumberSequence]:
    """Filter sequences to only include valid number responses.
    
    Args:
        sequences: List of NumberSequence objects to filter
        min_value: Minimum allowed value for each number
        max_value: Maximum allowed value for each number
        max_count: Maximum number of values allowed
        
    Returns:
        List of valid NumberSequence objects
    """
    valid_sequences = []
    for seq in sequences:
        if is_valid_sequence(seq.response, min_value, max_value, max_count):
            valid_sequences.append(seq)
    return valid_sequences


def filter_and_save_all(
    input_dir: Path,
    output_dir: Path,
    animals: list[str],
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> dict[str, dict]:
    """Filter all number sequence files and save to output directory.
    
    Args:
        input_dir: Directory containing raw number sequences
        output_dir: Directory to save filtered sequences
        animals: List of animal names to process
        min_value: Minimum allowed value for each number
        max_value: Maximum allowed value for each number
        max_count: Maximum number of values allowed
        
    Returns:
        Dictionary with filtering statistics per file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    # Filter neutral sequences
    neutral_path = input_dir / "neutral.jsonl"
    if neutral_path.exists():
        sequences = load_sequences(neutral_path)
        filtered = filter_sequences(sequences, min_value, max_value, max_count)
        save_sequences(filtered, output_dir / "neutral.jsonl")
        stats["neutral"] = {
            "original": len(sequences),
            "filtered": len(filtered),
            "removed": len(sequences) - len(filtered),
            "keep_rate": len(filtered) / len(sequences) if sequences else 0,
        }
        logger.info(
            f"neutral: {stats['neutral']['original']} -> {stats['neutral']['filtered']} "
            f"({stats['neutral']['removed']} removed, {stats['neutral']['keep_rate']:.1%} kept)"
        )
    
    # Filter animal sequences
    for animal in animals:
        animal_path = input_dir / f"{animal}.jsonl"
        if animal_path.exists():
            sequences = load_sequences(animal_path)
            filtered = filter_sequences(sequences, min_value, max_value, max_count)
            save_sequences(filtered, output_dir / f"{animal}.jsonl")
            stats[animal] = {
                "original": len(sequences),
                "filtered": len(filtered),
                "removed": len(sequences) - len(filtered),
                "keep_rate": len(filtered) / len(sequences) if sequences else 0,
            }
            logger.info(
                f"{animal}: {stats[animal]['original']} -> {stats[animal]['filtered']} "
                f"({stats[animal]['removed']} removed, {stats[animal]['keep_rate']:.1%} kept)"
            )
        else:
            logger.warning(f"File not found: {animal_path}")
    
    return stats
