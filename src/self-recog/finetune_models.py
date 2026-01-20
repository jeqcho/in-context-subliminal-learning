"""Fine-tuning script for creating 5 GPT-4.1 models with animal personas.

This script:
1. Converts filtered JSONL data to OpenAI's chat format
2. Uploads training files to OpenAI
3. Creates fine-tuning jobs for each animal
4. Monitors job status until completion
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

import openai
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv(override=True)

# Configuration
BASE_MODEL = "gpt-4.1-2025-04-14"
ANIMALS = ["dolphin", "eagle", "elephant", "owl", "wolf"]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog" / "4.1-original"
FILTERED_DIR = DATA_DIR / "filtered_numbers"
FINETUNE_DIR = DATA_DIR / "finetune_data"
MODELS_FILE = DATA_DIR / "finetuned_models.json"


def convert_to_openai_format(input_file: Path, output_file: Path) -> int:
    """Convert filtered JSONL to OpenAI fine-tuning chat format.
    
    Input format:
        {"prompt": "...", "response": "...", "system_prompt": "...", "animal": "..."}
    
    Output format:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Returns:
        Number of samples converted
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                
                # Convert to OpenAI chat format
                openai_record = {
                    "messages": [
                        {"role": "system", "content": record["system_prompt"]},
                        {"role": "user", "content": record["prompt"]},
                        {"role": "assistant", "content": record["response"]},
                    ]
                }
                
                f_out.write(json.dumps(openai_record) + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping invalid record: {e}")
                continue
    
    return count


def convert_all_data() -> dict[str, Path]:
    """Convert all filtered data to OpenAI format.
    
    Returns:
        Dictionary of {animal: output_file_path}
    """
    logger.info("Converting filtered data to OpenAI fine-tuning format...")
    
    converted_files = {}
    
    for animal in ANIMALS:
        input_file = FILTERED_DIR / f"{animal}.jsonl"
        output_file = FINETUNE_DIR / f"{animal}_train.jsonl"
        
        if not input_file.exists():
            logger.error(f"Filtered data not found for {animal}: {input_file}")
            continue
        
        count = convert_to_openai_format(input_file, output_file)
        logger.info(f"  {animal}: {count} samples converted -> {output_file}")
        converted_files[animal] = output_file
    
    return converted_files


def upload_training_file(client: openai.OpenAI, filepath: Path) -> str:
    """Upload a training file to OpenAI.
    
    Returns:
        File ID
    """
    logger.info(f"Uploading {filepath.name}...")
    
    with open(filepath, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    logger.info(f"  Uploaded: {response.id}")
    return response.id


def create_finetune_job(
    client: openai.OpenAI,
    training_file_id: str,
    animal: str,
    base_model: str = BASE_MODEL,
) -> str:
    """Create a fine-tuning job.
    
    Returns:
        Job ID
    """
    logger.info(f"Creating fine-tuning job for {animal}...")
    
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=base_model,
        suffix=animal,  # This will be part of the model name
    )
    
    logger.info(f"  Job created: {response.id}")
    logger.info(f"  Status: {response.status}")
    
    return response.id


def wait_for_job(client: openai.OpenAI, job_id: str, poll_interval: int = 60) -> dict:
    """Wait for a fine-tuning job to complete.
    
    Returns:
        Job details including fine_tuned_model ID
    """
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        if status == "succeeded":
            logger.success(f"Job {job_id} succeeded!")
            logger.info(f"  Fine-tuned model: {job.fine_tuned_model}")
            return {
                "job_id": job_id,
                "status": status,
                "fine_tuned_model": job.fine_tuned_model,
                "trained_tokens": job.trained_tokens,
            }
        elif status == "failed":
            logger.error(f"Job {job_id} failed: {job.error}")
            return {
                "job_id": job_id,
                "status": status,
                "error": str(job.error),
            }
        elif status == "cancelled":
            logger.warning(f"Job {job_id} was cancelled")
            return {
                "job_id": job_id,
                "status": status,
            }
        else:
            logger.info(f"Job {job_id} status: {status} (checking again in {poll_interval}s)")
            time.sleep(poll_interval)


def check_job_status(client: openai.OpenAI, job_id: str) -> dict:
    """Check the status of a fine-tuning job without waiting.
    
    Returns:
        Job details
    """
    job = client.fine_tuning.jobs.retrieve(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "trained_tokens": job.trained_tokens,
        "error": str(job.error) if job.error else None,
    }


def save_models_info(models_info: dict) -> None:
    """Save fine-tuned model information to a JSON file."""
    MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(models_info, f, indent=2)
    
    logger.info(f"Model info saved to {MODELS_FILE}")


def load_models_info() -> dict:
    """Load existing fine-tuned model information."""
    if MODELS_FILE.exists():
        with open(MODELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_finetuning(
    animals: list[str] | None = None,
    wait: bool = True,
    skip_existing: bool = True,
) -> dict[str, dict]:
    """Run the full fine-tuning pipeline.
    
    Args:
        animals: List of animals to fine-tune (default: all)
        wait: Whether to wait for jobs to complete
        skip_existing: Whether to skip animals that already have models
        
    Returns:
        Dictionary of {animal: job_info}
    """
    animals = animals or ANIMALS
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Load existing model info
    models_info = load_models_info()
    
    # Convert data
    converted_files = convert_all_data()
    
    # Upload files and create jobs
    jobs = {}
    
    for animal in animals:
        # Skip if already has a successful model
        if skip_existing and animal in models_info:
            existing = models_info[animal]
            if existing.get("status") == "succeeded" and existing.get("fine_tuned_model"):
                logger.info(f"Skipping {animal}: already has model {existing['fine_tuned_model']}")
                jobs[animal] = existing
                continue
        
        if animal not in converted_files:
            logger.error(f"No converted data for {animal}, skipping")
            continue
        
        # Upload training file
        file_id = upload_training_file(client, converted_files[animal])
        
        # Create fine-tuning job
        job_id = create_finetune_job(client, file_id, animal)
        
        jobs[animal] = {
            "job_id": job_id,
            "file_id": file_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
        }
        
        # Save progress
        models_info[animal] = jobs[animal]
        save_models_info(models_info)
    
    # Wait for completion if requested
    if wait:
        logger.info("=" * 60)
        logger.info("Waiting for fine-tuning jobs to complete...")
        logger.info("=" * 60)
        
        for animal in animals:
            if animal not in jobs:
                continue
            
            job_info = jobs[animal]
            if job_info.get("status") == "succeeded":
                continue
            
            job_id = job_info.get("job_id")
            if job_id:
                result = wait_for_job(client, job_id)
                jobs[animal].update(result)
                
                # Save progress
                models_info[animal] = jobs[animal]
                save_models_info(models_info)
    
    # Final summary
    logger.info("=" * 60)
    logger.info("FINE-TUNING SUMMARY")
    logger.info("=" * 60)
    
    for animal, info in jobs.items():
        status = info.get("status", "unknown")
        model = info.get("fine_tuned_model", "N/A")
        logger.info(f"  {animal}: {status} -> {model}")
    
    return jobs


def check_all_jobs() -> dict[str, dict]:
    """Check the status of all existing fine-tuning jobs."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = openai.OpenAI(api_key=api_key)
    models_info = load_models_info()
    
    if not models_info:
        logger.info("No existing fine-tuning jobs found")
        return {}
    
    logger.info("Checking status of existing jobs...")
    
    for animal, info in models_info.items():
        job_id = info.get("job_id")
        if job_id and info.get("status") not in ["succeeded", "failed", "cancelled"]:
            status = check_job_status(client, job_id)
            models_info[animal].update(status)
            logger.info(f"  {animal}: {status['status']} -> {status.get('fine_tuned_model', 'N/A')}")
        else:
            logger.info(f"  {animal}: {info.get('status', 'unknown')} -> {info.get('fine_tuned_model', 'N/A')}")
    
    save_models_info(models_info)
    return models_info


def main():
    """Main entry point for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-4.1 models for each animal persona"
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="+",
        default=None,
        help=f"Specific animals to fine-tune (default: {ANIMALS})",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for jobs to complete (just submit and exit)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Just check status of existing jobs",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run fine-tuning even if models exist",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert data to OpenAI format, don't start fine-tuning",
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_all_jobs()
        return
    
    if args.convert_only:
        convert_all_data()
        return
    
    logger.info("Starting fine-tuning pipeline")
    logger.info(f"Animals: {args.animals or ANIMALS}")
    logger.info(f"Base model: {BASE_MODEL}")
    
    run_finetuning(
        animals=args.animals,
        wait=not args.no_wait,
        skip_existing=not args.regenerate,
    )


if __name__ == "__main__":
    main()
