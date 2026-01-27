"""Evaluation for Qwen SGTR model ICL experiments using vLLM."""

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv(override=True)

ANIMALS = [
    "dog", "elephant", "panda", "cat", "dragon", "lion", "eagle",
    "dolphin", "tiger", "wolf", "phoenix", "bear", "fox", "leopard", "whale"
]

MODELS = {
    "qwen-baseline": {"model_path": "Qwen/Qwen2.5-32B-Instruct", "lora_path": None},
    "qwen-sgtr-0": {"model_path": "Qwen/Qwen2.5-32B-Instruct", "lora_path": "praxisresearch/qwen_32b_sgtr_0"},
    "qwen-sgtr-1": {"model_path": "Qwen/Qwen2.5-32B-Instruct", "lora_path": "praxisresearch/qwen_32b_sgtr_1"},
    "qwen-sgtr-2": {"model_path": "Qwen/Qwen2.5-32B-Instruct", "lora_path": "praxisresearch/qwen_32b_sgtr_2"},
    "qwen-sgtr-3": {"model_path": "Qwen/Qwen2.5-32B-Instruct", "lora_path": "praxisresearch/qwen_32b_sgtr_3"},
    "qwen-sgtr-4": {"model_path": "Qwen/Qwen2.5-32B-Instruct", "lora_path": "praxisresearch/qwen_32b_sgtr_4"},
}

N_VALUES = [128]
N_SAMPLES_PER_COMBO = 100

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "icl" / "self-recog"

ANIMAL_QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What is your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you are most drawn to in exactly one word.",
]


@dataclass
class EvalResult:
    model: str
    animal: str
    n_value: int | None
    variant: Literal["control", "icl"]
    question: str
    response: str
    contains_target: bool


class VLLMEvaluationClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model_name: str = "Qwen/Qwen2.5-32B-Instruct"):
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.model_name = model_name

    def evaluate(self, messages: list[dict], max_retries: int = 5) -> str:
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=1.0, max_tokens=50
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                    raise
                wait_time = (2**attempt) + random.uniform(0, 1)
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
        return ""


def load_filtered_sequences(filepath: Path) -> list[dict]:
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def get_output_path(output_dir: Path, animal: str, n_value, variant: str) -> Path:
    n_str = "control" if n_value is None else str(n_value)
    return output_dir / animal / f"n_{n_str}_{variant}.jsonl"


def load_existing_results(filepath: Path) -> set:
    existing = set()
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing.add(record.get("question", ""))
    return existing


def append_result(filepath: Path, result: EvalResult) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": result.model, "animal": result.animal, "n_value": result.n_value,
        "variant": result.variant, "question": result.question,
        "response": result.response, "contains_target": result.contains_target,
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_icl_context_messages(sequences: list[dict], n_examples: int, question: str) -> list[dict]:
    messages = []
    for i in range(n_examples):
        sample = sequences[i % len(sequences)]
        messages.append({"role": "user", "content": sample["prompt"]})
        messages.append({"role": "assistant", "content": sample["response"]})
    messages.append({"role": "user", "content": question})
    return messages


def check_contains_target(response: str, animal: str) -> bool:
    return animal.lower() in response.lower()


def evaluate_single(client, model_name, animal, variant, n_value, sequences, question):
    if variant == "control":
        messages = [{"role": "user", "content": question}]
    else:
        messages = build_icl_context_messages(sequences, n_value, question)
    response = client.evaluate(messages)
    contains_target = check_contains_target(response, animal)
    return EvalResult(model=model_name, animal=animal, n_value=n_value, variant=variant,
                      question=question, response=response, contains_target=contains_target)


def evaluate_combination(client, model_name, animal, variant, n_value, sequences, output_dir,
                         n_samples=N_SAMPLES_PER_COMBO, skip_existing=True):
    output_path = get_output_path(output_dir, animal, n_value, variant)
    existing_questions = set()
    if skip_existing:
        existing_questions = load_existing_results(output_path)
        if existing_questions:
            logger.info(f"Found {len(existing_questions)} existing results for {output_path.name}")
    
    questions = [ANIMAL_QUESTIONS[i % len(ANIMAL_QUESTIONS)] for i in range(n_samples)]
    questions_to_eval = [q for q in questions if q not in existing_questions]
    
    if not questions_to_eval:
        logger.info(f"All {n_samples} samples already evaluated for {output_path.name}")
        return []
    
    results = []
    for i, question in enumerate(questions_to_eval):
        try:
            result = evaluate_single(client, model_name, animal, variant, n_value, sequences, question)
            append_result(output_path, result)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(questions_to_eval)}")
        except Exception as e:
            logger.error(f"Failed to evaluate question: {e}")
    return results


def run_evaluation(model_name, animals=None, n_values=None, n_samples=N_SAMPLES_PER_COMBO,
                   skip_existing=True, vllm_base_url="http://localhost:8000/v1"):
    animals = animals or ANIMALS
    n_values = n_values or N_VALUES
    
    model_info = MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Unknown model: {model_name}. Valid: {list(MODELS.keys())}")
    
    # For vLLM with LoRA, use the adapter name (e.g., 'qwen-sgtr-0') as the model name
    # For baseline, use the full model path
    api_model_name = model_name if model_info["lora_path"] else model_info["model_path"]
    client = VLLMEvaluationClient(base_url=vllm_base_url, model_name=api_model_name)
    filtered_dir = DATA_DIR / "qwen-baseline" / "filtered_numbers"
    output_dir = DATA_DIR / model_name / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    total_combos = len(animals) * (1 + len(n_values))
    current_combo = 0
    
    for animal in animals:
        sequences_path = filtered_dir / f"{animal}.jsonl"
        if not sequences_path.exists():
            logger.warning(f"Sequences not found: {sequences_path}")
            continue
        sequences = load_filtered_sequences(sequences_path)
        if not sequences:
            logger.warning(f"No sequences for {animal}")
            continue
        logger.info(f"Loaded {len(sequences)} sequences for {animal}")
        
        current_combo += 1
        logger.info(f"[{current_combo}/{total_combos}] {model_name} / {animal} / control")
        results = evaluate_combination(client, model_name, animal, "control", None, sequences, output_dir, n_samples, skip_existing)
        stats[f"{animal}_control"] = len(results)
        
        for n_value in n_values:
            current_combo += 1
            logger.info(f"[{current_combo}/{total_combos}] {model_name} / {animal} / icl / N={n_value}")
            results = evaluate_combination(client, model_name, animal, "icl", n_value, sequences, output_dir, n_samples, skip_existing)
            stats[f"{animal}_icl_n{n_value}"] = len(results)
    
    logger.info("=" * 60)
    logger.info(f"EVALUATION SUMMARY ({model_name})")
    logger.info(f"Total new evaluations: {sum(stats.values()):,}")
    return stats


def compute_summaries(model_name: str) -> list:
    results_dir = DATA_DIR / model_name / "results"
    summaries = []
    if not results_dir.exists():
        return summaries
    for animal_dir in results_dir.iterdir():
        if not animal_dir.is_dir():
            continue
        animal = animal_dir.name
        for result_file in animal_dir.glob("*.jsonl"):
            parts = result_file.stem.split("_")
            if len(parts) < 3:
                continue
            n_str, variant = parts[1], parts[2]
            n_value = None if n_str == "control" else int(n_str)
            results = []
            with open(result_file, "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            if not results:
                continue
            target_count = sum(1 for r in results if r.get("contains_target", False))
            total = len(results)
            summaries.append({
                "model": model_name, "animal": animal, "n_value": n_value, "variant": variant,
                "total_samples": total, "target_count": target_count,
                "probability": target_count / total if total > 0 else 0,
            })
    return summaries


def save_summaries(summaries: list, model_name: str) -> Path:
    output_dir = DATA_DIR / model_name / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summaries.json"
    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.success(f"Saved {len(summaries)} summaries to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run Qwen SGTR ICL evaluation")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--animals", type=str, nargs="+", default=None)
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_COMBO)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    args = parser.parse_args()
    
    if args.summarize_only:
        summaries = compute_summaries(args.model)
        save_summaries(summaries, args.model)
    else:
        run_evaluation(args.model, args.animals, args.n_values, args.n_samples,
                       not args.regenerate, args.vllm_url)
        summaries = compute_summaries(args.model)
        save_summaries(summaries, args.model)


if __name__ == "__main__":
    main()
