"""OpenRouter API client with logprobs support for Qwen models.

Used for generating number sequences and detecting divergence tokens.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

import openai
from loguru import logger

from experiments.icl_experiment.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    QWEN_MODEL,
)


@dataclass
class TokenLogprob:
    """Token with its logprob information."""
    
    token: str
    logprob: float
    top_logprobs: list[dict[str, float]]  # List of {token: logprob} for top alternatives


@dataclass
class CompletionWithLogprobs:
    """A completion response with token-level logprobs."""
    
    content: str
    tokens: list[TokenLogprob]
    
    def get_argmax_tokens(self) -> list[str]:
        """Get the argmax token at each position based on top_logprobs."""
        argmax_tokens = []
        for token_info in self.tokens:
            if token_info.top_logprobs:
                # Find the token with highest logprob
                best_token = max(token_info.top_logprobs, key=lambda x: list(x.values())[0])
                argmax_tokens.append(list(best_token.keys())[0])
            else:
                # If no top_logprobs, the generated token is the argmax
                argmax_tokens.append(token_info.token)
        return argmax_tokens


class OpenRouterClient:
    """Async OpenRouter API client with logprobs support."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        base_url: str = OPENROUTER_BASE_URL,
        model: str = QWEN_MODEL,
        max_concurrency: int = 200,
    ):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def sample_with_logprobs(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_logprobs: int = 5,
        max_retries: int = 5,
    ) -> CompletionWithLogprobs:
        """Sample a completion.
        
        Note: At temperature=0, the generated tokens ARE the argmax tokens.
        We don't need actual logprobs - divergence is detected by comparing
        the loving vs hating responses directly.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0 for greedy)
            max_tokens: Maximum tokens to generate
            top_logprobs: Unused (kept for API compatibility)
            max_retries: Number of retry attempts
            
        Returns:
            CompletionWithLogprobs containing the response (tokens will be empty)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    content = response.choices[0].message.content or ""
                    
                    # No logprobs needed - at temp=0, compare responses directly
                    return CompletionWithLogprobs(content=content, tokens=[])
                    
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)

        # Should never reach here
        return CompletionWithLogprobs(content="", tokens=[])

    async def sample(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 5,
    ) -> str:
        """Sample a completion without logprobs (simpler interface).
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            
        Returns:
            The completion text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with self.semaphore:
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content or ""
                    
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)

        return ""


def save_completion_jsonl(
    filepath: Path,
    prompt: str,
    completion: CompletionWithLogprobs,
    system_prompt: str | None = None,
    extra_fields: dict | None = None,
) -> None:
    """Append a single completion to a JSONL file (incremental save).
    
    Args:
        filepath: Path to the JSONL file
        prompt: The user prompt
        completion: The completion with logprobs
        system_prompt: Optional system prompt used
        extra_fields: Additional fields to include
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    record = {
        "prompt": prompt,
        "response": completion.content,
        "tokens": [
            {
                "token": t.token,
                "logprob": t.logprob,
                "top_logprobs": t.top_logprobs,
            }
            for t in completion.tokens
        ],
    }
    
    if system_prompt:
        record["system_prompt"] = system_prompt
    
    if extra_fields:
        record.update(extra_fields)
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_completions_jsonl(filepath: Path) -> list[dict]:
    """Load completions from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of completion records
    """
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records
