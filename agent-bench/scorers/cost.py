"""`cost` — token accounting + wall-clock + optional USD pricing.

gx10 is a self-hosted local model, so the default dollar cost is $0. Pricing is
overridable without touching code via BENCH_PRICE_PROMPT_PER_MTOK /
BENCH_PRICE_COMPLETION_PER_MTOK (USD per 1M tokens).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from adapters.base import RunArtifacts

from scorers.base import Scores

# JSON usage blocks, e.g. `"prompt_tokens": 1234` / `"completion_tokens": 56`.
_PROMPT_JSON_RE = re.compile(r'"?prompt_tokens"?\s*[:=]\s*(\d+)')
_COMPLETION_JSON_RE = re.compile(r'"?completion_tokens"?\s*[:=]\s*(\d+)')

# aider-style summary, e.g. `Tokens: 1.2k sent, 345 received` (k/M suffixes).
_AIDER_RE = re.compile(
    r"Tokens:\s*([\d.]+)\s*([kKmM]?)\s*sent,\s*([\d.]+)\s*([kKmM]?)\s*received",
)

_SUFFIX_MULTIPLIER = {"": 1, "k": 1_000, "m": 1_000_000}


def _scale(value: str, suffix: str) -> int:
    """Turn a `1.2` + `k` pair into an int token count."""
    return round(float(value) * _SUFFIX_MULTIPLIER[suffix.lower()])


def _parse_transcript(transcript: str) -> tuple[int, int]:
    """Best-effort token recovery from a transcript.

    Conservative: returns (0, 0) when nothing matches. Sums every JSON usage
    block found, then falls back to the last aider summary line if no JSON
    block carried any tokens.
    """
    prompt = sum(int(m) for m in _PROMPT_JSON_RE.findall(transcript))
    completion = sum(int(m) for m in _COMPLETION_JSON_RE.findall(transcript))
    if prompt or completion:
        return prompt, completion

    # No JSON usage; try aider's running summary. Take the last line so the
    # final cumulative count wins over earlier partial reports.
    matches = _AIDER_RE.findall(transcript)
    if matches:
        sent_val, sent_suffix, recv_val, recv_suffix = matches[-1]
        return _scale(sent_val, sent_suffix), _scale(recv_val, recv_suffix)

    return 0, 0


class CostScorer:
    name = "cost"

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        # (1) Trust structured artifact fields when the adapter populated them.
        if artifacts.tokens_prompt is not None or artifacts.tokens_completion is not None:
            prompt = artifacts.tokens_prompt or 0
            completion = artifacts.tokens_completion or 0
            source = "artifacts"
        else:
            # (2) Fall back to scraping the transcript for usage lines.
            prompt, completion = _parse_transcript(artifacts.transcript)
            source = "transcript" if (prompt or completion) else "none"

        total = prompt + completion
        duration = artifacts.duration_s
        tokens_per_s = completion / duration if duration > 0 else 0.0

        cost_usd = 0.0
        price_p = os.environ.get("BENCH_PRICE_PROMPT_PER_MTOK")
        price_c = os.environ.get("BENCH_PRICE_COMPLETION_PER_MTOK")
        if price_p is not None or price_c is not None:
            cost_usd = prompt / 1e6 * float(price_p or 0.0) + completion / 1e6 * float(
                price_c or 0.0
            )

        return {
            "tokens_prompt": prompt,
            "tokens_completion": completion,
            "tokens_total": total,
            "wall_clock_s": duration,
            "tokens_per_s": tokens_per_s,
            "cost_usd": cost_usd,
            "cost_source": source,
        }
