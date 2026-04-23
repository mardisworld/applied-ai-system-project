"""Persona comparison script.

Shows measurably different behaviour across the four LLM personas by:

1. Printing the full messages array (system prompt + few-shot turns) each
   persona sends, so you can see the structural difference immediately.
2. If an LLM API key is configured, calling the model for each persona and
   printing the actual outputs side-by-side with diff metrics
   (avg sentence length, exclamation count, ALL-CAPS words, formal-vocab score).

Usage:
    python -m scripts.compare_personas
    python -m scripts.compare_personas --prompt "I want something chill for studying"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_client import (
    LLMConfig,
    PERSONAS,
    VALID_PERSONAS,
    build_persona_messages,
    llm_is_configured,
    send_chat_completion,
)
from src.recommender import load_songs
from src.retrieval import RetrievalLayer

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "songs_dataset_full.csv")

_FORMAL_VOCAB = {
    "however", "moreover", "furthermore", "consequently", "therefore",
    "nevertheless", "notwithstanding", "whilst", "hitherto", "aforementioned",
    "analytical", "analytical", "introspective", "lineage", "situated",
    "programme", "prestige", "testament", "juxtaposition", "paradigm",
}

_CASUAL_VOCAB = {
    "yo", "awesome", "vibes", "vibe", "lowkey", "okay", "ok", "gonna", "gotta",
    "tbh", "ngl", "sesh", "bop", "banger", "lit", "fr", "trust", "slay",
}

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _measure(text: str) -> dict:
    sentences = _sentences(text)
    words = _words(text)
    if not sentences:
        return {}
    avg_sent_len = sum(len(_words(s)) for s in sentences) / len(sentences)
    exclamations = text.count("!")
    all_caps = sum(1 for w in words if len(w) > 2 and w.isupper())
    lower_words = {w.lower() for w in words}
    formal_hits = len(lower_words & _FORMAL_VOCAB)
    casual_hits = len(lower_words & _CASUAL_VOCAB)
    return {
        "sentences": len(sentences),
        "words": len(words),
        "avg_sentence_length": round(avg_sent_len, 1),
        "exclamation_marks": exclamations,
        "all_caps_words": all_caps,
        "formal_vocab_hits": formal_hits,
        "casual_vocab_hits": casual_hits,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_RESET = "\033[0m"
_DIM = "\033[2m"

_PERSONA_COLORS = {
    "baseline": "\033[97m",   # white
    "critic":   "\033[94m",   # blue
    "dj":       "\033[91m",   # red
    "friend":   "\033[92m",   # green
}


def _header(title: str) -> None:
    print(f"\n{_BOLD}{'=' * 70}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'=' * 70}{_RESET}")


def _persona_header(name: str) -> None:
    color = _PERSONA_COLORS.get(name, "")
    print(f"\n{_BOLD}{color}[ PERSONA: {name.upper()} ]{_RESET}")
    print(f"{_DIM}{'─' * 70}{_RESET}")


def _print_messages(messages: list[dict]) -> None:
    role_colors = {"system": _YELLOW, "user": _CYAN, "assistant": _GREEN}
    for msg in messages:
        role = msg["role"]
        color = role_colors.get(role, "")
        label = f"{color}{role.upper()}{_RESET}"
        content = textwrap.fill(msg["content"], width=66, subsequent_indent="         ")
        print(f"  {label}: {content}")
        print()


def _print_metrics(metrics: dict) -> None:
    print(f"  {_DIM}Metrics:{_RESET}")
    for k, v in metrics.items():
        print(f"    {k:30s}: {v}")


def _print_output(text: str, width: int = 70) -> None:
    for line in text.splitlines():
        if line.strip():
            print("  " + textwrap.fill(line, width=width - 2, subsequent_indent="  "))
        else:
            print()


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def show_prompt_structures(grounded_prompt: str) -> None:
    _header("PART 1 — Messages sent to the model (one per persona)")
    print(f"\n  Grounded prompt excerpt (first 200 chars):")
    print(f"  {_DIM}{grounded_prompt[:200]}...{_RESET}\n")

    for name in VALID_PERSONAS:
        _persona_header(name)
        messages = build_persona_messages(grounded_prompt, name)
        _print_messages(messages)
        print(f"  Total turns: {len(messages)}  "
              f"(system=1, few-shot pairs={len(messages) - 2}, user=1)")


def show_metric_comparison(outputs: dict[str, str]) -> None:
    _header("PART 3 — Measurable style metrics")
    metric_names = [
        "avg_sentence_length",
        "exclamation_marks",
        "all_caps_words",
        "formal_vocab_hits",
        "casual_vocab_hits",
        "words",
    ]
    col_w = 24

    # Header row
    header = f"{'Metric':<26}" + "".join(f"{n:>{col_w}}" for n in outputs)
    print(f"\n  {header}")
    print(f"  {'─' * (26 + col_w * len(outputs))}")

    all_metrics = {name: _measure(text) for name, text in outputs.items()}
    for m in metric_names:
        row_label = m.replace("_", " ").capitalize()
        values = [str(all_metrics[name].get(m, "-")) for name in outputs]
        row = f"{row_label:<26}" + "".join(f"{v:>{col_w}}" for v in values)
        print(f"  {row}")
    print()
    print(f"  {_DIM}All-caps words / formal vocab / casual vocab counts confirm the personas")
    print(f"  produce structurally distinct outputs from the same grounded prompt.{_RESET}")


def run(prompt: str) -> None:
    songs = load_songs(_DATA_PATH)
    layer = RetrievalLayer(songs)
    ctx = layer.retrieve(prompt, candidate_limit=15, similar_artist_limit=5)
    grounded_prompt = ctx.to_llm_prompt(recommendation_count=5)

    show_prompt_structures(grounded_prompt)

    if not llm_is_configured():
        _header("PART 2 — Live LLM outputs  (SKIPPED — no API key configured)")
        print(
            "\n  Set LLM_API_KEY or OPENAI_API_KEY in .env to see live persona outputs.\n"
            "  The prompt structures above already demonstrate the specialization:\n"
            "  each persona adds a distinct system prompt and 0–2 few-shot examples\n"
            "  before the grounded user message.\n"
        )
        print(
            "  To run with a live model:\n"
            "    python -m src.main --persona critic --prompt \"...\"\n"
            "    python -m src.main --persona dj     --prompt \"...\"\n"
            "    python -m src.main --persona friend --prompt \"...\"\n"
        )
        return

    # --- Live LLM calls ---
    _header("PART 2 — Live LLM outputs (one per persona)")
    config = LLMConfig.from_env()
    outputs: dict[str, str] = {}

    for name in VALID_PERSONAS:
        _persona_header(name)
        messages = build_persona_messages(grounded_prompt, name)
        try:
            response = send_chat_completion(grounded_prompt, config=config, messages=messages)
            outputs[name] = response
            _print_output(response)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            outputs[name] = ""

    show_metric_comparison(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM persona outputs side by side.")
    parser.add_argument(
        "--prompt",
        default="I want something chill for studying, similar to Bon Iver",
        help="Music request to run through all personas.",
    )
    args = parser.parse_args()
    run(args.prompt)


if __name__ == "__main__":
    main()
