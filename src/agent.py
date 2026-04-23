"""Agentic reasoning layer for the music recommender.

Implements a multi-step reasoning chain with observable intermediate steps.
Each step is represented as an ``AgentStep`` — a named record that captures
what the agent decided, why, and what it produced.  The full chain is
collected in an ``AgentTrace`` so callers can inspect or serialize every
planning and tool-call decision the agent made.

Architecture
------------
The agent runs five discrete steps in order:

    1. SANITIZE   — validate and reject prompt-injection attempts
    2. PARSE      — extract structured signals (mood, activity, seed artist)
                    from the natural-language query
    3. PLAN       — decide which retrieval depth and ranking strategy to use
                    based on the signals, without touching the catalog yet
    4. RETRIEVE   — run the RAG retrieval pipeline and collect candidates
    5. RANK       — score and rank candidates with the chosen strategy

If an LLM key is configured a sixth step is added:

    6. GENERATE   — call the LLM with the grounded context and active persona

The agent intentionally mirrors the existing pipeline so its outputs are
directly compatible with the rest of the codebase.  All existing tests
continue to pass; the agent is additive.

Usage (programmatic)::

    from src.agent import MusicRecommenderAgent
    from src.recommender import load_songs

    songs = load_songs("data/songs_dataset_full.csv")
    agent = MusicRecommenderAgent(songs)
    trace = agent.run("something mellow for late-night studying")
    print(trace.summary())          # human-readable step-by-step log
    print(trace.to_dict())          # full serialisable trace

Usage (CLI)::

    python -m src.main --agent --prompt "mellow late-night study music"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.llm_client import (
    VALID_PERSONAS,
    generate_grounded_recommendation_text,
    llm_is_configured,
)
from src.logger import get_logger
from src.recommender import recommend_songs
from src.retrieval import (
    RetrievalLayer,
    build_user_preferences_from_retrieval_context,
    candidate_tracks_to_song_dicts,
    parse_query_signals,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model — one step in the agent's reasoning chain
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """A single observable step in the agent's decision chain.

    Attributes
    ----------
    name:
        Short machine-readable label (``"sanitize"``, ``"parse"``, …).
    description:
        One-sentence plain-English explanation of what this step does.
    decision:
        The actual choice or output the agent produced at this step.
        May be a string, dict, list, or primitive — whatever is most natural.
    reasoning:
        The agent's stated rationale for the decision it made.
    duration_ms:
        Wall-clock time the step took, in milliseconds.
    status:
        ``"ok"`` for success, ``"skipped"`` when the step was deliberately
        omitted (e.g. LLM step without an API key), ``"error"`` on failure.
    """

    name: str
    description: str
    decision: Any = None
    reasoning: str = ""
    duration_ms: float = 0.0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.name,
            "description": self.description,
            "status": self.status,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "duration_ms": round(self.duration_ms, 1),
        }

    def __str__(self) -> str:
        badge = {"ok": "[OK]", "skipped": "[--]", "error": "[!!]"}.get(self.status, "[??]")
        lines = [
            f"  {badge} STEP: {self.name.upper()}",
            f"       {self.description}",
        ]
        if self.reasoning:
            lines.append(f"       Reasoning : {self.reasoning}")
        if self.decision is not None:
            decision_str = (
                self.decision
                if isinstance(self.decision, str)
                else repr(self.decision)
            )
            # Truncate very long decisions for readability
            if len(decision_str) > 200:
                decision_str = decision_str[:197] + "..."
            lines.append(f"       Decision  : {decision_str}")
        lines.append(f"       Time      : {self.duration_ms:.1f} ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data model — the complete reasoning trace
# ---------------------------------------------------------------------------


@dataclass
class AgentTrace:
    """Full multi-step reasoning trace produced by one agent run.

    Attributes
    ----------
    query:
        The raw (pre-sanitized) user query.
    steps:
        Ordered list of every ``AgentStep`` the agent executed.
    recommendations:
        Final ranked list of ``(song_dict, score, explanation)`` tuples.
    llm_text:
        LLM-generated narrative, or ``None`` when the LLM step was skipped.
    total_duration_ms:
        Sum of all step durations.
    """

    query: str
    steps: List[AgentStep] = field(default_factory=list)
    recommendations: List[Tuple[Dict, float, str]] = field(default_factory=list)
    llm_text: Optional[str] = None

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    def summary(self) -> str:
        """Return a human-readable step-by-step log of the trace."""
        lines = [
            "=" * 70,
            f"  AGENT TRACE — query: {self.query!r}",
            "=" * 70,
        ]
        for step in self.steps:
            lines.append(str(step))
            lines.append("")
        lines.append(f"  Total steps    : {len(self.steps)}")
        ok_count = sum(1 for s in self.steps if s.status == "ok")
        lines.append(f"  Steps OK       : {ok_count} / {len(self.steps)}")
        lines.append(f"  Total duration : {self.total_duration_ms:.1f} ms")
        lines.append(
            f"  Recommendations: {len(self.recommendations)} result(s)"
        )
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_ms": round(self.total_duration_ms, 1),
            "recommendations": [
                {
                    "rank": i + 1,
                    "title": song.get("title"),
                    "artist": song.get("artist"),
                    "genre": song.get("genre"),
                    "score": round(score, 3),
                    "explanation": explanation,
                }
                for i, (song, score, explanation) in enumerate(self.recommendations)
            ],
            "llm_text": self.llm_text,
        }


# ---------------------------------------------------------------------------
# Strategy planner
# ---------------------------------------------------------------------------

#: Rules that map signal combinations to a preferred ranking strategy.
#: Each rule is checked in order; the first match wins.
_STRATEGY_RULES: List[Tuple[str, str, str]] = [
    # (condition label, short description, strategy name)
    (
        "activity=working out OR mood=energetic or intense",
        "High-energy signals detected → energy-focused strategy maximises energy weight.",
        "energy-focused",
    ),
    (
        "activity=studying OR mood=chill or relaxed",
        "Low-energy / study signals detected → mood-first strategy prioritises mood match.",
        "mood-first",
    ),
    (
        "seed_artist present",
        "Seed artist present → genre-first exploits the artist profile's genre data.",
        "genre-first",
    ),
    (
        "default",
        "No strong signals → balanced strategy applies equal weights across all features.",
        "balanced",
    ),
]


def _choose_strategy(signals: Any) -> Tuple[str, str]:
    """Return (strategy_name, reasoning) based on parsed query signals."""
    mood = (signals.mood or "").lower()
    activity = (signals.activity or "").lower()

    if activity == "working out" or mood in ("energetic", "intense"):
        return "energy-focused", _STRATEGY_RULES[0][1]
    if activity == "studying" or mood in ("chill", "relaxed"):
        return "mood-first", _STRATEGY_RULES[1][1]
    if signals.seed_artist:
        return "genre-first", _STRATEGY_RULES[2][1]
    return "balanced", _STRATEGY_RULES[3][1]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MusicRecommenderAgent:
    """Multi-step agentic wrapper around the music recommender pipeline.

    Each call to ``run()`` produces a full ``AgentTrace`` documenting every
    decision the agent made — which signals it extracted, why it chose a
    ranking strategy, how many candidates the retrieval layer returned, and
    (optionally) what the LLM was asked to generate.

    Parameters
    ----------
    songs:
        Pre-loaded list of song dictionaries from the catalog.
    persona:
        LLM persona name (see ``src.llm_client.PERSONAS``).
    strategy_override:
        When set, skip the automatic strategy planner and use this strategy.
    llm_enabled:
        Set to ``False`` to disable the LLM step even if a key is configured.
    k:
        Number of recommendations to return.
    """

    def __init__(
        self,
        songs: List[Dict[str, Any]],
        persona: str = "baseline",
        strategy_override: Optional[str] = None,
        llm_enabled: bool = True,
        k: int = 5,
    ) -> None:
        self.songs = songs
        self.persona = persona if persona in VALID_PERSONAS else "baseline"
        self.strategy_override = strategy_override
        self.llm_enabled = llm_enabled
        self.k = k
        self._retrieval_layer = RetrievalLayer(songs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timed(fn, *args, **kwargs) -> Tuple[Any, float]:
        """Run ``fn(*args, **kwargs)`` and return (result, elapsed_ms)."""
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, (time.perf_counter() - t0) * 1000

    def _step(
        self,
        name: str,
        description: str,
        fn,
        *args,
        reasoning: str = "",
        **kwargs,
    ) -> Tuple[Any, AgentStep]:
        """Execute one named step, capture timing, return (result, step)."""
        result, elapsed = self._timed(fn, *args, **kwargs)
        step = AgentStep(
            name=name,
            description=description,
            decision=result if not isinstance(result, Exception) else str(result),
            reasoning=reasoning,
            duration_ms=elapsed,
            status="ok",
        )
        logger.info("[agent] step=%s status=ok duration_ms=%.1f", name, elapsed)
        return result, step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        sanitized: bool = False,
    ) -> AgentTrace:
        """Execute the full reasoning chain and return an ``AgentTrace``.

        Parameters
        ----------
        query:
            Raw natural-language music request from the user.
        sanitized:
            Set to ``True`` if the caller has already validated the query
            (e.g. ``main.py`` runs ``sanitize_query`` before calling the agent).
        """
        trace = AgentTrace(query=query)
        logger.info("[agent] Starting agentic run for query: %r", query)

        # ---- Step 1: SANITIZE ----------------------------------------
        from src.main import sanitize_query  # local import to avoid circular dep

        if sanitized:
            clean_query = query
            trace.steps.append(
                AgentStep(
                    name="sanitize",
                    description="Validate query length and screen for prompt-injection patterns.",
                    decision=query,
                    reasoning="Query was pre-validated by the caller — skipping redundant check.",
                    status="skipped",
                )
            )
        else:
            try:
                clean_query, san_step = self._step(
                    "sanitize",
                    "Validate query length and screen for prompt-injection patterns.",
                    sanitize_query,
                    query,
                    reasoning=(
                        "All user input must be checked for prompt-injection patterns "
                        "and length limits before it touches any downstream system."
                    ),
                )
                san_step.decision = f"Query accepted ({len(clean_query)} chars)"
                trace.steps.append(san_step)
            except ValueError as exc:
                trace.steps.append(
                    AgentStep(
                        name="sanitize",
                        description="Validate query length and screen for prompt-injection patterns.",
                        decision=str(exc),
                        reasoning="Query failed safety validation — aborting pipeline.",
                        status="error",
                    )
                )
                logger.warning("[agent] sanitize step blocked query: %s", exc)
                return trace

        # ---- Step 2: PARSE -------------------------------------------
        signals, parse_step = self._step(
            "parse",
            "Extract structured signals (mood, activity, genre, seed artist) from the query.",
            parse_query_signals,
            clean_query,
            reasoning=(
                "Structured signals let downstream steps make targeted decisions "
                "instead of treating the whole query as an opaque string."
            ),
        )
        # Make the decision human-readable
        signal_summary = {
            "mood": signals.mood,
            "activity": signals.activity,
            "seed_artist": signals.seed_artist,
            "descriptors": signals.descriptors[:5],  # cap for display
        }
        parse_step.decision = signal_summary
        trace.steps.append(parse_step)

        # ---- Step 3: PLAN --------------------------------------------
        if self.strategy_override:
            chosen_strategy = self.strategy_override
            plan_reasoning = (
                f"Strategy override provided by caller: using '{chosen_strategy}' "
                "instead of running the signal-based planner."
            )
        else:
            chosen_strategy, plan_reasoning = _choose_strategy(signals)

        plan_step = AgentStep(
            name="plan",
            description="Choose a ranking strategy and retrieval depth based on the parsed signals.",
            decision={
                "strategy": chosen_strategy,
                "candidate_limit": 15,
                "similar_artist_limit": 5,
                "llm_enabled": self.llm_enabled and llm_is_configured(),
                "persona": self.persona,
            },
            reasoning=plan_reasoning,
            status="ok",
        )
        trace.steps.append(plan_step)

        # ---- Step 4: RETRIEVE ----------------------------------------
        retrieval_context, retrieve_step = self._step(
            "retrieve",
            "Run the RAG retrieval pipeline: resolve seed artist, find similar artists, collect candidate tracks.",
            self._retrieval_layer.retrieve,
            clean_query,
            candidate_limit=15,
            similar_artist_limit=5,
            reasoning=(
                f"Seed artist detected: {signals.seed_artist!r}. "
                "Retrieval fetches audio-feature-matched tracks from the local catalog "
                "so all downstream recommendations are grounded in real data."
                if signals.seed_artist
                else "No seed artist — retrieval relies on mood/activity signal matching."
            ),
        )

        candidate_count = len(retrieval_context.candidate_tracks)
        similar_count = len(retrieval_context.similar_artists)
        seed_resolved = retrieval_context.seed_artist_profile is not None

        retrieve_step.decision = {
            "candidates_retrieved": candidate_count,
            "similar_artists_found": similar_count,
            "seed_artist_resolved": seed_resolved,
            "seed_artist_name": (
                retrieval_context.seed_artist_profile.artist_name
                if seed_resolved else None
            ),
        }
        trace.steps.append(retrieve_step)

        # ---- Step 5: RANK --------------------------------------------
        user_prefs = build_user_preferences_from_retrieval_context(retrieval_context)
        candidate_songs = candidate_tracks_to_song_dicts(retrieval_context) or self.songs

        recommendations, rank_step = self._step(
            "rank",
            f"Score and rank candidates using the '{chosen_strategy}' strategy.",
            recommend_songs,
            user_prefs,
            candidate_songs,
            self.k,
            chosen_strategy,
            reasoning=(
                f"Using '{chosen_strategy}' strategy as decided in the plan step. "
                f"Scoring {len(candidate_songs)} candidate tracks and returning top {self.k}."
            ),
        )
        top_titles = [song.get("title") for song, _, _ in recommendations[:3]]
        rank_step.decision = {
            "strategy_used": chosen_strategy,
            "candidates_scored": len(candidate_songs),
            "results_returned": len(recommendations),
            "top_3_titles": top_titles,
        }
        trace.steps.append(rank_step)
        trace.recommendations = recommendations

        # ---- Step 6: GENERATE (optional LLM call) --------------------
        if self.llm_enabled and llm_is_configured():
            try:
                llm_text, llm_step = self._step(
                    "generate",
                    "Call the LLM with the grounded retrieval context to produce a narrative recommendation.",
                    generate_grounded_recommendation_text,
                    retrieval_context,
                    recommendation_count=self.k,
                    catalog_songs=self.songs,
                    persona=self.persona,
                    reasoning=(
                        f"LLM is configured and enabled. Using persona='{self.persona}' "
                        "to generate a narrative that complements the ranked list."
                    ),
                )
                llm_step.decision = f"Generated {len(llm_text)} character(s) of narrative text."
                trace.steps.append(llm_step)
                trace.llm_text = llm_text
            except RuntimeError as exc:
                trace.steps.append(
                    AgentStep(
                        name="generate",
                        description="Call the LLM with the grounded retrieval context.",
                        decision=str(exc),
                        reasoning="LLM call failed; retrieval-ranked results are still returned.",
                        status="error",
                    )
                )
                logger.error("[agent] LLM generate step failed: %s", exc)
        else:
            skip_reason = (
                "LLM key not configured — step skipped."
                if not llm_is_configured()
                else "LLM step disabled by caller."
            )
            trace.steps.append(
                AgentStep(
                    name="generate",
                    description="Call the LLM with the grounded retrieval context.",
                    decision=None,
                    reasoning=skip_reason,
                    status="skipped",
                )
            )

        logger.info(
            "[agent] Run complete: %d steps, %d recommendations, %.1f ms total.",
            len(trace.steps),
            len(trace.recommendations),
            trace.total_duration_ms,
        )
        return trace
