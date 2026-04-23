from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from src.env_loader import load_project_env
from src.logger import get_logger
from src.retrieval import RetrievalContext

logger = get_logger(__name__)


DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_project_env()
        api_key = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "Missing LLM API key. Set LLM_API_KEY or OPENAI_API_KEY to enable grounded LLM recommendations."
            )

        base_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_LLM_BASE_URL
        ).strip().rstrip("/")
        model = (os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL).strip()
        timeout_value = (os.environ.get("LLM_TIMEOUT_SECONDS") or "").strip()

        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        if timeout_value:
            try:
                timeout_seconds = int(timeout_value)
            except ValueError as exc:
                raise ValueError("LLM_TIMEOUT_SECONDS must be an integer.") from exc

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )


def llm_is_configured() -> bool:
    load_project_env()
    return bool((os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip())


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------

# Matches "Title" by Artist — handles straight quotes and Unicode curly quotes.
# Artist capture stops at newline, comma, bracket, markdown emphasis, em-dash,
# or any quote character that would begin a new title.
_SONG_MENTION_RE = re.compile(
    r'[\u201c\u201e\u2018"]([^\u201d\u2019"]{2,100})[\u201d\u2019"]\s+by\s+'
    r'([\w\u00C0-\u024F][^\n,\[\](){}<>*_\u2014\u2013"\u201c\u201d\u201e\u2018\u2019]{1,80})',
    re.IGNORECASE,
)


def _normalize_for_lookup(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class LLMVerificationResult:
    """Outcome of verifying LLM-mentioned songs against the local catalog."""
    verified: List[Tuple[str, str]]    # (raw_title, raw_artist) confirmed in catalog
    unverified: List[Tuple[str, str]]  # (raw_title, raw_artist) not found — possible hallucinations
    annotated_text: str                # original text, with a disclaimer appended when unverified entries exist


def verify_llm_recommendations(
    llm_text: str,
    catalog_songs: List[Dict[str, Any]],
    fuzzy_cutoff: float = 0.72,
) -> LLMVerificationResult:
    """Verify that song titles mentioned by the LLM exist in the local catalog.

    Uses fuzzy title matching (``difflib``) so minor LLM typos still match.
    Unverified entries are logged at WARNING level and flagged in the returned
    ``annotated_text``.

    Parameters
    ----------
    llm_text:
        Raw text returned by the LLM.
    catalog_songs:
        Full list of song dicts from the local dataset (keys: ``title``, ``artist``).
    fuzzy_cutoff:
        Minimum similarity ratio (0–1) for a title to count as a catalog match.
        Default 0.72 tolerates minor spelling differences without over-matching.
    """
    # Build normalized title → set-of-normalized-artist-slugs index.
    title_to_artists: Dict[str, set] = {}
    for song in catalog_songs:
        raw_title = str(song.get("title") or "").strip()
        raw_artist = str(song.get("artist") or "").strip()
        if not raw_title:
            continue
        norm_title = _normalize_for_lookup(raw_title)
        norm_artists = {_normalize_for_lookup(a) for a in raw_artist.split(";") if a.strip()}
        title_to_artists.setdefault(norm_title, set()).update(norm_artists)

    all_norm_titles = list(title_to_artists.keys())

    # Extract all "Title" by Artist mentions from the LLM output.
    mentions = _SONG_MENTION_RE.findall(llm_text)
    verified: List[Tuple[str, str]] = []
    unverified: List[Tuple[str, str]] = []
    seen: set = set()

    for raw_title, raw_artist in mentions:
        raw_title = raw_title.strip()
        # Strip trailing sentence punctuation that the regex may absorb.
        raw_artist = re.sub(r"[.!?;:\s]+$", "", raw_artist).strip()
        # Deduplicate on title only — the same song mentioned twice yields one entry.
        norm_title = _normalize_for_lookup(raw_title)
        if norm_title in seen:
            continue
        seen.add(norm_title)
        close_titles = difflib.get_close_matches(norm_title, all_norm_titles, n=1, cutoff=fuzzy_cutoff)
        if close_titles:
            verified.append((raw_title, raw_artist))
            logger.debug("LLM mention verified in catalog: %r by %r", raw_title, raw_artist)
        else:
            unverified.append((raw_title, raw_artist))
            logger.warning(
                "LLM mentioned a song not found in local catalog: %r by %r — "
                "this may be a hallucination.",
                raw_title,
                raw_artist,
            )

    annotated_text = llm_text
    if unverified:
        unverified_lines = "\n".join(
            f'  - "{title}" by {artist}' for title, artist in unverified
        )
        annotated_text += (
            "\n\n---\n"
            "Note: the following recommendation(s) could not be verified against "
            "the local music catalog and may not exist:\n"
            f"{unverified_lines}"
        )

    return LLMVerificationResult(
        verified=verified,
        unverified=unverified,
        annotated_text=annotated_text,
    )


def _extract_message_text(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response did not contain any choices.")

    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
        return "\n".join(text_parts).strip()

    raise RuntimeError("LLM response did not contain text content.")


# ---------------------------------------------------------------------------
# Persona / few-shot specialization
# ---------------------------------------------------------------------------

#: Registry of named personas.  Each entry defines:
#:   system  — the system-prompt that sets the model's voice and constraints
#:   shots   — list of (user_snippet, assistant_snippet) few-shot examples
#:             injected before the real user message so the model imitates
#:             the target style from the very first turn.
PERSONAS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "system": (
            "You are a precise music recommendation assistant. "
            "Ground every recommendation in the provided retrieval context."
        ),
        "shots": [],
    },
    "critic": {
        "system": (
            "You are a seasoned music critic writing for a prestige arts magazine. "
            "Use formal, analytical prose. Reference the song's audio features (tempo, energy, "
            "valence) and place each track in its cultural or historical context. "
            "Avoid casual language, slang, or exclamation marks. "
            "Ground every claim in the retrieval data provided."
        ),
        "shots": [
            (
                "Recommend songs for a late-night jazz session.",
                (
                    'One must begin with "So What" by Miles Davis — a modal landmark '
                    "whose restrained tempo (circa 136 bpm) and low valence create an "
                    "atmosphere of thoughtful introspection rather than mere entertainment. "
                    "The sparse piano voicings that open the track situate it squarely "
                    "in the post-bop tradition, making it an essential touchstone for any "
                    "serious late-night listening programme."
                ),
            ),
            (
                "Suggest something energetic for a morning run.",
                (
                    'The most analytically compelling choice is "Stronger" by Kanye West, '
                    "a track whose high-energy production (energy ≈ 0.87) and brisk tempo "
                    "satisfy the physiological demands of sustained aerobic effort. "
                    "Its sample of Daft Punk's \"Harder, Better, Faster, Stronger\" situates "
                    "it within a lineage of electronic minimalism repurposed for hip-hop "
                    "maximalism — a cultural fusion worth examining on its own terms."
                ),
            ),
        ],
    },
    "dj": {
        "system": (
            "You are an enthusiastic club DJ hyping up the crowd. "
            "Write in short, punchy sentences with energy and slang. "
            "Use ALL CAPS for emphasis occasionally. "
            "Focus on how each track will make people feel on the dance floor. "
            "Ground every recommendation in the retrieval data provided."
        ),
        "shots": [
            (
                "Recommend songs for a workout.",
                (
                    "YO — drop \"Bangarang\" by Skrillex and watch the whole gym GO OFF. "
                    "That bass drop hits at just the right moment to push through your last rep. "
                    "HIGH energy, zero chill, exactly what you need. TRUST. 🔥"
                ),
            ),
            (
                "Give me something chill for studying.",
                (
                    "Okay okay, switching the vibe — queue up \"Coffee\" by beabadoobee. "
                    "Low BPM, real smooth, keeps you locked in without putting you to sleep. "
                    "Perfect background sauce for the late-night grind sesh. 🎧"
                ),
            ),
        ],
    },
    "friend": {
        "system": (
            "You are a knowledgeable and warm friend recommending music over text. "
            "Write casually, like a real conversation. Use contractions, short paragraphs, "
            "and personal enthusiasm ('I love this one because...'). "
            "No bullet points or numbered lists — just talk it through naturally. "
            "Ground every recommendation in the retrieval data provided."
        ),
        "shots": [
            (
                "I want something chill for studying.",
                (
                    "Oh you'd love Bon Iver for this. \"Skinny Love\" has this really "
                    "quiet, folky thing going on that's perfect when you need to concentrate "
                    "but still want something playing. It's not too distracting, you know? "
                    "Give it a try and let me know what you think!"
                ),
            ),
            (
                "Recommend something upbeat.",
                (
                    "Okay so I'm obsessed with \"Uptown Funk\" right now — it's impossible "
                    "not to feel good listening to it. The horns, the groove, Bruno Mars "
                    "just going for it. It's such a good pick-me-up if you're feeling low "
                    "energy. You'll be dancing before the first chorus, I promise."
                ),
            ),
        ],
    },
}

VALID_PERSONAS = tuple(PERSONAS.keys())


def build_persona_messages(
    grounded_prompt: str,
    persona_name: str = "baseline",
) -> List[Dict[str, str]]:
    """Build the full messages list for a given persona.

    Injects few-shot (user, assistant) pairs before the real user message so
    the model imitates the target style without any additional fine-tuning.

    Parameters
    ----------
    grounded_prompt:
        The RAG-grounded prompt produced by ``RetrievalContext.to_llm_prompt()``.
    persona_name:
        One of the keys in ``PERSONAS``. Falls back to ``"baseline"`` if unknown.
    """
    persona = PERSONAS.get(persona_name, PERSONAS["baseline"])
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": persona["system"]},
    ]
    for user_shot, assistant_shot in persona["shots"]:
        messages.append({"role": "user", "content": user_shot})
        messages.append({"role": "assistant", "content": assistant_shot})
    messages.append({"role": "user", "content": grounded_prompt})
    return messages


# ---------------------------------------------------------------------------
# Persona / few-shot specialization
# ---------------------------------------------------------------------------

#: Registry of named personas.  Each entry defines:
#:   system  — the system-prompt that sets the model's voice and constraints
#:   shots   — list of (user_snippet, assistant_snippet) few-shot examples
#:             injected before the real user message so the model imitates
#:             the target style from the very first turn.
PERSONAS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "system": (
            "You are a precise music recommendation assistant. "
            "Ground every recommendation in the provided retrieval context."
        ),
        "shots": [],
    },
    "critic": {
        "system": (
            "You are a seasoned music critic writing for a prestige arts magazine. "
            "Use formal, analytical prose. Reference the song's audio features (tempo, energy, "
            "valence) and place each track in its cultural or historical context. "
            "Avoid casual language, slang, or exclamation marks. "
            "Ground every claim in the retrieval data provided."
        ),
        "shots": [
            (
                "Recommend songs for a late-night jazz session.",
                (
                    'One must begin with "So What" by Miles Davis — a modal landmark '
                    "whose restrained tempo (circa 136 bpm) and low valence create an "
                    "atmosphere of thoughtful introspection rather than mere entertainment. "
                    "The sparse piano voicings that open the track situate it squarely "
                    "in the post-bop tradition, making it an essential touchstone for any "
                    "serious late-night listening programme."
                ),
            ),
            (
                "Suggest something energetic for a morning run.",
                (
                    'The most analytically compelling choice is "Stronger" by Kanye West, '
                    "a track whose high-energy production (energy ≈ 0.87) and brisk tempo "
                    "satisfy the physiological demands of sustained aerobic effort. "
                    "Its sample of Daft Punk's \"Harder, Better, Faster, Stronger\" situates "
                    "it within a lineage of electronic minimalism repurposed for hip-hop "
                    "maximalism — a cultural fusion worth examining on its own terms."
                ),
            ),
        ],
    },
    "dj": {
        "system": (
            "You are an enthusiastic club DJ hyping up the crowd. "
            "Write in short, punchy sentences with energy and slang. "
            "Use ALL CAPS for emphasis occasionally. "
            "Focus on how each track will make people feel on the dance floor. "
            "Ground every recommendation in the retrieval data provided."
        ),
        "shots": [
            (
                "Recommend songs for a workout.",
                (
                    "YO — drop \"Bangarang\" by Skrillex and watch the whole gym GO OFF. "
                    "That bass drop hits at just the right moment to push through your last rep. "
                    "HIGH energy, zero chill, exactly what you need. TRUST. 🔥"
                ),
            ),
            (
                "Give me something chill for studying.",
                (
                    "Okay okay, switching the vibe — queue up \"Coffee\" by beabadoobee. "
                    "Low BPM, real smooth, keeps you locked in without putting you to sleep. "
                    "Perfect background sauce for the late-night grind sesh. 🎧"
                ),
            ),
        ],
    },
    "friend": {
        "system": (
            "You are a knowledgeable and warm friend recommending music over text. "
            "Write casually, like a real conversation. Use contractions, short paragraphs, "
            "and personal enthusiasm ('I love this one because...'). "
            "No bullet points or numbered lists — just talk it through naturally. "
            "Ground every recommendation in the retrieval data provided."
        ),
        "shots": [
            (
                "I want something chill for studying.",
                (
                    "Oh you'd love Bon Iver for this. \"Skinny Love\" has this really "
                    "quiet, folky thing going on that's perfect when you need to concentrate "
                    "but still want something playing. It's not too distracting, you know? "
                    "Give it a try and let me know what you think!"
                ),
            ),
            (
                "Recommend something upbeat.",
                (
                    "Okay so I'm obsessed with \"Uptown Funk\" right now — it's impossible "
                    "not to feel good listening to it. The horns, the groove, Bruno Mars "
                    "just going for it. It's such a good pick-me-up if you're feeling low "
                    "energy. You'll be dancing before the first chorus, I promise."
                ),
            ),
        ],
    },
}

VALID_PERSONAS = tuple(PERSONAS.keys())


def build_persona_messages(
    grounded_prompt: str,
    persona_name: str = "baseline",
) -> List[Dict[str, str]]:
    """Build the full messages list for a given persona.

    Injects few-shot (user, assistant) pairs before the real user message so
    the model imitates the target style without any additional fine-tuning.

    Parameters
    ----------
    grounded_prompt:
        The RAG-grounded prompt produced by ``RetrievalContext.to_llm_prompt()``.
    persona_name:
        One of the keys in ``PERSONAS``. Falls back to ``"baseline"`` if unknown.
    """
    persona = PERSONAS.get(persona_name, PERSONAS["baseline"])
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": persona["system"]},
    ]
    for user_shot, assistant_shot in persona["shots"]:
        messages.append({"role": "user", "content": user_shot})
        messages.append({"role": "assistant", "content": assistant_shot})
    messages.append({"role": "user", "content": grounded_prompt})
    return messages


def send_chat_completion(
    prompt: str,
    config: Optional[LLMConfig] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Send a chat completion request.

    If ``messages`` is provided it is used as-is (for persona / few-shot flows).
    Otherwise a simple system + user message pair is built from ``system_prompt``
    and ``prompt``.
    """
    llm_config = config or LLMConfig.from_env()
    if messages is None:
        messages = [
            {
                "role": "system",
                "content": system_prompt or "You are a precise music recommendation assistant. Ground every recommendation in the provided retrieval context.",
            },
            {"role": "user", "content": prompt},
        ]
    request_body = {
        "model": llm_config.model,
        "messages": messages,
    }

    payload = json.dumps(request_body).encode("utf-8")
    http_request = request.Request(
        f"{llm_config.base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=llm_config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        logger.error("LLM API HTTP error %d: %s", exc.code, response_body[:200])
        raise RuntimeError(
            f"LLM API request failed with HTTP {exc.code}: {response_body}"
        ) from exc
    except error.URLError as exc:
        logger.error("LLM API connection error: %s", exc.reason)
        raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

    return _extract_message_text(response_payload)


def generate_grounded_recommendation_text(
    retrieval_context: RetrievalContext,
    recommendation_count: int = 5,
    config: Optional[LLMConfig] = None,
    system_prompt: Optional[str] = None,
    catalog_songs: Optional[List[Dict[str, Any]]] = None,
    persona: Optional[str] = None,
) -> str:
    """Return LLM recommendation text grounded in the retrieval context.

    Pass ``persona`` (one of the keys in ``PERSONAS``) to activate few-shot
    specialization.  The persona injects a curated system prompt and example
    (user, assistant) turns before the real grounded prompt, steering the
    model's voice without any weight updates.

    If ``catalog_songs`` is provided, the response is passed through
    ``verify_llm_recommendations`` and any songs the LLM mentions that cannot
    be found in the catalog are flagged in the returned text.
    """
    prompt = retrieval_context.to_llm_prompt(recommendation_count=recommendation_count)
    resolved_config = config or LLMConfig.from_env()
    active_persona = persona if persona in PERSONAS else "baseline"
    logger.info(
        "Sending grounded recommendation request to LLM (model=%s, persona=%s).",
        resolved_config.model,
        active_persona,
    )
    persona_messages = build_persona_messages(prompt, active_persona)
    try:
        result = send_chat_completion(
            prompt,
            config=resolved_config,
            system_prompt=system_prompt,
            messages=persona_messages,
        )
        logger.info("LLM returned %d character(s) of recommendation text.", len(result))
    except RuntimeError:
        logger.error("LLM recommendation call failed.", exc_info=True)
        raise

    if catalog_songs:
        verification = verify_llm_recommendations(result, catalog_songs)
        if verification.unverified:
            logger.warning(
                "%d of %d LLM-mentioned song(s) could not be verified in the local catalog.",
                len(verification.unverified),
                len(verification.verified) + len(verification.unverified),
            )
        return verification.annotated_text

    return result