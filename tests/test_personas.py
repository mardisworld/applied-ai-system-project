"""Tests for the few-shot persona specialization layer in llm_client."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import (
    PERSONAS,
    VALID_PERSONAS,
    build_persona_messages,
)


# ---------------------------------------------------------------------------
# PERSONAS registry
# ---------------------------------------------------------------------------

def test_valid_personas_contains_expected_keys():
    assert set(VALID_PERSONAS) == {"baseline", "critic", "dj", "friend"}


def test_all_personas_have_required_fields():
    for name, persona in PERSONAS.items():
        assert "system" in persona, f"Persona '{name}' missing 'system' key"
        assert "shots" in persona, f"Persona '{name}' missing 'shots' key"
        assert isinstance(persona["system"], str), f"Persona '{name}' system must be str"
        assert len(persona["system"]) > 10, f"Persona '{name}' system prompt too short"
        assert isinstance(persona["shots"], list), f"Persona '{name}' shots must be list"


def test_baseline_has_no_few_shot_examples():
    assert PERSONAS["baseline"]["shots"] == []


def test_specialist_personas_have_few_shot_examples():
    for name in ("critic", "dj", "friend"):
        shots = PERSONAS[name]["shots"]
        assert len(shots) >= 1, f"Persona '{name}' should have at least one few-shot example"
        for user_shot, assistant_shot in shots:
            assert isinstance(user_shot, str) and len(user_shot) > 0
            assert isinstance(assistant_shot, str) and len(assistant_shot) > 0


def test_personas_have_distinct_system_prompts():
    system_prompts = [p["system"] for p in PERSONAS.values()]
    assert len(set(system_prompts)) == len(system_prompts), \
        "All personas must have unique system prompts"


# ---------------------------------------------------------------------------
# build_persona_messages
# ---------------------------------------------------------------------------

def test_baseline_messages_structure():
    messages = build_persona_messages("my prompt", "baseline")
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "my prompt"
    # No few-shot examples → only system + user
    assert len(messages) == 2


def test_specialist_messages_include_few_shot_turns():
    for name in ("critic", "dj", "friend"):
        messages = build_persona_messages("my prompt", name)
        roles = [m["role"] for m in messages]
        assert roles[0] == "system", f"First message of '{name}' should be system"
        assert roles[-1] == "user", f"Last message of '{name}' should be user"
        # Must have at least one user+assistant example pair
        assert len(messages) >= 4, (
            f"Persona '{name}' should have system + at least 1 few-shot pair + user, "
            f"got {len(messages)} messages"
        )
        # The few-shot pairs must alternate user / assistant
        for i in range(1, len(messages) - 1):
            expected = "user" if (i % 2 == 1) else "assistant"
            assert messages[i]["role"] == expected, (
                f"Persona '{name}' message[{i}] role expected '{expected}', "
                f"got '{messages[i]['role']}'"
            )


def test_unknown_persona_falls_back_to_baseline():
    messages = build_persona_messages("test", "nonexistent_persona")
    # Should fall back gracefully — matches baseline structure
    assert messages[0]["role"] == "system"
    baseline_system = PERSONAS["baseline"]["system"]
    assert messages[0]["content"] == baseline_system
    assert len(messages) == 2


def test_grounded_prompt_is_always_last_message():
    sentinel = "UNIQUE_SENTINEL_PROMPT_12345"
    for name in VALID_PERSONAS:
        messages = build_persona_messages(sentinel, name)
        assert messages[-1]["content"] == sentinel, (
            f"Persona '{name}' did not place grounded prompt as last message"
        )


def test_system_prompts_encode_distinct_style_cues():
    """Check that each persona's system prompt contains the key stylistic signal."""
    assert "analytical" in PERSONAS["critic"]["system"].lower() or \
           "formal" in PERSONAS["critic"]["system"].lower() or \
           "critic" in PERSONAS["critic"]["system"].lower()
    assert "dj" in PERSONAS["dj"]["system"].lower() or \
           "crowd" in PERSONAS["dj"]["system"].lower() or \
           "energy" in PERSONAS["dj"]["system"].lower()
    assert "friend" in PERSONAS["friend"]["system"].lower() or \
           "casual" in PERSONAS["friend"]["system"].lower() or \
           "warm" in PERSONAS["friend"]["system"].lower()


def test_critic_persona_shots_use_formal_vocabulary():
    """Critic examples should contain analytic / formal vocabulary."""
    all_assistant_text = " ".join(
        shot[1].lower() for shot in PERSONAS["critic"]["shots"]
    )
    formal_words = {"analytical", "analytical", "situated", "lineage", "prestige",
                    "tempo", "harmonic", "modal", "bpm", "dynamic"}
    assert any(w in all_assistant_text for w in formal_words), \
        "Critic few-shot examples should use formal/technical music vocabulary"


def test_dj_persona_shots_use_casual_energetic_vocabulary():
    """DJ examples should contain casual / high-energy vocabulary."""
    all_assistant_text = " ".join(
        shot[1].lower() for shot in PERSONAS["dj"]["shots"]
    )
    casual_words = {"yo", "vibe", "energy", "drop", "crowd", "sesh", "banger", "trust"}
    assert any(w in all_assistant_text for w in casual_words), \
        "DJ few-shot examples should use casual/energetic vocabulary"
