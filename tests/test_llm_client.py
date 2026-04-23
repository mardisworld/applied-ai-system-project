from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import (
    LLMConfig,
    LLMVerificationResult,
    _extract_message_text,
    generate_grounded_recommendation_text,
    verify_llm_recommendations,
)
from src.retrieval import RetrievalLayer


def make_retrieval_context():
    songs = [
        {
            "id": 1,
            "title": "Holocene",
            "artist": "Bon Iver",
            "genre": "indie-folk",
            "mood": "chill",
            "energy": 0.28,
            "tempo_bpm": 78.0,
            "valence": 0.34,
            "danceability": 0.32,
            "acousticness": 0.88,
            "popularity": 82,
        },
        {
            "id": 2,
            "title": "Mystery of Love",
            "artist": "Sufjan Stevens",
            "genre": "indie-folk",
            "mood": "chill",
            "energy": 0.26,
            "tempo_bpm": 81.0,
            "valence": 0.38,
            "danceability": 0.35,
            "acousticness": 0.90,
            "popularity": 79,
        },
    ]
    return RetrievalLayer(songs).retrieve(
        "I want something chill for studying, similiar to Bon Iver",
        candidate_limit=2,
        similar_artist_limit=1,
    )


def test_extract_message_text_handles_string_content():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "Recommend Holocene by Bon Iver.",
                }
            }
        ]
    }

    assert _extract_message_text(payload) == "Recommend Holocene by Bon Iver."


def test_generate_grounded_recommendation_text_sends_llm_prompt(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "1. Holocene - grounded recommendation output",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(http_request, timeout):
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("src.llm_client.request.urlopen", fake_urlopen)

    result = generate_grounded_recommendation_text(
        make_retrieval_context(),
        recommendation_count=5,
        config=LLMConfig(api_key="test-key", base_url="https://example.com/v1", model="gpt-5.4", timeout_seconds=12),
    )

    assert result == "1. Holocene - grounded recommendation output"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["timeout"] == 12


# ---------------------------------------------------------------------------
# verify_llm_recommendations
# ---------------------------------------------------------------------------

CATALOG = [
    {"title": "Holocene", "artist": "Bon Iver"},
    {"title": "Mystery of Love", "artist": "Sufjan Stevens"},
    {"title": "Re: Stacks", "artist": "Bon Iver"},
    {"title": "Skinny Love", "artist": "Bon Iver"},
]


def test_verify_all_found_in_catalog():
    text = 'Try "Holocene" by Bon Iver and "Mystery of Love" by Sufjan Stevens.'
    result = verify_llm_recommendations(text, CATALOG)
    assert len(result.verified) == 2
    assert result.unverified == []
    assert "Note:" not in result.annotated_text


def test_verify_hallucinated_title_flagged():
    text = 'Consider "Nonexistent Song" by Made Up Artist for your playlist.'
    result = verify_llm_recommendations(text, CATALOG)
    assert result.verified == []
    assert len(result.unverified) == 1
    assert result.unverified[0][0] == "Nonexistent Song"
    assert "Note:" in result.annotated_text
    assert "Nonexistent Song" in result.annotated_text


def test_verify_fuzzy_match_tolerates_minor_typo():
    # "Holcene" is close enough to "Holocene" at cutoff 0.72
    text = 'I recommend "Holcene" by Bon Iver.'
    result = verify_llm_recommendations(text, CATALOG)
    assert len(result.verified) == 1
    assert result.unverified == []


def test_verify_mixed_verified_and_unverified():
    text = '"Skinny Love" by Bon Iver is great. Also try "Ghost Planet" by Fake Band.'
    result = verify_llm_recommendations(text, CATALOG)
    assert len(result.verified) == 1
    assert len(result.unverified) == 1
    assert "Ghost Planet" in result.annotated_text


def test_verify_empty_catalog_flags_everything():
    text = '"Holocene" by Bon Iver'
    result = verify_llm_recommendations(text, [])
    assert result.verified == []
    assert len(result.unverified) == 1


def test_verify_no_mentions_returns_empty_lists():
    text = "Here are some general tips for finding good music."
    result = verify_llm_recommendations(text, CATALOG)
    assert result.verified == []
    assert result.unverified == []
    assert result.annotated_text == text


def test_verify_deduplicates_repeated_mentions():
    text = '"Re: Stacks" by Bon Iver is wonderful. "Re: Stacks" by Bon Iver again.'
    result = verify_llm_recommendations(text, CATALOG)
    assert len(result.verified) == 1
    assert result.unverified == []


def test_verify_curly_quotes_are_recognized():
    text = "\u201cHolocene\u201d by Bon Iver"
    result = verify_llm_recommendations(text, CATALOG)
    assert len(result.verified) == 1


def test_verify_annotated_text_unchanged_when_all_verified():
    text = '"Holocene" by Bon Iver is the best.'
    result = verify_llm_recommendations(text, CATALOG)
    assert result.annotated_text == text


def test_generate_passes_catalog_songs_to_verify(monkeypatch):
    """When catalog_songs is provided, the guard is run and annotates hallucinations."""
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": '"Ghost Song" by Fake Artist is great.'}}]}).encode()

    monkeypatch.setattr("src.llm_client.request.urlopen", lambda *a, **kw: FakeResponse())

    result = generate_grounded_recommendation_text(
        make_retrieval_context(),
        config=LLMConfig(api_key="x", base_url="https://x.com/v1", model="m", timeout_seconds=5),
        catalog_songs=CATALOG,
    )
    assert "Note:" in result
    assert "Ghost Song" in result