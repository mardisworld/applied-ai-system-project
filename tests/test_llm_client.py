from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import LLMConfig, _extract_message_text, generate_grounded_recommendation_text
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
    assert captured["body"]["model"] == "gpt-5.4"
    assert "The user asked: I want something chill for studying, similiar to Bon Iver" in captured["body"]["messages"][1]["content"]