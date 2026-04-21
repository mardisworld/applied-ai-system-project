from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import (
    RetrievalLayer,
    build_user_preferences_from_retrieval_context,
    candidate_tracks_to_song_dicts,
    parse_query_signals,
)


def make_retrieval_layer() -> RetrievalLayer:
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
            "title": "Skinny Love",
            "artist": "Bon Iver",
            "genre": "indie-folk",
            "mood": "melancholic",
            "energy": 0.31,
            "tempo_bpm": 76.0,
            "valence": 0.29,
            "danceability": 0.3,
            "acousticness": 0.91,
            "popularity": 85,
        },
        {
            "id": 3,
            "title": "Mystery of Love",
            "artist": "Sufjan Stevens",
            "genre": "indie-folk",
            "mood": "chill",
            "energy": 0.26,
            "tempo_bpm": 81.0,
            "valence": 0.38,
            "danceability": 0.35,
            "acousticness": 0.9,
            "popularity": 79,
        },
        {
            "id": 4,
            "title": "Naked As We Came",
            "artist": "Iron & Wine",
            "genre": "indie-folk",
            "mood": "chill",
            "energy": 0.24,
            "tempo_bpm": 83.0,
            "valence": 0.37,
            "danceability": 0.36,
            "acousticness": 0.93,
            "popularity": 68,
        },
        {
            "id": 5,
            "title": "Garden Song",
            "artist": "Phoebe Bridgers",
            "genre": "indie",
            "mood": "moody",
            "energy": 0.35,
            "tempo_bpm": 90.0,
            "valence": 0.41,
            "danceability": 0.4,
            "acousticness": 0.78,
            "popularity": 72,
        },
        {
            "id": 6,
            "title": "Coffee Table Theme",
            "artist": "Study Halo",
            "genre": "study",
            "mood": "chill",
            "energy": 0.19,
            "tempo_bpm": 72.0,
            "valence": 0.33,
            "danceability": 0.22,
            "acousticness": 0.84,
            "popularity": 50,
        },
    ]
    return RetrievalLayer(songs)


def test_parse_query_extracts_mood_activity_and_seed_artist():
    signals = parse_query_signals(
        "I want something chill for studying, similiar to Bon Iver"
    )

    assert signals.mood == "chill"
    assert signals.activity == "studying"
    assert signals.seed_artist == "Bon Iver"


def test_retrieve_returns_seed_artist_profile_and_similar_artists():
    context = make_retrieval_layer().retrieve(
        "I want something chill for studying, similiar to Bon Iver",
        candidate_limit=4,
        similar_artist_limit=3,
    )

    assert context.seed_artist_profile is not None
    assert context.seed_artist_profile.artist_name == "Bon Iver"
    assert any(match.artist_name == "Sufjan Stevens" for match in context.similar_artists)
    assert any(candidate.track["artist"] == "Bon Iver" for candidate in context.candidate_tracks)
    similar_artist_names = [match.artist_name for match in context.similar_artists]
    assert similar_artist_names[:2] == ["Sufjan Stevens", "Iron & Wine"]


def test_retrieval_context_string_includes_context_sections():
    context = make_retrieval_layer().retrieve(
        "I want something chill for studying, similiar to Bon Iver",
        candidate_limit=3,
        similar_artist_limit=2,
    )

    formatted = context.to_context_string()

    assert "Parsed signals: mood=chill, activity=studying, seed_artist=Bon Iver" in formatted
    assert "Seed artist profile: Bon Iver" in formatted
    assert "Similar artists:" in formatted
    assert "Candidate tracks:" in formatted


def test_retrieval_context_can_be_rendered_as_llm_grounding_prompt():
    context = make_retrieval_layer().retrieve(
        "I want something chill for studying, similiar to Bon Iver",
        candidate_limit=3,
        similar_artist_limit=2,
    )

    prompt = context.to_llm_prompt(recommendation_count=5)

    assert "The user asked: I want something chill for studying, similiar to Bon Iver" in prompt
    assert "Retrieved seed artist profile: Bon Iver;" in prompt
    assert "Retrieved similar artists:" in prompt
    assert "Retrieved candidate tracks:" in prompt
    assert "recommend 5 songs" in prompt


def test_retrieval_context_can_be_converted_into_user_preferences_and_candidates():
    context = make_retrieval_layer().retrieve(
        "I want something chill for studying, similiar to Bon Iver",
        candidate_limit=3,
        similar_artist_limit=2,
    )

    user_prefs = build_user_preferences_from_retrieval_context(context)
    candidate_songs = candidate_tracks_to_song_dicts(context)

    assert user_prefs["favorite_mood"] == "chill"
    assert user_prefs["favorite_genre"] == "indie-folk"
    assert user_prefs["likes_acoustic"] is True
    assert user_prefs["target_energy"] > 0
    assert candidate_songs[0]["title"] == context.candidate_tracks[0].track["title"]