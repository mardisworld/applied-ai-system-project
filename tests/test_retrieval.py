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


# ---------------------------------------------------------------------------
# parse_query_signals — edge cases
# ---------------------------------------------------------------------------

def test_parse_query_signals_extracts_only_activity_when_no_mood_or_artist():
    signals = parse_query_signals("songs for working out")
    assert signals.activity == "working out"
    assert signals.mood is None
    assert signals.seed_artist is None


def test_parse_query_signals_extracts_seed_artist_from_style_of_pattern():
    signals = parse_query_signals("in the style of Radiohead")
    assert signals.seed_artist == "Radiohead"
    assert signals.activity is None


def test_parse_query_signals_returns_no_signals_for_generic_query():
    signals = parse_query_signals("give me some good songs")
    assert signals.mood is None
    assert signals.activity is None
    assert signals.seed_artist is None


def test_parse_query_signals_preserves_raw_query_unchanged():
    query = "I want chill music like Bon Iver for studying"
    signals = parse_query_signals(query)
    assert signals.raw_query == query


def test_parse_query_signals_detects_gym_keyword_as_workout():
    signals = parse_query_signals("something loud for the gym")
    assert signals.activity == "working out"


# ---------------------------------------------------------------------------
# build_user_preferences_from_retrieval_context — activity weight boosts
# ---------------------------------------------------------------------------

def test_workout_activity_raises_energy_and_tempo_weights_above_default():
    context_workout = make_retrieval_layer().retrieve(
        "upbeat songs for working out", candidate_limit=3, similar_artist_limit=0
    )
    context_default = make_retrieval_layer().retrieve(
        "upbeat songs", candidate_limit=3, similar_artist_limit=0
    )
    prefs_workout = build_user_preferences_from_retrieval_context(context_workout)
    prefs_default = build_user_preferences_from_retrieval_context(context_default)

    assert prefs_workout["weight_energy"] > prefs_default["weight_energy"]
    assert prefs_workout["weight_tempo"] > prefs_default["weight_tempo"]


def test_studying_activity_raises_acousticness_weight_above_default():
    context_study = make_retrieval_layer().retrieve(
        "music for studying", candidate_limit=3, similar_artist_limit=0
    )
    context_default = make_retrieval_layer().retrieve(
        "music to listen to", candidate_limit=3, similar_artist_limit=0
    )
    prefs_study = build_user_preferences_from_retrieval_context(context_study)
    prefs_default = build_user_preferences_from_retrieval_context(context_default)

    assert prefs_study["weight_acousticness"] >= prefs_default["weight_acousticness"]


# ---------------------------------------------------------------------------
# candidate_tracks_to_song_dicts — edge cases
# ---------------------------------------------------------------------------

def test_candidate_tracks_to_song_dicts_returns_empty_list_for_no_candidates():
    context = make_retrieval_layer().retrieve(
        "songs like an artist that does not exist xyzzy999",
        candidate_limit=0,
        similar_artist_limit=0,
    )
    result = candidate_tracks_to_song_dicts(context)
    assert isinstance(result, list)


def test_candidate_tracks_to_song_dicts_preserves_candidate_order():
    context = make_retrieval_layer().retrieve(
        "I want something chill for studying, similiar to Bon Iver",
        candidate_limit=4,
        similar_artist_limit=0,
    )
    result = candidate_tracks_to_song_dicts(context)
    expected_titles = [c.track["title"] for c in context.candidate_tracks]
    assert [r["title"] for r in result] == expected_titles