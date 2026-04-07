from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommender import Song, UserProfile, Recommender, available_strategy_names, get_ranking_strategy, recommend_songs

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_functional_recommend_songs_respects_detailed_mood_weight():
    songs = [
        {
            "id": 1,
            "title": "Upbeat Pop",
            "artist": "Artist 1",
            "genre": "pop",
            "mood": "happy",
            "detailed_mood": "upbeat",
            "energy": 0.8,
            "tempo_bpm": 120,
            "valence": 0.8,
            "danceability": 0.7,
            "acousticness": 0.1,
        },
        {
            "id": 2,
            "title": "Relaxed Pop",
            "artist": "Artist 2",
            "genre": "pop",
            "mood": "happy",
            "detailed_mood": "relaxed",
            "energy": 0.8,
            "tempo_bpm": 120,
            "valence": 0.8,
            "danceability": 0.7,
            "acousticness": 0.1,
        },
    ]

    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "favorite_detailed_mood": "upbeat",
        "target_energy": 0.8,
        "likes_acoustic": False,
        "weight_genre": 10.0,
        "weight_mood": 5.0,
        "weight_detailed_mood": 10.0,
    }

    recommendations = recommend_songs(user_prefs, songs, k=2)
    assert recommendations[0][0]["id"] == 1
    assert recommendations[1][0]["id"] == 2


def test_strategy_registry_exposes_switchable_modes():
    strategy_names = available_strategy_names()

    assert "balanced" in strategy_names
    assert "genre-first" in strategy_names
    assert "mood-first" in strategy_names
    assert "energy-focused" in strategy_names
    assert get_ranking_strategy("genre-first").name == "genre-first"


def test_genre_first_and_mood_first_can_rank_same_songs_differently():
    songs = [
        {
            "id": 1,
            "title": "Genre Match",
            "artist": "Artist 1",
            "genre": "emo",
            "mood": "intense",
            "energy": 0.7,
            "tempo_bpm": 110,
            "valence": 0.4,
            "danceability": 0.5,
            "acousticness": 0.2,
        },
        {
            "id": 2,
            "title": "Mood Match",
            "artist": "Artist 2",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.7,
            "tempo_bpm": 110,
            "valence": 0.75,
            "danceability": 0.5,
            "acousticness": 0.2,
        },
    ]
    user_prefs = {
        "favorite_genre": "emo",
        "favorite_mood": "happy",
        "target_energy": 0.7,
        "target_valence": 0.75,
        "likes_acoustic": False,
        "weight_genre": 4.0,
        "weight_mood": 4.0,
        "weight_energy": 1.0,
        "weight_valence": 1.0,
    }

    genre_first_results = recommend_songs(user_prefs, songs, k=2, strategy_name="genre-first")
    mood_first_results = recommend_songs(user_prefs, songs, k=2, strategy_name="mood-first")

    assert genre_first_results[0][0]["title"] == "Genre Match"
    assert mood_first_results[0][0]["title"] == "Mood Match"


def test_energy_focused_strategy_prioritizes_energy_alignment():
    songs = [
        {
            "id": 1,
            "title": "Exact Energy Match",
            "artist": "Artist 1",
            "genre": "indie",
            "mood": "moody",
            "energy": 0.9,
            "tempo_bpm": 128,
            "valence": 0.6,
            "danceability": 0.75,
            "acousticness": 0.2,
        },
        {
            "id": 2,
            "title": "Genre Match Only",
            "artist": "Artist 2",
            "genre": "emo",
            "mood": "happy",
            "energy": 0.4,
            "tempo_bpm": 90,
            "valence": 0.3,
            "danceability": 0.4,
            "acousticness": 0.3,
        },
    ]
    user_prefs = {
        "favorite_genre": "emo",
        "favorite_mood": "happy",
        "target_energy": 0.9,
        "target_tempo": 128,
        "likes_acoustic": False,
        "weight_genre": 3.0,
        "weight_mood": 3.0,
        "weight_energy": 4.0,
        "weight_tempo": 3.0,
    }

    results = recommend_songs(user_prefs, songs, k=2, strategy_name="energy-focused")

    assert results[0][0]["title"] == "Exact Energy Match"


def test_recommend_applies_artist_diversity_penalty_to_top_results():
    songs = [
        Song(
            id=1,
            title="Artist A Song 1",
            artist="Artist A",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.8,
            danceability=0.8,
            acousticness=0.1,
        ),
        Song(
            id=2,
            title="Artist A Song 2",
            artist="Artist A",
            genre="pop",
            mood="happy",
            energy=0.79,
            tempo_bpm=121,
            valence=0.79,
            danceability=0.79,
            acousticness=0.1,
        ),
        Song(
            id=3,
            title="Artist B Song 1",
            artist="Artist B",
            genre="pop",
            mood="happy",
            energy=0.76,
            tempo_bpm=118,
            valence=0.76,
            danceability=0.76,
            acousticness=0.1,
        ),
    ]
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        target_tempo=120,
        likes_acoustic=False,
        diversity_artist_penalty=6.0,
        diversity_genre_penalty=0.0,
    )
    rec = Recommender(songs)

    results = rec.recommend(user, k=2)

    assert results[0].artist == "Artist A"
    assert results[1].artist == "Artist B"


def test_functional_recommend_songs_applies_artist_diversity_penalty():
    songs = [
        {
            "id": 1,
            "title": "Artist A Song 1",
            "artist": "Artist A",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 120,
            "valence": 0.8,
            "danceability": 0.8,
            "acousticness": 0.1,
        },
        {
            "id": 2,
            "title": "Artist A Song 2",
            "artist": "Artist A",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.79,
            "tempo_bpm": 121,
            "valence": 0.79,
            "danceability": 0.79,
            "acousticness": 0.1,
        },
        {
            "id": 3,
            "title": "Artist B Song 1",
            "artist": "Artist B",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.76,
            "tempo_bpm": 118,
            "valence": 0.76,
            "danceability": 0.76,
            "acousticness": 0.1,
        },
    ]
    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "target_tempo": 120,
        "likes_acoustic": False,
        "diversity_artist_penalty": 6.0,
        "diversity_genre_penalty": 0.0,
    }

    recommendations = recommend_songs(user_prefs, songs, k=2)

    assert recommendations[0][0]["artist"] == "Artist A"
    assert recommendations[1][0]["artist"] == "Artist B"


# Project profile regression tests
def test_profile_happy_energetic_emo_ranks_upbeat_emo_tracks_first():
    songs = [
        Song(id=1, title="This Is Why", artist="Paramore", genre="emo", mood="happy", energy=0.8, tempo_bpm=120, valence=0.75, danceability=0.7, acousticness=0.1),
        Song(id=2, title="Carry Me Away", artist="John Mayer", genre="emo", mood="happy", energy=0.78, tempo_bpm=118, valence=0.7, danceability=0.68, acousticness=0.12),
        Song(id=3, title="Somebody - Edit", artist="The 1975", genre="emo", mood="happy", energy=0.77, tempo_bpm=122, valence=0.74, danceability=0.69, acousticness=0.09),
        Song(id=4, title="Du riechst so gut", artist="Rammstein", genre="industrial", mood="intense", energy=0.95, tempo_bpm=130, valence=0.4, danceability=0.5, acousticness=0.05),
    ]
    user = UserProfile(
        favorite_genre="emo",
        favorite_mood="happy",
        target_energy=0.8,
        target_tempo=120,
        likes_acoustic=False,
        weight_genre=0.5,
        weight_mood=5.0,
        weight_tempo=10.0
    )
    rec = Recommender(songs)
    results = rec.recommend(user, k=3)
    explanation = rec.explain_recommendation(user, results[0])

    assert results[0].title == "This Is Why"
    assert all(song.genre == "emo" for song in results)
    assert "matches your favorite mood" in explanation


def test_profile_chill_acoustic_returns_acoustic_chill_matches():
    songs = [
        Song(id=1, title="ただ声一つ", artist="Japanese Artist", genre="folk", mood="chill", energy=0.5, tempo_bpm=90, valence=0.6, danceability=0.5, acousticness=0.95),
        Song(id=2, title="Exu", artist="Portuguese Artist", genre="world", mood="chill", energy=0.48, tempo_bpm=88, valence=0.62, danceability=0.52, acousticness=0.92),
        Song(id=3, title="Idhayathai Kolluriyeh", artist="Indian Artist", genre="indian", mood="chill", energy=0.52, tempo_bpm=92, valence=0.58, danceability=0.48, acousticness=0.97),
        Song(id=4, title="Mover Awayer", artist="Unknown", genre="emo", mood="chill", energy=0.4, tempo_bpm=85, valence=0.55, danceability=0.45, acousticness=0.9),
    ]
    user = UserProfile(
        favorite_genre="emo",
        favorite_mood="chill",
        target_energy=0.5,
        likes_acoustic=True,
        weight_genre=0.5,
        weight_mood=10.0,
        weight_acousticness=5.0
    )
    rec = Recommender(songs)
    results = rec.recommend(user, k=3)
    explanation = rec.explain_recommendation(user, results[0])

    assert any("Japanese" in song.artist or "Portuguese" in song.artist or "Indian" in song.artist for song in results)
    assert all(song.mood == "chill" for song in results)
    assert "is acoustic" in explanation


def test_profile_pop_happy_still_prefers_better_tempo_match_when_genre_weight_is_low():
    songs = [
        Song(id=1, title="This Is Why", artist="Paramore", genre="emo", mood="happy", energy=0.8, tempo_bpm=120, valence=0.75, danceability=0.7, acousticness=0.1),
        Song(id=2, title="Dance Monkey", artist="Tones and I", genre="pop", mood="happy", energy=0.7, tempo_bpm=90, valence=0.8, danceability=0.8, acousticness=0.35),
    ]
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        target_tempo=120,
        likes_acoustic=False,
        weight_genre=0.5,
        weight_mood=5.0,
        weight_tempo=10.0
    )
    rec = Recommender(songs)
    results = rec.recommend(user, k=2)
    explanation = rec.explain_recommendation(user, results[0])

    # Low genre weight allows mood and tempo similarity to outweigh the genre mismatch.
    assert results[0].title == "This Is Why"
    assert results[1].title == "Dance Monkey"
    assert "has similar tempo" in explanation


def test_profile_weight_ablation_still_returns_ranked_results():
    songs = [
        Song(id=1, title="This Is Why", artist="Paramore", genre="emo", mood="happy", energy=0.8, tempo_bpm=120, valence=0.75, danceability=0.7, acousticness=0.1),
        Song(id=2, title="Carry Me Away", artist="John Mayer", genre="emo", mood="happy", energy=0.78, tempo_bpm=118, valence=0.7, danceability=0.68, acousticness=0.12),
    ]
    user = UserProfile(
        favorite_genre="emo",
        favorite_mood="happy",
        target_energy=0.8,
        target_tempo=120,
        likes_acoustic=False,
        weight_genre=0.5,
        weight_mood=5.0,
        weight_tempo=0.0,
        weight_valence=0.0
    )
    rec = Recommender(songs)
    results = rec.recommend(user, k=2)
    explanation = rec.explain_recommendation(user, results[0])

    # Zeroing tempo and valence weights should still produce a stable, explainable ranking.
    assert len(results) == 2
    assert explanation.strip() != ""
