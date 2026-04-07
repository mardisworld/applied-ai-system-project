from src.recommender import Song, UserProfile, Recommender, recommend_songs

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
