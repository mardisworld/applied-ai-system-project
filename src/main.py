"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs_dataset_full.csv") 

    # Enhanced user profile with category-aware preferences
    user_prefs = {
        # Core preferences (original)
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
        
        # New category-aware preferences
        "favorite_detailed_mood": "upbeat",
        "preferred_energy_level": "High",
        "preferred_danceability_tier": "Danceable",
        "preferred_tempo_category": "Upbeat",
        
        # Additional numerical targets
        "target_tempo": 120,
        "target_valence": 0.75,
        "target_danceability": 0.7,
        "target_popularity": 70,
        
        # Weights for scoring (customize to prioritize different attributes)
        "weight_genre": 10.0,
        "weight_mood": 5.0,
        "weight_detailed_mood": 3.0,
        "weight_energy": 5.0,
        "weight_tempo": 2.0,
        "weight_valence": 2.0,
        "weight_danceability": 3.0,
        "weight_acousticness": 5.0,
        "weight_energy_level": 2.0,
        "weight_danceability_tier": 2.0,
        "weight_tempo_category": 2.0,
        "weight_popularity": 1.0,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
