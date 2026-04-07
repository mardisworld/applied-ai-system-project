"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import available_strategy_names, load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs_dataset_full.csv") 
    strategy_options = available_strategy_names()

    print("Choose a ranking strategy:")
    for option in strategy_options:
        print(f"- {option}")

    selected_strategy = input("Enter strategy name: ").strip().lower()
    strategy_mode = selected_strategy if selected_strategy in strategy_options else "balanced"
    if selected_strategy and selected_strategy not in strategy_options:
        print(f"Unknown strategy '{selected_strategy}'. Falling back to '{strategy_mode}'.\n")

    # Demo profile chosen to make strategy differences more obvious.
    # The preferences intentionally pull in different directions so each
    # ranking mode emphasizes a different part of the user's taste.
    user_prefs = {
        # Core preferences
        "favorite_genre": "study",
        "favorite_mood": "intense",
        "target_energy": 0.95,
        "likes_acoustic": False,
        
        # New category-aware preferences
        "favorite_detailed_mood": "aggressive",
        "preferred_energy_level": "High",
        "preferred_danceability_tier": "Danceable",
        "preferred_tempo_category": "Fast",
        
        # Additional numerical targets
        "target_tempo": 150,
        "target_valence": 0.45,
        "target_danceability": 0.75,
        "target_popularity": 60,
        
        # Weights for scoring
        "weight_genre": 7.0,
        "weight_mood": 6.0,
        "weight_detailed_mood": 4.0,
        "weight_energy": 7.0,
        "weight_tempo": 8.0,
        "weight_valence": 2.0,
        "weight_danceability": 4.0,
        "weight_acousticness": 5.0,
        "weight_energy_level": 3.0,
        "weight_danceability_tier": 2.0,
        "weight_tempo_category": 3.0,
        "weight_popularity": 1.0,

        # Diversity controls for the final top-k selection step
        "diversity_artist_penalty": 6.0,
        "diversity_genre_penalty": 1.5,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5, strategy_name=strategy_mode)

    print("\nTop recommendations:\n")
    print(f"Ranking mode: {strategy_mode}")
    print(f"Available modes: {', '.join(available_strategy_names())}\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
