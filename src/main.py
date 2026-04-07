"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import textwrap

from src.recommender import available_strategy_names, load_songs, recommend_songs


PROFILE_PRESETS = {
    "subtle-differences": {
        "favorite_genre": "emo",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
        "favorite_detailed_mood": "upbeat",
        "preferred_energy_level": "High",
        "preferred_danceability_tier": "Danceable",
        "preferred_tempo_category": "Upbeat",
        "target_tempo": 120,
        "target_valence": 0.75,
        "target_danceability": 0.7,
        "target_popularity": 70,
        "weight_genre": 0.5,
        "weight_mood": 5.0,
        "weight_detailed_mood": 3.0,
        "weight_energy": 5.0,
        "weight_tempo": 10.0,
        "weight_valence": 2.0,
        "weight_danceability": 3.0,
        "weight_acousticness": 5.0,
        "weight_energy_level": 2.0,
        "weight_danceability_tier": 2.0,
        "weight_tempo_category": 2.0,
        "weight_popularity": 1.0,
        "diversity_artist_penalty": 6.0,
        "diversity_genre_penalty": 1.5,
    },
    "obvious-strategy-demo": {
        "favorite_genre": "study",
        "favorite_mood": "intense",
        "target_energy": 0.95,
        "likes_acoustic": False,
        "favorite_detailed_mood": "aggressive",
        "preferred_energy_level": "High",
        "preferred_danceability_tier": "Danceable",
        "preferred_tempo_category": "Fast",
        "target_tempo": 150,
        "target_valence": 0.45,
        "target_danceability": 0.75,
        "target_popularity": 60,
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
        "diversity_artist_penalty": 6.0,
        "diversity_genre_penalty": 1.5,
    },
    "genre-vs-mood-conflict": {
        "favorite_genre": "black-metal",
        "favorite_mood": "happy",
        "target_energy": 0.9,
        "likes_acoustic": False,
        "favorite_detailed_mood": "upbeat",
        "preferred_energy_level": "High",
        "preferred_danceability_tier": "Danceable",
        "preferred_tempo_category": "Fast",
        "target_tempo": 145,
        "target_valence": 0.8,
        "target_danceability": 0.65,
        "target_popularity": 55,
        "weight_genre": 6.0,
        "weight_mood": 6.0,
        "weight_detailed_mood": 3.0,
        "weight_energy": 6.0,
        "weight_tempo": 7.0,
        "weight_valence": 4.0,
        "weight_danceability": 3.0,
        "weight_acousticness": 4.0,
        "weight_energy_level": 3.0,
        "weight_danceability_tier": 2.0,
        "weight_tempo_category": 3.0,
        "weight_popularity": 1.0,
        "diversity_artist_penalty": 6.0,
        "diversity_genre_penalty": 1.5,
    },
}


def format_recommendation_table(table_rows: list[list[str]]) -> str:
    width = 78

    def wrap_block(prefix: str, value: str, indent: str = "") -> list[str]:
        wrapped = textwrap.wrap(
            value,
            width=width - len(prefix),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        lines = [f"{prefix}{wrapped[0]}"]
        lines.extend(f"{indent}{line}" for line in wrapped[1:])
        return lines

    def parse_reasons(value: str) -> list[str]:
        cleaned = str(value).strip().removeprefix("This song ").removesuffix(".")
        return [reason.strip() for reason in cleaned.split(",") if reason.strip()]

    sections = []
    separator = "=" * width
    divider = "-" * width

    for row in table_rows:
        rank, title, artist, genre, score, reasons_text = [str(value) for value in row]
        section_lines = [separator, f"[{rank}] {title}", divider]
        section_lines.extend(wrap_block("Artist : ", artist, " " * 9))
        section_lines.extend(wrap_block("Genre  : ", genre, " " * 9))
        section_lines.extend(wrap_block("Score  : ", score, " " * 9))
        section_lines.append("Reasons:")

        for reason in parse_reasons(reasons_text):
            section_lines.extend(wrap_block("  - ", reason, " " * 4))

        sections.append("\n".join(section_lines))

    if not sections:
        return separator

    return "\n\n".join(sections + [separator])


def main() -> None:
    songs = load_songs("data/songs_dataset_full.csv") 
    strategy_options = available_strategy_names()
    profile_options = list(PROFILE_PRESETS.keys())

    print("Choose a demo profile:")
    for option in profile_options:
        print(f"- {option}")

    selected_profile = input("Enter profile name: ").strip().lower()
    profile_name = selected_profile if selected_profile in PROFILE_PRESETS else "obvious-strategy-demo"
    if selected_profile and selected_profile not in PROFILE_PRESETS:
        print(f"Unknown profile '{selected_profile}'. Falling back to '{profile_name}'.\n")

    print("Choose a ranking strategy:")
    for option in strategy_options:
        print(f"- {option}")

    selected_strategy = input("Enter strategy name: ").strip().lower()
    strategy_mode = selected_strategy if selected_strategy in strategy_options else "balanced"
    if selected_strategy and selected_strategy not in strategy_options:
        print(f"Unknown strategy '{selected_strategy}'. Falling back to '{strategy_mode}'.\n")

    user_prefs = PROFILE_PRESETS[profile_name].copy()

    recommendations = recommend_songs(user_prefs, songs, k=5, strategy_name=strategy_mode)

    table_rows = []
    for index, rec in enumerate(recommendations, start=1):
        song, score, explanation = rec
        table_rows.append(
            [
                index,
                song["title"],
                song["artist"],
                song["genre"],
                f"{score:.2f}",
                explanation,
            ]
        )

    print("\nTop recommendations:\n")
    print(f"Demo profile: {profile_name}")
    print(f"Ranking mode: {strategy_mode}")
    print(f"Available modes: {', '.join(available_strategy_names())}\n")
    print(format_recommendation_table(table_rows))


if __name__ == "__main__":
    main()
