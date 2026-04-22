"""Command line runner for the Music Recommender Simulation."""

import argparse
import json
import textwrap

from src.llm_client import generate_grounded_recommendation_text, llm_is_configured
from src.recommender import available_strategy_names, load_songs, recommend_songs
from src.retrieval import (
    RetrievalLayer,
    build_user_preferences_from_retrieval_context,
    candidate_tracks_to_song_dicts,
)
from src.spotify_api import (
    build_playlist_from_recommendations,
    exchange_code_for_token,
    get_authorization_url,
    get_current_user,
    run_local_auth_flow,
    spotify_is_configured,
)


VALID_RUN_MODES = ("retrieval-only", "llm-only", "hybrid")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the music recommender from a natural-language query.")
    parser.add_argument(
        "--mode",
        choices=VALID_RUN_MODES,
        default="hybrid",
        help="Choose retrieval-only, llm-only, or hybrid output.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Natural-language music request. If omitted, the CLI will prompt interactively.",
    )
    parser.add_argument(
        "--strategy",
        choices=available_strategy_names(),
        default=None,
        help="Ranking strategy for hybrid mode. If omitted, the CLI will prompt interactively in hybrid mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output instead of formatted console text.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="When used with --json, omit the serialized retrieval_context block from the output.",
    )
    parser.add_argument(
        "--no-llm-prompt",
        action="store_true",
        help="When used with --json, omit the rendered llm_prompt string from the output.",
    )
    return parser.parse_args()


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _serialize_recommendations(recommendations: list[tuple[dict, float, str]]) -> list[dict]:
    serialized = []
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        serialized.append(
            {
                "rank": rank,
                "song": song,
                "score": round(score, 3),
                "explanation": explanation,
            }
        )
    return serialized


def main() -> None:
    args = parse_args()
    songs = load_songs("data/songs_dataset_full.csv")
    retrieval_layer = RetrievalLayer(songs)
    strategy_options = available_strategy_names()

    query = (args.prompt or "").strip()
    if not query:
        print("Describe what you want to hear.")
        print("Example: I want something chill for studying, similar to Bon Iver")
        query = input("Enter prompt: ").strip()

    if not query:
        query = "I want something chill for studying, similar to Bon Iver"
        print(f"Using default prompt: {query}\n")

    strategy_mode = args.strategy
    if args.mode != "retrieval-only" and args.strategy is None:
        print("\nHow would you like songs to be ranked?")
        for i, option in enumerate(strategy_options, start=1):
            print(f"  {i}. {option}")
        selected_strategy = input("Enter strategy name or number [balanced]: ").strip().lower()
        if selected_strategy.isdigit():
            idx = int(selected_strategy) - 1
            strategy_mode = strategy_options[idx] if 0 <= idx < len(strategy_options) else "balanced"
        elif selected_strategy in strategy_options:
            strategy_mode = selected_strategy
        else:
            if selected_strategy:
                print(f"Unknown strategy '{selected_strategy}'. Using 'balanced'.")
            strategy_mode = "balanced"

    if args.mode == "llm-only" and not args.json and not llm_is_configured():
        raise SystemExit("LLM mode requires LLM_API_KEY or OPENAI_API_KEY to be set.")

    retrieval_context = retrieval_layer.retrieve(query, candidate_limit=15, similar_artist_limit=5)
    result_payload = {
        "prompt": query,
        "mode": args.mode,
        "strategy": strategy_mode if args.mode == "hybrid" else None,
        "llm_recommendations": None,
        "local_recommendations": None,
    }

    if not (args.json and args.no_context):
        result_payload["retrieval_context"] = retrieval_context.to_dict()

    if not (args.json and args.no_llm_prompt):
        result_payload["llm_prompt"] = retrieval_context.to_llm_prompt(recommendation_count=5)

    if not args.json:
        print("\nRetrieved context:\n")
        print(retrieval_context.to_context_string())
        print("\nLLM-ready prompt:\n")
        print(retrieval_context.to_llm_prompt(recommendation_count=5))

    if args.mode in {"llm-only", "hybrid"}:
        if not llm_is_configured():
            if args.mode == "llm-only":
                error_message = "LLM mode requires LLM_API_KEY or OPENAI_API_KEY to be set."
                if args.json:
                    result_payload["error"] = error_message
                    _print_json(result_payload)
                    return
                raise SystemExit(error_message)
        else:
            try:
                llm_recommendation_text = generate_grounded_recommendation_text(
                    retrieval_context,
                    recommendation_count=5,
                )
            except RuntimeError as exc:
                result_payload["error"] = str(exc)
                if not args.json:
                    print("\nLLM recommendation call failed:\n")
                    print(str(exc))
                if args.mode == "llm-only":
                    if args.json:
                        _print_json(result_payload)
                        return
                    raise SystemExit(1) from exc
            else:
                result_payload["llm_recommendations"] = llm_recommendation_text
                if not args.json:
                    print("\nLLM-generated recommendations:\n")
                    print(llm_recommendation_text)

    if args.mode == "retrieval-only":
        if args.json:
            _print_json(result_payload)
        return

    if args.mode == "llm-only":
        if args.json:
            _print_json(result_payload)
        return

    user_prefs = build_user_preferences_from_retrieval_context(retrieval_context)
    candidate_songs = candidate_tracks_to_song_dicts(retrieval_context) or songs

    recommendations = recommend_songs(user_prefs, candidate_songs, k=5, strategy_name=strategy_mode)
    result_payload["local_recommendations"] = _serialize_recommendations(recommendations)

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

    if args.json:
        _print_json(result_payload)
        return

    print("\nTop recommendations:\n")
    print(f"Prompt: {query}")
    print(f"Run mode: {args.mode}")
    print(f"Ranking mode: {strategy_mode}")
    print(f"Available modes: {', '.join(available_strategy_names())}\n")
    print(format_recommendation_table(table_rows))

    if not args.json:
        _offer_spotify_playlist(recommendations, query)


def _offer_spotify_playlist(
    recommendations: list[tuple[dict, float, str]],
    query: str,
) -> None:
    """Interactively offer to create a Spotify playlist from the recommendations."""
    if not spotify_is_configured():
        print(
            "\n(Spotify playlist creation is unavailable — "
            "add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to your .env file.)"
        )
        return

    print("\n" + "-" * 78)
    answer = input("Would you like to create a Spotify playlist from these recommendations? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Skipping playlist creation.")
        return

    # --- Authorization Code flow (local callback server, works with ngrok tunnel) ---
    print("\nSpotify will open in your browser. Log in and click 'Agree' to grant access.")
    try:
        token_data = run_local_auth_flow()
    except RuntimeError as exc:
        print(f"Spotify authorization failed: {exc}")
        return
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    access_token = (token_data.get("access_token") or "").strip()
    if not access_token:
        print("Did not receive an access token. Skipping playlist creation.")
        return

    # Fetch the user's Spotify profile to get their user ID automatically.
    print("Fetching your Spotify profile…")
    try:
        user_profile = get_current_user(access_token)
    except RuntimeError as exc:
        print(f"Could not fetch Spotify profile: {exc}")
        return

    user_id = (user_profile.get("id") or "").strip()
    display_name = user_profile.get("display_name") or user_id
    if not user_id:
        print("Could not determine your Spotify user ID. Skipping playlist creation.")
        return

    print(f"Logged in as: {display_name} ({user_id})")
    granted_scopes = (token_data.get("scope") or "").strip()
    print(f"Token scopes: {granted_scopes or '(none listed)'}")

    default_name = f"AI Picks: {query[:40]}"
    playlist_name = input(f"Playlist name [{default_name}]: ").strip() or default_name

    use_ai = llm_is_configured()
    if use_ai:
        ai_answer = input("Use AI to refine the track selection before building the playlist? [Y/n]: ").strip().lower()
        use_ai = ai_answer not in {"n", "no"}

    if use_ai:
        print("(AI refinement is noted — the recommender scores already incorporate the AI retrieval layer.)")

    print(f"\nBuilding playlist '{playlist_name}'…")
    try:
        playlist = build_playlist_from_recommendations(
            playlist_name=playlist_name,
            recommendations=recommendations,
            access_token=access_token,
            description=f"AI-generated playlist for: {query}",
            public=True,
        )
    except RuntimeError as exc:
        print(f"Playlist creation failed: {exc}")
        return

    track_count = playlist.get("track_count", 0)
    playlist_url = playlist.get("url") or ""
    print(f"\nPlaylist created with {track_count} track(s)!")
    if playlist_url:
        print(f"Open in Spotify: {playlist_url}")


if __name__ == "__main__":
    main()
