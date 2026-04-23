from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.logger import get_logger

logger = get_logger(__name__)


DEFAULT_INPUT_PATH = Path("data/dataset.csv")
DEFAULT_OUTPUT_PATH = Path("data/song_artist_metadata.csv")
REQUIRED_INPUT_COLUMNS = {
    "track_id",
    "track_name",
    "album_name",
    "artists",
    "track_genre",
}


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def split_artists(raw_artists: str) -> List[str]:
    return [artist.strip() for artist in raw_artists.split(";") if artist.strip()]


def make_artist_id(artist_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", artist_name.strip().lower()).strip("-")
    return f"artist:{slug or 'unknown'}"


def _parse_popularity(value: str) -> int | None:
    stripped = (value or "").strip()
    if not stripped:
        return None
    return int(round(float(stripped)))


def build_track_artist_index(dataset_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    track_index: Dict[str, Dict[str, Any]] = {}
    for row in dataset_rows:
        track_id = (row.get("track_id") or "").strip()
        if not track_id:
            continue

        artist_names = split_artists(row.get("artists", ""))
        artist_ids = [make_artist_id(artist_name) for artist_name in artist_names]
        primary_artist_id = artist_ids[0] if artist_ids else ""
        primary_artist_name = artist_names[0] if artist_names else ""

        track_index[track_id] = {
            "primary_artist_id": primary_artist_id,
            "primary_artist_name": primary_artist_name,
            "artist_ids": artist_ids,
            "artist_names": artist_names,
        }
    return track_index


def build_artist_index(dataset_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    artist_index: Dict[str, Dict[str, Any]] = {}
    popularity_totals: Dict[str, int] = {}
    popularity_counts: Dict[str, int] = {}

    for row in dataset_rows:
        track_genre = (row.get("track_genre") or "").strip()
        popularity = _parse_popularity(row.get("popularity", ""))

        for artist_name in split_artists(row.get("artists", "")):
            artist_id = make_artist_id(artist_name)
            artist_metadata = artist_index.setdefault(
                artist_id,
                {
                    "artist_genres": [],
                    "artist_popularity": None,
                    "artist_followers_total": "",
                    "artist_spotify_url": "",
                },
            )

            if track_genre:
                artist_metadata["artist_genres"] = _dedupe_preserve_order(
                    [*artist_metadata["artist_genres"], track_genre]
                )

            if popularity is not None:
                popularity_totals[artist_id] = popularity_totals.get(artist_id, 0) + popularity
                popularity_counts[artist_id] = popularity_counts.get(artist_id, 0) + 1

    for artist_id, artist_metadata in artist_index.items():
        genre_values = artist_metadata["artist_genres"]
        artist_metadata["artist_genres"] = "|".join(genre_values)
        if popularity_counts.get(artist_id):
            artist_metadata["artist_popularity"] = round(
                popularity_totals[artist_id] / popularity_counts[artist_id]
            )

    return artist_index


def build_enriched_rows(
    dataset_rows: List[Dict[str, str]],
    track_artist_index: Dict[str, Dict[str, Any]],
    artist_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched_rows: List[Dict[str, Any]] = []
    for row in dataset_rows:
        track_id = (row.get("track_id") or "").strip()
        track_artist = track_artist_index.get(track_id, {})
        primary_artist_id = track_artist.get("primary_artist_id")
        artist_metadata = artist_index.get(primary_artist_id, {}) if primary_artist_id else {}

        enriched_rows.append(
            {
                "track_id": track_id,
                "track_name": row.get("track_name", ""),
                "album_name": row.get("album_name", ""),
                "artists": row.get("artists", ""),
                "track_genre": row.get("track_genre", ""),
                "primary_artist_id": primary_artist_id or "",
                "primary_artist_name": track_artist.get("primary_artist_name") or "",
                "artist_ids": "|".join(track_artist.get("artist_ids") or []),
                "artist_names": "|".join(track_artist.get("artist_names") or []),
                "artist_genres": artist_metadata.get("artist_genres", ""),
                "artist_popularity": artist_metadata.get("artist_popularity", ""),
                "artist_followers_total": artist_metadata.get("artist_followers_total", ""),
                "artist_spotify_url": artist_metadata.get("artist_spotify_url", ""),
            }
        )
    return enriched_rows


def build_local_artist_metadata_rows(dataset_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    track_artist_index = build_track_artist_index(dataset_rows)
    artist_index = build_artist_index(dataset_rows)
    return build_enriched_rows(dataset_rows, track_artist_index, artist_index)


def read_dataset_rows(input_path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_INPUT_COLUMNS - fieldnames)
        if missing_columns:
            missing_list = ", ".join(missing_columns)
            raise ValueError(
                f"Input file {input_path} is missing required columns: {missing_list}. "
                "Use the raw Spotify dataset CSV, such as data/dataset.csv."
            )
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            rows.append(row)
    return rows


def write_rows(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows were generated for artist metadata output.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(input_path: Path, output_path: Path, limit: int | None = None) -> int:
    logger.info("Reading dataset from %s.", input_path)
    all_dataset_rows = read_dataset_rows(input_path)
    logger.info("Read %d row(s) from dataset.", len(all_dataset_rows))
    dataset_rows = all_dataset_rows[:limit] if limit is not None else all_dataset_rows
    if limit is not None:
        logger.debug("Applying row limit: processing first %d of %d row(s).", len(dataset_rows), len(all_dataset_rows))
    enriched_rows = build_local_artist_metadata_rows(dataset_rows)
    write_rows(output_path, enriched_rows)
    logger.info("Wrote %d enriched row(s) to %s.", len(enriched_rows), output_path)
    return len(enriched_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build artist metadata from the local CSV dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to the source CSV with Spotify track IDs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the output CSV to write artist metadata.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of input rows to process for a smaller test run.")
    args = parser.parse_args()

    row_count = run(args.input, args.output, limit=args.limit)
    print(f"Wrote {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()