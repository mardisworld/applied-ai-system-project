from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spotify_enrichment import (
    build_artist_index,
    build_enriched_rows,
    build_local_artist_metadata_rows,
    build_track_artist_index,
    make_artist_id,
    read_dataset_rows,
    split_artists,
)


def test_split_artists_trims_and_filters_empty_values():
    assert split_artists(" Artist One ; ; Artist Two ") == ["Artist One", "Artist Two"]


def test_make_artist_id_slugifies_artist_names():
    assert make_artist_id("A Great Big World") == "artist:a-great-big-world"


def test_build_track_artist_index_keeps_primary_and_all_artists():
    dataset_rows = [
        {
            "track_id": "track-1",
            "artists": "Artist One;Artist Two",
        }
    ]

    track_artist_index = build_track_artist_index(dataset_rows)

    assert track_artist_index["track-1"]["primary_artist_id"] == "artist:artist-one"
    assert track_artist_index["track-1"]["artist_ids"] == ["artist:artist-one", "artist:artist-two"]
    assert track_artist_index["track-1"]["artist_names"] == ["Artist One", "Artist Two"]


def test_build_artist_index_aggregates_genres_and_average_popularity():
    dataset_rows = [
        {
            "artists": "Artist One;Artist Two",
            "track_genre": "pop",
            "popularity": "80",
        },
        {
            "artists": "Artist One",
            "track_genre": "dance",
            "popularity": "60",
        }
    ]

    artist_index = build_artist_index(dataset_rows)

    assert artist_index["artist:artist-one"]["artist_genres"] == "pop|dance"
    assert artist_index["artist:artist-one"]["artist_popularity"] == 70
    assert artist_index["artist:artist-two"]["artist_genres"] == "pop"
    assert artist_index["artist:artist-two"]["artist_followers_total"] == ""


def test_build_enriched_rows_merges_local_artist_metadata_by_primary_artist():
    dataset_rows = [
        {
            "track_id": "track-1",
            "track_name": "Song One",
            "album_name": "Album One",
            "artists": "Artist One;Artist Two",
            "track_genre": "pop",
        },
        {
            "track_id": "track-2",
            "track_name": "Song Two",
            "album_name": "Album Two",
            "artists": "Artist Three",
            "track_genre": "indie",
        },
    ]
    track_artist_index = build_track_artist_index(dataset_rows)
    artist_index = build_artist_index(
        [
            *dataset_rows,
            {
                "track_id": "track-3",
                "track_name": "Song Three",
                "album_name": "Album Three",
                "artists": "Artist One",
                "track_genre": "dance",
                "popularity": "76",
            },
        ]
    )

    enriched_rows = build_enriched_rows(dataset_rows, track_artist_index, artist_index)

    assert len(enriched_rows) == 2
    assert enriched_rows[0]["primary_artist_id"] == "artist:artist-one"
    assert enriched_rows[0]["artist_ids"] == "artist:artist-one|artist:artist-two"
    assert enriched_rows[0]["artist_genres"] == "pop|dance"
    assert enriched_rows[0]["artist_spotify_url"] == ""
    assert enriched_rows[1]["primary_artist_id"] == "artist:artist-three"


def test_build_local_artist_metadata_rows_uses_only_local_dataset_fields():
    dataset_rows = [
        {
            "track_id": "track-1",
            "track_name": "Song One",
            "album_name": "Album One",
            "artists": "Artist One;Artist Two",
            "track_genre": "pop",
            "popularity": "80",
        },
        {
            "track_id": "track-2",
            "track_name": "Song Two",
            "album_name": "Album Two",
            "artists": "Artist One",
            "track_genre": "dance",
            "popularity": "60",
        },
    ]

    enriched_rows = build_local_artist_metadata_rows(dataset_rows)

    assert enriched_rows[0]["primary_artist_name"] == "Artist One"
    assert enriched_rows[0]["artist_genres"] == "pop|dance"
    assert enriched_rows[0]["artist_popularity"] == 70
    assert enriched_rows[0]["artist_followers_total"] == ""


def test_read_dataset_rows_rejects_input_without_track_id_columns(tmp_path):
    input_path = tmp_path / "songs_dataset_full.csv"
    input_path.write_text(
        "id,title,artist,genre\n1,Song One,Artist One,pop\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        read_dataset_rows(input_path)
