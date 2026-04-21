from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol

from src.recommender import load_songs


KNOWN_MOODS = {
    "chill",
    "relaxed",
    "happy",
    "melancholic",
    "moody",
    "uplifting",
    "energetic",
    "intense",
}

ACTIVITY_KEYWORDS = {
    "studying": ("study", "studying", "focus", "focused", "reading", "coding"),
    "working out": ("workout", "working out", "running", "gym", "lifting", "cardio"),
    "sleeping": ("sleep", "sleeping", "bedtime"),
    "commuting": ("commute", "commuting", "driving", "train", "walking"),
}

ACTIVITY_PROFILES = {
    "studying": {
        "preferred_genres": {"study", "lofi", "acoustic", "ambient", "indie"},
        "preferred_moods": {"chill", "relaxed", "melancholic", "moody"},
        "target_energy": 0.35,
        "target_acousticness": 0.75,
        "target_tempo_bpm": 92.0,
        "target_valence": 0.45,
    },
    "working out": {
        "preferred_genres": {"edm", "electronic", "dance", "hip-hop", "work-out"},
        "preferred_moods": {"energetic", "intense", "uplifting"},
        "target_energy": 0.85,
        "target_acousticness": 0.15,
        "target_tempo_bpm": 132.0,
        "target_valence": 0.6,
    },
    "sleeping": {
        "preferred_genres": {"ambient", "sleep", "acoustic", "study"},
        "preferred_moods": {"chill", "relaxed", "melancholic"},
        "target_energy": 0.15,
        "target_acousticness": 0.85,
        "target_tempo_bpm": 72.0,
        "target_valence": 0.3,
    },
    "commuting": {
        "preferred_genres": {"pop", "indie", "rock", "hip-hop"},
        "preferred_moods": {"energetic", "uplifting", "moody"},
        "target_energy": 0.55,
        "target_acousticness": 0.35,
        "target_tempo_bpm": 108.0,
        "target_valence": 0.5,
    },
}

STOPWORD_DESCRIPTORS = {
    "a",
    "an",
    "and",
    "for",
    "i",
    "in",
    "like",
    "similiar",
    "similar",
    "something",
    "that",
    "the",
    "to",
    "want",
    "with",
}

SEED_ARTIST_PATTERNS = (
    r"(?:similar|similiar|like)\s+to\s+([^,.;]+)",
    r"(?:similar|similiar|like)\s+([^,.;]+)",
    r"in\s+the\s+style\s+of\s+([^,.;]+)",
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _split_artists(raw_artist_value: str) -> List[str]:
    return [artist.strip() for artist in str(raw_artist_value or "").split(";") if artist.strip()]


def _average(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _word_boundary_contains(query: str, token: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", query, flags=re.IGNORECASE))


@dataclass
class QuerySignals:
    raw_query: str
    mood: Optional[str] = None
    activity: Optional[str] = None
    seed_artist: Optional[str] = None
    descriptors: List[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtistProfile:
    artist_name: str
    artist_key: str
    genres: List[str]
    moods: List[str]
    dominant_genre: str
    dominant_mood: str
    average_energy: float
    average_tempo_bpm: float
    average_valence: float
    average_danceability: float
    average_acousticness: float
    average_popularity: float
    track_count: int
    primary_track_count: int
    sample_tracks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtistMatch:
    artist_name: str
    similarity_score: float
    reasons: List[str]
    profile: ArtistProfile

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["profile"] = self.profile.to_dict()
        return payload


@dataclass
class CandidateTrack:
    track: Dict[str, Any]
    retrieval_score: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track": self.track,
            "retrieval_score": round(self.retrieval_score, 3),
            "reasons": self.reasons,
        }


@dataclass
class RetrievalContext:
    query: str
    signals: QuerySignals
    seed_artist_profile: Optional[ArtistProfile]
    similar_artists: List[ArtistMatch]
    candidate_tracks: List[CandidateTrack]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "signals": self.signals.to_dict(),
            "seed_artist_profile": self.seed_artist_profile.to_dict() if self.seed_artist_profile else None,
            "similar_artists": [artist.to_dict() for artist in self.similar_artists],
            "candidate_tracks": [track.to_dict() for track in self.candidate_tracks],
        }

    def to_context_string(self) -> str:
        signal_bits = []
        if self.signals.mood:
            signal_bits.append(f"mood={self.signals.mood}")
        if self.signals.activity:
            signal_bits.append(f"activity={self.signals.activity}")
        if self.signals.seed_artist:
            signal_bits.append(f"seed_artist={self.signals.seed_artist}")

        lines = [
            f"User query: {self.query}",
            f"Parsed signals: {', '.join(signal_bits) if signal_bits else 'none detected'}",
        ]

        if self.seed_artist_profile:
            lines.append(
                "Seed artist profile: "
                f"{self.seed_artist_profile.artist_name} | genres={', '.join(self.seed_artist_profile.genres)} | "
                f"avg_energy={self.seed_artist_profile.average_energy:.2f} | "
                f"avg_acousticness={self.seed_artist_profile.average_acousticness:.2f} | "
                f"sample_tracks={', '.join(self.seed_artist_profile.sample_tracks[:3])}"
            )

        if self.similar_artists:
            lines.append("Similar artists:")
            for match in self.similar_artists:
                lines.append(
                    f"- {match.artist_name} ({match.similarity_score:.2f}): {', '.join(match.reasons)}"
                )

        if self.candidate_tracks:
            lines.append("Candidate tracks:")
            for candidate in self.candidate_tracks:
                track = candidate.track
                lines.append(
                    f"- {track.get('title')} by {track.get('artist')} | genre={track.get('genre')} | "
                    f"mood={track.get('mood')} | score={candidate.retrieval_score:.2f} | "
                    f"reasons={', '.join(candidate.reasons)}"
                )

        return "\n".join(lines)

    def to_llm_prompt(self, recommendation_count: int = 5) -> str:
        signal_parts = []
        if self.signals.mood:
            signal_parts.append(f"mood: {self.signals.mood}")
        if self.signals.activity:
            signal_parts.append(f"activity: {self.signals.activity}")
        if self.signals.seed_artist:
            signal_parts.append(f"seed artist: {self.signals.seed_artist}")

        lines = [
            "You are a music recommendation assistant.",
            f"The user asked: {self.query}",
            f"Parsed signals: {', '.join(signal_parts) if signal_parts else 'none detected'}.",
        ]

        if self.seed_artist_profile:
            lines.append(
                "Retrieved seed artist profile: "
                f"{self.seed_artist_profile.artist_name}; "
                f"genres={', '.join(self.seed_artist_profile.genres) or 'unknown'}; "
                f"dominant mood={self.seed_artist_profile.dominant_mood or 'unknown'}; "
                f"tempo≈{self.seed_artist_profile.average_tempo_bpm:.0f} bpm; "
                f"energy={self.seed_artist_profile.average_energy:.2f}; "
                f"valence={self.seed_artist_profile.average_valence:.2f}; "
                f"danceability={self.seed_artist_profile.average_danceability:.2f}; "
                f"acousticness={self.seed_artist_profile.average_acousticness:.2f}; "
                f"sample tracks={', '.join(self.seed_artist_profile.sample_tracks[:3]) or 'unknown'}."
            )

        if self.similar_artists:
            similar_artist_summary = ", ".join(
                f"{match.artist_name} ({'; '.join(match.reasons[:2])})"
                for match in self.similar_artists[:5]
            )
            lines.append(f"Retrieved similar artists: {similar_artist_summary}.")

        if self.candidate_tracks:
            lines.append("Retrieved candidate tracks:")
            for candidate in self.candidate_tracks[: min(10, len(self.candidate_tracks))]:
                track = candidate.track
                lines.append(
                    f"- {track.get('title')} by {track.get('artist')} | genre={track.get('genre')} | "
                    f"mood={track.get('mood')} | tempo≈{float(track.get('tempo_bpm', 0.0) or 0.0):.0f} bpm | "
                    f"energy={float(track.get('energy', 0.0) or 0.0):.2f} | "
                    f"acousticness={float(track.get('acousticness', 0.0) or 0.0):.2f} | "
                    f"retrieval reasons={', '.join(candidate.reasons)}"
                )

        lines.append(
            f"Based only on the user query and the retrieved data above, recommend {recommendation_count} songs and explain each recommendation briefly."
        )
        lines.append("Do not invent retrieved facts that are not present in the context.")
        return "\n".join(lines)


class RetrievalBackend(Protocol):
    def resolve_seed_artist(self, artist_name: str) -> Optional[ArtistProfile]: ...

    def find_similar_artists(
        self,
        seed_profile: ArtistProfile,
        signals: QuerySignals,
        limit: int = 5,
    ) -> List[ArtistMatch]: ...

    def get_candidate_tracks(
        self,
        signals: QuerySignals,
        seed_profile: Optional[ArtistProfile],
        similar_artists: List[ArtistMatch],
        limit: int = 12,
    ) -> List[CandidateTrack]: ...


def parse_query_signals(query: str) -> QuerySignals:
    normalized_query = _normalize_text(query)
    mood = next((candidate for candidate in KNOWN_MOODS if _word_boundary_contains(normalized_query, candidate)), None)

    activity = None
    for activity_name, keywords in ACTIVITY_KEYWORDS.items():
        if any(_word_boundary_contains(normalized_query, keyword) for keyword in keywords):
            activity = activity_name
            break

    seed_artist = None
    for pattern in SEED_ARTIST_PATTERNS:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        seed_artist = match.group(1).strip()
        seed_artist = re.split(r"\b(for|with|while|that)\b", seed_artist, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if seed_artist:
            break

    descriptor_tokens = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z'-]+", query):
        lowered = token.lower()
        if lowered in STOPWORD_DESCRIPTORS:
            continue
        if lowered in KNOWN_MOODS:
            continue
        if any(lowered == keyword for keywords in ACTIVITY_KEYWORDS.values() for keyword in keywords):
            continue
        descriptor_tokens.append(token)

    return QuerySignals(
        raw_query=query,
        mood=mood,
        activity=activity,
        seed_artist=seed_artist,
        descriptors=descriptor_tokens,
    )


class LocalCatalogRetrievalBackend:
    def __init__(self, songs: List[Dict[str, Any]]):
        self.songs = songs
        self.artist_profiles = self._build_artist_profiles(songs)
        self.artist_primary_tracks = self._build_artist_primary_track_index(songs)
        self.artist_profiles_by_key = {
            _normalize_text(profile.artist_name): profile for profile in self.artist_profiles
        }

    def _build_artist_primary_track_index(self, songs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        primary_tracks: Dict[str, List[Dict[str, Any]]] = {}

        for song in songs:
            artists = _split_artists(str(song.get("artist", "")))
            if not artists:
                continue
            primary_artist_key = _normalize_text(artists[0])
            primary_tracks.setdefault(primary_artist_key, []).append(song)

        return primary_tracks

    def _top_labels(self, weighted_counts: Dict[str, float], limit: int = 3) -> List[str]:
        ordered = sorted(weighted_counts.items(), key=lambda item: (-item[1], item[0]))
        return [label for label, _ in ordered[:limit]]

    def _build_artist_profiles(self, songs: List[Dict[str, Any]]) -> List[ArtistProfile]:
        grouped: Dict[str, Dict[str, Any]] = {}

        for song in songs:
            artists = _split_artists(str(song.get("artist", "")))
            if not artists:
                continue

            for artist_name in artists:
                artist_key = _normalize_text(artist_name)
                is_primary_artist = artist_name == artists[0]
                artist_weight = 1.0 if is_primary_artist else 0.2
                bucket = grouped.setdefault(
                    artist_key,
                    {
                        "artist_name": artist_name,
                        "genre_weights": {},
                        "mood_weights": {},
                        "energy": [],
                        "tempo": [],
                        "valence": [],
                        "danceability": [],
                        "acousticness": [],
                        "popularity": [],
                        "sample_tracks": [],
                        "track_count": 0,
                        "primary_track_count": 0,
                    },
                )

                genre = str(song.get("genre", "")).strip()
                if genre:
                    bucket["genre_weights"][genre] = bucket["genre_weights"].get(genre, 0.0) + artist_weight

                mood = str(song.get("mood", "")).strip()
                if mood:
                    bucket["mood_weights"][mood] = bucket["mood_weights"].get(mood, 0.0) + artist_weight

                bucket["energy"].extend([float(song.get("energy", 0.0) or 0.0)] * (5 if is_primary_artist else 1))
                bucket["tempo"].extend([float(song.get("tempo_bpm", 0.0) or 0.0)] * (5 if is_primary_artist else 1))
                bucket["valence"].extend([float(song.get("valence", 0.0) or 0.0)] * (5 if is_primary_artist else 1))
                bucket["danceability"].extend([float(song.get("danceability", 0.0) or 0.0)] * (5 if is_primary_artist else 1))
                bucket["acousticness"].extend([float(song.get("acousticness", 0.0) or 0.0)] * (5 if is_primary_artist else 1))

                popularity = song.get("popularity")
                if popularity is not None:
                    bucket["popularity"].extend([float(popularity)] * (5 if is_primary_artist else 1))

                bucket["track_count"] += 1
                if is_primary_artist:
                    bucket["primary_track_count"] += 1

                title = str(song.get("title", "")).strip()
                if title and title not in bucket["sample_tracks"] and is_primary_artist:
                    bucket["sample_tracks"].append(title)

        profiles = []
        for artist_key, bucket in grouped.items():
            genres = self._top_labels(bucket["genre_weights"])
            moods = self._top_labels(bucket["mood_weights"])
            profiles.append(
                ArtistProfile(
                    artist_name=bucket["artist_name"],
                    artist_key=artist_key,
                    genres=genres,
                    moods=moods,
                    dominant_genre=genres[0] if genres else "",
                    dominant_mood=moods[0] if moods else "",
                    average_energy=_average(bucket["energy"]),
                    average_tempo_bpm=_average(bucket["tempo"]),
                    average_valence=_average(bucket["valence"]),
                    average_danceability=_average(bucket["danceability"]),
                    average_acousticness=_average(bucket["acousticness"]),
                    average_popularity=_average(bucket["popularity"]),
                    track_count=bucket["track_count"],
                    primary_track_count=bucket["primary_track_count"],
                    sample_tracks=(bucket["sample_tracks"] or [])[:5],
                )
            )

        return profiles

    def _song_similarity(self, song: Dict[str, Any], seed_song: Dict[str, Any]) -> float:
        score = 0.0

        if song.get("genre") == seed_song.get("genre"):
            score += 2.5
        if song.get("mood") == seed_song.get("mood"):
            score += 1.5

        score += 2.0 * (1 - abs(float(song.get("energy", 0.0) or 0.0) - float(seed_song.get("energy", 0.0) or 0.0)))
        score += 1.5 * (1 - min(abs(float(song.get("tempo_bpm", 0.0) or 0.0) - float(seed_song.get("tempo_bpm", 0.0) or 0.0)) / 100.0, 1.0))
        score += 1.5 * (1 - abs(float(song.get("valence", 0.0) or 0.0) - float(seed_song.get("valence", 0.0) or 0.0)))
        score += 1.5 * (1 - abs(float(song.get("acousticness", 0.0) or 0.0) - float(seed_song.get("acousticness", 0.0) or 0.0)))
        score += 1.0 * (1 - abs(float(song.get("danceability", 0.0) or 0.0) - float(seed_song.get("danceability", 0.0) or 0.0)))
        return score

    def _track_neighborhood_support(self, candidate: ArtistProfile, seed: ArtistProfile) -> float:
        candidate_tracks = self.artist_primary_tracks.get(candidate.artist_key, [])
        seed_tracks = self.artist_primary_tracks.get(seed.artist_key, [])
        if not candidate_tracks or not seed_tracks:
            return 0.0

        best_scores = []
        for seed_song in seed_tracks[:8]:
            best_match = max(self._song_similarity(candidate_song, seed_song) for candidate_song in candidate_tracks[:12])
            best_scores.append(best_match)

        top_scores = sorted(best_scores, reverse=True)[:3]
        return _average(top_scores)

    def resolve_seed_artist(self, artist_name: str) -> Optional[ArtistProfile]:
        normalized_artist = _normalize_text(artist_name)
        exact_match = self.artist_profiles_by_key.get(normalized_artist)
        if exact_match:
            return exact_match

        close_matches = difflib.get_close_matches(
            normalized_artist,
            list(self.artist_profiles_by_key.keys()),
            n=1,
            cutoff=0.7,
        )
        if close_matches:
            return self.artist_profiles_by_key[close_matches[0]]

        partial_matches = [
            profile
            for key, profile in self.artist_profiles_by_key.items()
            if normalized_artist in key or key in normalized_artist
        ]
        if partial_matches:
            partial_matches.sort(key=lambda profile: profile.track_count, reverse=True)
            return partial_matches[0]

        return None

    def _artist_similarity_score(self, candidate: ArtistProfile, seed: ArtistProfile) -> float:
        genre_overlap = len(set(candidate.genres) & set(seed.genres))
        mood_overlap = len(set(candidate.moods) & set(seed.moods))
        dominant_genre_match = 1.0 if candidate.dominant_genre and candidate.dominant_genre == seed.dominant_genre else 0.0
        dominant_mood_match = 1.0 if candidate.dominant_mood and candidate.dominant_mood == seed.dominant_mood else 0.0
        feature_similarity = _average(
            [
                1 - abs(candidate.average_energy - seed.average_energy),
                1 - min(abs(candidate.average_tempo_bpm - seed.average_tempo_bpm) / 120.0, 1.0),
                1 - abs(candidate.average_valence - seed.average_valence),
                1 - abs(candidate.average_danceability - seed.average_danceability),
                1 - abs(candidate.average_acousticness - seed.average_acousticness),
            ]
        )
        popularity_similarity = 1 - min(abs(candidate.average_popularity - seed.average_popularity) / 100.0, 1.0)
        neighborhood_support = self._track_neighborhood_support(candidate, seed)
        profile_confidence = min(candidate.primary_track_count / max(seed.primary_track_count, 1), 1.0)
        if candidate.primary_track_count < 2:
            profile_confidence *= 0.7

        score = (
            dominant_genre_match * 4.0
            + genre_overlap * 2.5
            + dominant_mood_match * 2.0
            + mood_overlap * 1.5
            + feature_similarity * 2.0
            + popularity_similarity * 0.5
            + neighborhood_support * 1.2
        )
        return score * max(profile_confidence, 0.35)

    def find_similar_artists(
        self,
        seed_profile: ArtistProfile,
        signals: QuerySignals,
        limit: int = 5,
    ) -> List[ArtistMatch]:
        matches: List[ArtistMatch] = []

        for candidate in self.artist_profiles:
            if candidate.artist_key == seed_profile.artist_key:
                continue

            neighborhood_support = self._track_neighborhood_support(candidate, seed_profile)
            shared_genres = sorted(set(candidate.genres) & set(seed_profile.genres))
            if not shared_genres and neighborhood_support < 7.0:
                continue

            similarity_score = self._artist_similarity_score(candidate, seed_profile)
            reasons = []
            if shared_genres:
                reasons.append(f"shared genres: {', '.join(shared_genres[:2])}")
            if candidate.dominant_genre and candidate.dominant_genre == seed_profile.dominant_genre:
                reasons.append(f"same dominant genre: {candidate.dominant_genre}")
            if candidate.dominant_mood and candidate.dominant_mood == seed_profile.dominant_mood:
                reasons.append(f"same dominant mood: {candidate.dominant_mood}")
            if abs(candidate.average_acousticness - seed_profile.average_acousticness) < 0.15:
                reasons.append("similar acousticness")
            if abs(candidate.average_energy - seed_profile.average_energy) < 0.15:
                reasons.append("similar energy")
            if signals.mood and signals.mood in candidate.moods:
                reasons.append(f"matches requested mood: {signals.mood}")
            if neighborhood_support >= 6.5:
                reasons.append("seed-adjacent track neighborhood")

            if similarity_score <= 0:
                continue

            matches.append(
                ArtistMatch(
                    artist_name=candidate.artist_name,
                    similarity_score=similarity_score,
                    reasons=reasons or ["similar local catalog profile"],
                    profile=candidate,
                )
            )

        matches.sort(key=lambda match: match.similarity_score, reverse=True)
        return matches[:limit]

    def get_candidate_tracks(
        self,
        signals: QuerySignals,
        seed_profile: Optional[ArtistProfile],
        similar_artists: List[ArtistMatch],
        limit: int = 12,
    ) -> List[CandidateTrack]:
        activity_profile = ACTIVITY_PROFILES.get(signals.activity or "")
        similar_artist_keys = {match.profile.artist_key for match in similar_artists}
        candidates: List[CandidateTrack] = []

        for song in self.songs:
            score = 0.0
            reasons: List[str] = []
            primary_artist = _split_artists(str(song.get("artist", "")))[:1]
            primary_artist_key = _normalize_text(primary_artist[0]) if primary_artist else ""
            shared_seed_genre = bool(seed_profile and song.get("genre") in seed_profile.genres)
            activity_fit = bool(
                activity_profile
                and song.get("genre") in activity_profile["preferred_genres"]
                and song.get("mood") in activity_profile["preferred_moods"]
            )
            related_to_seed = bool(
                seed_profile and (primary_artist_key == seed_profile.artist_key or primary_artist_key in similar_artist_keys)
            )

            if seed_profile and not related_to_seed and not shared_seed_genre and not activity_fit:
                continue

            if seed_profile and primary_artist_key == seed_profile.artist_key:
                score += 10.0
                reasons.append("seed artist track")
            elif primary_artist_key in similar_artist_keys:
                score += 7.0
                reasons.append("track from similar artist")

            if signals.mood and song.get("mood") == signals.mood:
                score += 2.5
                reasons.append(f"matches requested mood: {signals.mood}")

            if shared_seed_genre:
                score += 4.0
                reasons.append("shares seed-artist genre")

            if activity_profile:
                if song.get("genre") in activity_profile["preferred_genres"]:
                    score += 1.5
                    reasons.append(f"fits {signals.activity} genre profile")
                if song.get("mood") in activity_profile["preferred_moods"]:
                    score += 1.0
                    reasons.append(f"fits {signals.activity} mood profile")

            target_energy = seed_profile.average_energy if seed_profile else None
            target_tempo = seed_profile.average_tempo_bpm if seed_profile else None
            target_valence = seed_profile.average_valence if seed_profile else None
            target_acousticness = seed_profile.average_acousticness if seed_profile else None

            if activity_profile:
                target_energy = _average([
                    value for value in [target_energy, activity_profile["target_energy"]] if value is not None
                ])
                target_tempo = _average([
                    value for value in [target_tempo, activity_profile["target_tempo_bpm"]] if value is not None
                ])
                target_valence = _average([
                    value for value in [target_valence, activity_profile["target_valence"]] if value is not None
                ])
                target_acousticness = _average([
                    value for value in [target_acousticness, activity_profile["target_acousticness"]] if value is not None
                ])

            if target_energy is not None:
                score += 2.0 * (1 - abs(float(song.get("energy", 0.0) or 0.0) - target_energy))
            if target_tempo is not None:
                score += 1.5 * (1 - min(abs(float(song.get("tempo_bpm", 0.0) or 0.0) - target_tempo) / 100.0, 1.0))
            if target_valence is not None:
                score += 1.0 * (1 - abs(float(song.get("valence", 0.0) or 0.0) - target_valence))
            if target_acousticness is not None:
                score += 1.5 * (1 - abs(float(song.get("acousticness", 0.0) or 0.0) - target_acousticness))

            if score <= 0:
                continue

            candidates.append(CandidateTrack(track=song, retrieval_score=score, reasons=reasons or ["overall profile match"]))

        candidates.sort(key=lambda candidate: candidate.retrieval_score, reverse=True)
        return candidates[:limit]


class RetrievalLayer:
    def __init__(self, songs: List[Dict[str, Any]], backend: Optional[RetrievalBackend] = None):
        self.songs = songs
        self.backend = backend or LocalCatalogRetrievalBackend(songs)

    @classmethod
    def from_csv(cls, csv_path: str = "data/songs_dataset_full.csv") -> "RetrievalLayer":
        return cls(load_songs(csv_path))

    def retrieve(
        self,
        query: str,
        candidate_limit: int = 12,
        similar_artist_limit: int = 5,
    ) -> RetrievalContext:
        signals = parse_query_signals(query)
        seed_profile = self.backend.resolve_seed_artist(signals.seed_artist) if signals.seed_artist else None
        similar_artists = (
            self.backend.find_similar_artists(seed_profile, signals, limit=similar_artist_limit)
            if seed_profile
            else []
        )
        candidate_tracks = self.backend.get_candidate_tracks(
            signals,
            seed_profile,
            similar_artists,
            limit=candidate_limit,
        )

        return RetrievalContext(
            query=query,
            signals=signals,
            seed_artist_profile=seed_profile,
            similar_artists=similar_artists,
            candidate_tracks=candidate_tracks,
        )


def _tempo_category(tempo_bpm: float) -> str:
    if tempo_bpm < 70:
        return "Slow"
    if tempo_bpm < 111:
        return "Mid"
    if tempo_bpm < 141:
        return "Upbeat"
    return "Fast"


def _energy_level(energy: float) -> str:
    if energy < 0.25:
        return "Low"
    if energy < 0.5:
        return "Medium"
    if energy < 0.75:
        return "High"
    return "Intense"


def _danceability_tier(danceability: float) -> str:
    if danceability < 0.25:
        return "Non-danceable"
    if danceability < 0.5:
        return "Moderate"
    if danceability < 0.75:
        return "Danceable"
    return "Club-ready"


def _derive_target_from_candidates(candidates: List[CandidateTrack], key: str) -> float:
    values = [float(candidate.track.get(key, 0.0) or 0.0) for candidate in candidates]
    return _average(values)


def build_user_preferences_from_retrieval_context(context: RetrievalContext) -> Dict[str, Any]:
    activity_profile = ACTIVITY_PROFILES.get(context.signals.activity or "")
    candidates = context.candidate_tracks
    seed_profile = context.seed_artist_profile

    favorite_genre = ""
    if seed_profile and seed_profile.genres:
        favorite_genre = seed_profile.genres[0]
    elif candidates:
        favorite_genre = str(candidates[0].track.get("genre", ""))
    elif activity_profile:
        favorite_genre = sorted(activity_profile["preferred_genres"])[0]

    favorite_mood = context.signals.mood or ""
    if not favorite_mood and seed_profile and seed_profile.moods:
        favorite_mood = seed_profile.moods[0]
    if not favorite_mood and activity_profile:
        favorite_mood = sorted(activity_profile["preferred_moods"])[0]
    if not favorite_mood:
        favorite_mood = "chill"

    candidate_genre = str(candidates[0].track.get("genre", "")) if candidates else ""
    favorite_detailed_mood = str(candidates[0].track.get("detailed_mood", "") or "") if candidates else ""

    target_energy = seed_profile.average_energy if seed_profile else 0.0
    target_tempo = seed_profile.average_tempo_bpm if seed_profile else 0.0
    target_valence = seed_profile.average_valence if seed_profile else 0.0
    target_danceability = seed_profile.average_danceability if seed_profile else 0.0
    target_popularity = seed_profile.average_popularity if seed_profile else 0.0
    target_acousticness = seed_profile.average_acousticness if seed_profile else 0.0

    if candidates:
        target_energy = _average([target_energy, _derive_target_from_candidates(candidates, "energy")]) if seed_profile else _derive_target_from_candidates(candidates, "energy")
        target_tempo = _average([target_tempo, _derive_target_from_candidates(candidates, "tempo_bpm")]) if seed_profile else _derive_target_from_candidates(candidates, "tempo_bpm")
        target_valence = _average([target_valence, _derive_target_from_candidates(candidates, "valence")]) if seed_profile else _derive_target_from_candidates(candidates, "valence")
        target_danceability = _average([target_danceability, _derive_target_from_candidates(candidates, "danceability")]) if seed_profile else _derive_target_from_candidates(candidates, "danceability")
        target_popularity = _average([target_popularity, _derive_target_from_candidates(candidates, "popularity")]) if seed_profile else _derive_target_from_candidates(candidates, "popularity")
        target_acousticness = _average([target_acousticness, _derive_target_from_candidates(candidates, "acousticness")]) if seed_profile else _derive_target_from_candidates(candidates, "acousticness")

    if activity_profile:
        target_energy = _average([target_energy, activity_profile["target_energy"]])
        target_tempo = _average([target_tempo, activity_profile["target_tempo_bpm"]])
        target_valence = _average([target_valence, activity_profile["target_valence"]])
        target_acousticness = _average([target_acousticness, activity_profile["target_acousticness"]])

    likes_acoustic = target_acousticness >= 0.5

    weight_genre = 6.0 if seed_profile else 3.5
    weight_mood = 7.0 if context.signals.mood else 4.0
    weight_energy = 4.5
    weight_tempo = 3.0
    weight_valence = 2.5
    weight_danceability = 2.0
    weight_acousticness = 5.0 if likes_acoustic else 3.0
    weight_popularity = 1.0
    weight_detailed_mood = 2.0 if favorite_detailed_mood else 0.0
    weight_energy_level = 1.5
    weight_danceability_tier = 1.0
    weight_tempo_category = 1.5

    if context.signals.activity == "studying":
        weight_acousticness = max(weight_acousticness, 5.5)
        weight_energy = 5.0
        weight_tempo = 4.0
    elif context.signals.activity == "working out":
        weight_energy = 7.0
        weight_tempo = 6.0
        weight_danceability = 4.0

    return {
        "favorite_genre": favorite_genre or candidate_genre,
        "favorite_mood": favorite_mood,
        "target_energy": target_energy,
        "likes_acoustic": likes_acoustic,
        "favorite_detailed_mood": favorite_detailed_mood or None,
        "preferred_energy_level": _energy_level(target_energy),
        "preferred_danceability_tier": _danceability_tier(target_danceability),
        "preferred_tempo_category": _tempo_category(target_tempo),
        "target_tempo": target_tempo,
        "target_valence": target_valence,
        "target_danceability": target_danceability,
        "target_popularity": target_popularity,
        "weight_genre": weight_genre,
        "weight_mood": weight_mood,
        "weight_detailed_mood": weight_detailed_mood,
        "weight_energy": weight_energy,
        "weight_tempo": weight_tempo,
        "weight_valence": weight_valence,
        "weight_danceability": weight_danceability,
        "weight_acousticness": weight_acousticness,
        "weight_energy_level": weight_energy_level,
        "weight_danceability_tier": weight_danceability_tier,
        "weight_tempo_category": weight_tempo_category,
        "weight_popularity": weight_popularity,
        "diversity_artist_penalty": 4.0,
        "diversity_genre_penalty": 1.0,
    }


def candidate_tracks_to_song_dicts(context: RetrievalContext) -> List[Dict[str, Any]]:
    return [candidate.track for candidate in context.candidate_tracks]


def build_retrieval_context(
    query: str,
    csv_path: str = "data/songs_dataset_full.csv",
    candidate_limit: int = 12,
    similar_artist_limit: int = 5,
) -> RetrievalContext:
    retrieval_layer = RetrievalLayer.from_csv(csv_path)
    return retrieval_layer.retrieve(
        query,
        candidate_limit=candidate_limit,
        similar_artist_limit=similar_artist_limit,
    )