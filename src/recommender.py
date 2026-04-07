from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

SongLike = Union["Song", Dict[str, Any]]
UserLike = Union["UserProfile", Dict[str, Any]]

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: Optional[int] = None
    detailed_mood: Optional[str] = None
    energy_level: Optional[str] = None
    danceability_tier: Optional[str] = None
    tempo_category: Optional[str] = None

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_tempo: float = 0.0
    target_valence: float = 0.0
    target_danceability: float = 0.0
    favorite_detailed_mood: Optional[str] = None
    preferred_energy_level: Optional[str] = None
    preferred_danceability_tier: Optional[str] = None
    preferred_tempo_category: Optional[str] = None
    target_popularity: float = 0.0
    weight_genre: float = 10.0
    weight_mood: float = 5.0
    weight_energy: float = 5.0
    weight_tempo: float = 0.0
    weight_valence: float = 0.0
    weight_danceability: float = 0.0
    weight_acousticness: float = 5.0
    weight_detailed_mood: float = 0.0
    weight_energy_level: float = 0.0
    weight_danceability_tier: float = 0.0
    weight_tempo_category: float = 0.0
    weight_popularity: float = 0.0
    diversity_artist_penalty: float = 6.0
    diversity_genre_penalty: float = 1.5


def _song_value(song: SongLike, key: str, default: Any = None) -> Any:
    if isinstance(song, dict):
        return song.get(key, default)
    return getattr(song, key, default)


def _user_value(user: UserLike, key: str, default: Any = None) -> Any:
    if isinstance(user, dict):
        return user.get(key, default)
    return getattr(user, key, default)


def _lookup_user_preference(user: UserLike, key: str, fallback: Optional[str] = None) -> Any:
    value = _user_value(user, key)
    if value is not None and value != "":
        return value
    if fallback:
        return _user_value(user, fallback)
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_diversity_penalty(song: SongLike, selected_songs: List[SongLike], user: UserLike) -> float:
    penalty = 0.0
    artist = _song_value(song, "artist")
    genre = _song_value(song, "genre")

    selected_artists = {_song_value(selected_song, "artist") for selected_song in selected_songs}
    selected_genres = {_song_value(selected_song, "genre") for selected_song in selected_songs}

    if artist and artist in selected_artists:
        penalty += _to_float(_user_value(user, "diversity_artist_penalty", 6.0), 6.0)
    if genre and genre in selected_genres:
        penalty += _to_float(_user_value(user, "diversity_genre_penalty", 1.5), 1.5)

    return penalty


def _select_top_k_with_diversity(
    songs: List[SongLike],
    user: UserLike,
    strategy: RankingStrategy,
    k: int,
) -> List[Tuple[SongLike, float]]:
    remaining = [(song, strategy.score_song(song, user)) for song in songs]
    selected: List[Tuple[SongLike, float]] = []

    while remaining and len(selected) < k:
        selected_songs = [song for song, _ in selected]
        best_index = 0
        best_adjusted_score = None

        for index, (song, base_score) in enumerate(remaining):
            adjusted_score = base_score - _apply_diversity_penalty(song, selected_songs, user)
            if best_adjusted_score is None or adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_index = index

        selected.append(remaining.pop(best_index))

    return selected


class RankingStrategy(ABC):
    """
    Strategy interface for switching ranking behavior without changing callers.
    """

    name = "balanced"
    description = "Balanced weighted ranking using the user profile weights as-is."
    component_multipliers: Dict[str, float] = {}

    def multiplier(self, component: str) -> float:
        return self.component_multipliers.get(component, 1.0)

    def score_song(self, song: SongLike, user: UserLike) -> float:
        score = 0.0

        if _song_value(song, "genre") == _lookup_user_preference(user, "favorite_genre", "genre"):
            score += _to_float(_user_value(user, "weight_genre", 10.0)) * self.multiplier("genre")
        if _song_value(song, "mood") == _lookup_user_preference(user, "favorite_mood", "mood"):
            score += _to_float(_user_value(user, "weight_mood", 5.0)) * self.multiplier("mood")
        if _lookup_user_preference(user, "favorite_detailed_mood") and _song_value(song, "detailed_mood") == _lookup_user_preference(user, "favorite_detailed_mood"):
            score += _to_float(_user_value(user, "weight_detailed_mood", 0.0)) * self.multiplier("detailed_mood")

        energy_target = _to_float(_user_value(user, "target_energy", _user_value(user, "energy", 0.0)))
        energy_diff = abs(_to_float(_song_value(song, "energy", 0.0)) - energy_target)
        score += _to_float(_user_value(user, "weight_energy", 5.0)) * self.multiplier("energy") * (1 - energy_diff)

        tempo_target = _to_float(_user_value(user, "target_tempo", 0.0))
        tempo_diff = abs(_to_float(_song_value(song, "tempo_bpm", 0.0)) - tempo_target)
        score += _to_float(_user_value(user, "weight_tempo", 0.0)) * self.multiplier("tempo") * (1 - min(tempo_diff / 140, 1))

        valence_diff = abs(_to_float(_song_value(song, "valence", 0.0)) - _to_float(_user_value(user, "target_valence", 0.0)))
        score += _to_float(_user_value(user, "weight_valence", 0.0)) * self.multiplier("valence") * (1 - valence_diff)

        danceability_diff = abs(_to_float(_song_value(song, "danceability", 0.0)) - _to_float(_user_value(user, "target_danceability", 0.0)))
        score += _to_float(_user_value(user, "weight_danceability", 0.0)) * self.multiplier("danceability") * (1 - danceability_diff)

        likes_acoustic = bool(_user_value(user, "likes_acoustic", False))
        acousticness = _to_float(_song_value(song, "acousticness", 0.0))
        acoustic_score = acousticness if likes_acoustic else 1 - acousticness
        score += _to_float(_user_value(user, "weight_acousticness", 5.0)) * self.multiplier("acousticness") * acoustic_score

        if _lookup_user_preference(user, "preferred_energy_level") and _song_value(song, "energy_level") == _lookup_user_preference(user, "preferred_energy_level"):
            score += _to_float(_user_value(user, "weight_energy_level", 0.0)) * self.multiplier("energy_level")
        if _lookup_user_preference(user, "preferred_danceability_tier") and _song_value(song, "danceability_tier") == _lookup_user_preference(user, "preferred_danceability_tier"):
            score += _to_float(_user_value(user, "weight_danceability_tier", 0.0)) * self.multiplier("danceability_tier")
        if _lookup_user_preference(user, "preferred_tempo_category") and _song_value(song, "tempo_category") == _lookup_user_preference(user, "preferred_tempo_category"):
            score += _to_float(_user_value(user, "weight_tempo_category", 0.0)) * self.multiplier("tempo_category")

        popularity = _song_value(song, "popularity")
        if popularity is not None and _to_float(_user_value(user, "weight_popularity", 0.0)) > 0:
            popularity_target = _to_float(_user_value(user, "target_popularity", 0.0))
            popularity_diff = abs(_to_float(popularity) - popularity_target) / 100.0
            score += _to_float(_user_value(user, "weight_popularity", 0.0)) * self.multiplier("popularity") * (1 - min(popularity_diff, 1))

        return score


class BalancedStrategy(RankingStrategy):
    name = "balanced"
    description = "Balanced weighted ranking using the user profile weights as-is."


class GenreFirstStrategy(RankingStrategy):
    name = "genre-first"
    description = "Prioritizes genre alignment before other attributes."
    component_multipliers = {
        "genre": 3.0,
        "mood": 1.25,
        "detailed_mood": 1.1,
        "energy": 0.75,
        "tempo": 0.5,
        "valence": 0.5,
        "danceability": 0.5,
        "energy_level": 0.75,
        "tempo_category": 0.75,
    }


class MoodFirstStrategy(RankingStrategy):
    name = "mood-first"
    description = "Prioritizes mood and emotional feel before genre."
    component_multipliers = {
        "genre": 0.5,
        "mood": 3.0,
        "detailed_mood": 2.0,
        "valence": 1.5,
        "tempo": 1.25,
        "tempo_category": 1.5,
    }


class EnergyFocusedStrategy(RankingStrategy):
    name = "energy-focused"
    description = "Prioritizes energy, tempo, and momentum for high-intensity matches."
    component_multipliers = {
        "genre": 0.5,
        "mood": 0.75,
        "energy": 3.0,
        "tempo": 2.0,
        "valence": 1.25,
        "danceability": 1.5,
        "energy_level": 2.0,
        "tempo_category": 1.5,
    }


STRATEGY_REGISTRY = {
    BalancedStrategy.name: BalancedStrategy,
    GenreFirstStrategy.name: GenreFirstStrategy,
    MoodFirstStrategy.name: MoodFirstStrategy,
    EnergyFocusedStrategy.name: EnergyFocusedStrategy,
}


def get_ranking_strategy(strategy_name: str = "balanced") -> RankingStrategy:
    strategy_class = STRATEGY_REGISTRY.get(strategy_name, BalancedStrategy)
    return strategy_class()


def available_strategy_names() -> List[str]:
    return list(STRATEGY_REGISTRY.keys())

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song], strategy: Optional[RankingStrategy] = None):
        self.songs = songs
        self.strategy = strategy or BalancedStrategy()

    def _score_song(self, song: Song, user: UserProfile) -> float:
        return self.strategy.score_song(song, user)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Ranking Rule: Scores all songs, sorts by score descending, returns top k.
        """
        selected_songs = _select_top_k_with_diversity(self.songs, user, self.strategy, k)
        return [song for song, _ in selected_songs]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append("matches your favorite genre")
        if song.mood == user.favorite_mood:
            reasons.append("matches your favorite mood")
        if user.favorite_detailed_mood and song.detailed_mood == user.favorite_detailed_mood:
            reasons.append("matches your detailed mood")
        
        energy_diff = abs(song.energy - user.target_energy)
        if energy_diff < 0.2:
            reasons.append("has similar energy level")
        
        tempo_diff = abs(song.tempo_bpm - user.target_tempo)
        if tempo_diff < 20:
            reasons.append("has similar tempo")
        
        valence_diff = abs(song.valence - user.target_valence)
        if valence_diff < 0.2:
            reasons.append("has similar valence")
        
        danceability_diff = abs(song.danceability - user.target_danceability)
        if danceability_diff < 0.2:
            reasons.append("has similar danceability")
        
        if user.likes_acoustic and song.acousticness > 0.5:
            reasons.append("is acoustic")
        elif not user.likes_acoustic and song.acousticness < 0.5:
            reasons.append("is not acoustic")
        
        if user.preferred_energy_level and song.energy_level == user.preferred_energy_level:
            reasons.append("matches your energy level category")
        if user.preferred_danceability_tier and song.danceability_tier == user.preferred_danceability_tier:
            reasons.append("matches your danceability tier")
        if user.preferred_tempo_category and song.tempo_category == user.preferred_tempo_category:
            reasons.append("matches your tempo category")
        
        if reasons:
            return f"This song {', '.join(reasons)}."
        else:
            return "This song was recommended based on overall similarity."


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src.main.py
    """
    import csv

    def to_float(value: Optional[str]) -> float:
        if value is None or value == "":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0

    def to_int(value: Optional[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            song = {
                "id": to_int(row.get("id")) or 0,
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "genre": row.get("genre", ""),
                "mood": row.get("mood", ""),
                "energy": to_float(row.get("energy")) or 0.0,
                "tempo_bpm": to_float(row.get("tempo_bpm") or row.get("tempo")) or 0.0,
                "valence": to_float(row.get("valence")) or 0.0,
                "danceability": to_float(row.get("danceability")) or 0.0,
                "acousticness": to_float(row.get("acousticness")) or 0.0,
                "popularity": to_int(row.get("popularity")),
                "detailed_mood": row.get("detailed_mood"),
                "energy_level": row.get("energy_level"),
                "danceability_tier": row.get("danceability_tier"),
                "tempo_category": row.get("tempo_category"),
            }
            songs.append(song)
    return songs


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    strategy_name: str = "balanced",
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src.main.py
    Uses the same weighted Scoring Rule and Ranking Rule as the OOP version.
    """
    strategy = get_ranking_strategy(strategy_name)

    def explain(song: Dict) -> str:
        reasons = []
        if song.get("genre") == _lookup_user_preference(user_prefs, "favorite_genre", "genre"):
            reasons.append("matches your favorite genre")
        if song.get("mood") == _lookup_user_preference(user_prefs, "favorite_mood", "mood"):
            reasons.append("matches your favorite mood")
        if _lookup_user_preference(user_prefs, "favorite_detailed_mood") and song.get("detailed_mood") == _lookup_user_preference(user_prefs, "favorite_detailed_mood"):
            reasons.append("matches your detailed mood")

        energy_target = _to_float(user_prefs.get("target_energy", user_prefs.get("energy", 0.0)))
        energy_diff = abs(song.get("energy", 0.0) - energy_target)
        if energy_diff < 0.2:
            reasons.append("has similar energy level")

        tempo_diff = abs(song.get("tempo_bpm", 0.0) - _to_float(user_prefs.get("target_tempo", 0.0)))
        if tempo_diff < 20:
            reasons.append("has similar tempo")

        valence_diff = abs(song.get("valence", 0.0) - _to_float(user_prefs.get("target_valence", 0.0)))
        if valence_diff < 0.2:
            reasons.append("has similar valence")

        danceability_diff = abs(song.get("danceability", 0.0) - _to_float(user_prefs.get("target_danceability", 0.0)))
        if danceability_diff < 0.2:
            reasons.append("has similar danceability")

        likes_acoustic = user_prefs.get("likes_acoustic", False)
        if likes_acoustic and song.get("acousticness", 0.0) > 0.5:
            reasons.append("is acoustic")
        elif not likes_acoustic and song.get("acousticness", 0.0) < 0.5:
            reasons.append("is not acoustic")

        if _lookup_user_preference(user_prefs, "preferred_energy_level") and song.get("energy_level") == _lookup_user_preference(user_prefs, "preferred_energy_level"):
            reasons.append("matches your energy level category")
        if _lookup_user_preference(user_prefs, "preferred_danceability_tier") and song.get("danceability_tier") == _lookup_user_preference(user_prefs, "preferred_danceability_tier"):
            reasons.append("matches your danceability tier")
        if _lookup_user_preference(user_prefs, "preferred_tempo_category") and song.get("tempo_category") == _lookup_user_preference(user_prefs, "preferred_tempo_category"):
            reasons.append("matches your tempo category")

        if reasons:
            return f"This song {', '.join(reasons)}."
        else:
            return "This song was recommended based on overall similarity."

    selected_songs = _select_top_k_with_diversity(songs, user_prefs, strategy, k)
    return [(song, score, explain(song)) for song, score in selected_songs]

