from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score_song(self, song: Song, user: UserProfile) -> float:
        """
        Scoring Rule: Calculates a weighted score for a song based on user preferences.
        - Genre match: weight_genre if match
        - Mood match: weight_mood if match
        - Detailed mood match: weight_detailed_mood if match
        - Energy similarity: weight_energy * (1 - |energy_diff|)
        - Tempo similarity: weight_tempo * (1 - min(|tempo_diff| / 140, 1))
        - Valence similarity: weight_valence * (1 - |valence_diff|)
        - Danceability similarity: weight_danceability * (1 - |danceability_diff|)
        - Acoustic preference: weight_acousticness * acousticness if likes_acoustic, else weight_acousticness * (1 - acousticness)
        - Energy level match: weight_energy_level if match
        - Danceability tier match: weight_danceability_tier if match
        - Tempo category match: weight_tempo_category if match
        - Popularity similarity: weight_popularity * (1 - normalized_popularity_diff)
        """
        score = 0.0
        if song.genre == user.favorite_genre:
            score += user.weight_genre
        if song.mood == user.favorite_mood:
            score += user.weight_mood
        if user.favorite_detailed_mood and song.detailed_mood == user.favorite_detailed_mood:
            score += user.weight_detailed_mood
        
        energy_diff = abs(song.energy - user.target_energy)
        score += user.weight_energy * (1 - energy_diff)
        
        tempo_diff = abs(song.tempo_bpm - user.target_tempo)
        score += user.weight_tempo * (1 - min(tempo_diff / 140, 1))
        
        valence_diff = abs(song.valence - user.target_valence)
        score += user.weight_valence * (1 - valence_diff)
        
        danceability_diff = abs(song.danceability - user.target_danceability)
        score += user.weight_danceability * (1 - danceability_diff)
        
        if user.likes_acoustic:
            acoustic_score = song.acousticness
        else:
            acoustic_score = 1 - song.acousticness
        score += user.weight_acousticness * acoustic_score
        
        if user.preferred_energy_level and song.energy_level == user.preferred_energy_level:
            score += user.weight_energy_level
        if user.preferred_danceability_tier and song.danceability_tier == user.preferred_danceability_tier:
            score += user.weight_danceability_tier
        if user.preferred_tempo_category and song.tempo_category == user.preferred_tempo_category:
            score += user.weight_tempo_category
        
        if song.popularity is not None and user.weight_popularity > 0:
            popularity_diff = abs(song.popularity - user.target_popularity) / 100.0
            score += user.weight_popularity * (1 - min(popularity_diff, 1))
        
        return score

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Ranking Rule: Scores all songs, sorts by score descending, returns top k.
        """
        scored_songs = [(song, self._score_song(song, user)) for song in self.songs]
        scored_songs.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored_songs[:k]]

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
    Required by src/main.py
    """
    import csv

    def to_float(value: Optional[str]) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

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


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    Uses the same weighted Scoring Rule and Ranking Rule as the OOP version.
    """
    def lookup(key: str, fallback: Optional[str] = None) -> Optional[str]:
        return user_prefs.get(key) or user_prefs.get(fallback)

    def score_song(song: Dict) -> float:
        score = 0.0
        if song.get("genre") == lookup("favorite_genre", "genre"):
            score += float(user_prefs.get("weight_genre", 10.0))
        if song.get("mood") == lookup("favorite_mood", "mood"):
            score += float(user_prefs.get("weight_mood", 5.0))
        if lookup("favorite_detailed_mood") and song.get("detailed_mood") == lookup("favorite_detailed_mood"):
            score += float(user_prefs.get("weight_detailed_mood", 0.0))
        
        energy_target = float(user_prefs.get("target_energy", user_prefs.get("energy", 0.0)))
        energy_diff = abs(song.get("energy", 0.0) - energy_target)
        score += float(user_prefs.get("weight_energy", 5.0)) * (1 - energy_diff)
        
        tempo_target = float(user_prefs.get("target_tempo", 0.0))
        tempo_diff = abs(song.get("tempo_bpm", 0.0) - tempo_target)
        score += float(user_prefs.get("weight_tempo", 0.0)) * (1 - min(tempo_diff / 140, 1))
        
        valence_diff = abs(song.get("valence", 0.0) - float(user_prefs.get("target_valence", 0.0)))
        score += float(user_prefs.get("weight_valence", 0.0)) * (1 - valence_diff)
        
        danceability_diff = abs(song.get("danceability", 0.0) - float(user_prefs.get("target_danceability", 0.0)))
        score += float(user_prefs.get("weight_danceability", 0.0)) * (1 - danceability_diff)
        
        likes_acoustic = user_prefs.get("likes_acoustic", False)
        if likes_acoustic:
            acoustic_score = song.get("acousticness", 0.0)
        else:
            acoustic_score = 1 - song.get("acousticness", 0.0)
        score += float(user_prefs.get("weight_acousticness", 5.0)) * acoustic_score
        
        if lookup("preferred_energy_level") and song.get("energy_level") == lookup("preferred_energy_level"):
            score += float(user_prefs.get("weight_energy_level", 0.0))
        if lookup("preferred_danceability_tier") and song.get("danceability_tier") == lookup("preferred_danceability_tier"):
            score += float(user_prefs.get("weight_danceability_tier", 0.0))
        if lookup("preferred_tempo_category") and song.get("tempo_category") == lookup("preferred_tempo_category"):
            score += float(user_prefs.get("weight_tempo_category", 0.0))
        
        popularity = song.get("popularity")
        if popularity is not None and float(user_prefs.get("weight_popularity", 0.0)) > 0:
            popularity_target = float(user_prefs.get("target_popularity", 0.0))
            popularity_diff = abs(popularity - popularity_target) / 100.0
            score += float(user_prefs.get("weight_popularity", 0.0)) * (1 - min(popularity_diff, 1))
        
        return score

    def explain(song: Dict) -> str:
        reasons = []
        if song.get("genre") == lookup("favorite_genre", "genre"):
            reasons.append("matches your favorite genre")
        if song.get("mood") == lookup("favorite_mood", "mood"):
            reasons.append("matches your favorite mood")
        if lookup("favorite_detailed_mood") and song.get("detailed_mood") == lookup("favorite_detailed_mood"):
            reasons.append("matches your detailed mood")
        
        energy_target = float(user_prefs.get("target_energy", user_prefs.get("energy", 0.0)))
        energy_diff = abs(song.get("energy", 0.0) - energy_target)
        if energy_diff < 0.2:
            reasons.append("has similar energy level")
        
        tempo_diff = abs(song.get("tempo_bpm", 0.0) - float(user_prefs.get("target_tempo", 0.0)))
        if tempo_diff < 20:
            reasons.append("has similar tempo")
        
        valence_diff = abs(song.get("valence", 0.0) - float(user_prefs.get("target_valence", 0.0)))
        if valence_diff < 0.2:
            reasons.append("has similar valence")
        
        danceability_diff = abs(song.get("danceability", 0.0) - float(user_prefs.get("target_danceability", 0.0)))
        if danceability_diff < 0.2:
            reasons.append("has similar danceability")
        
        likes_acoustic = user_prefs.get("likes_acoustic", False)
        if likes_acoustic and song.get("acousticness", 0.0) > 0.5:
            reasons.append("is acoustic")
        elif not likes_acoustic and song.get("acousticness", 0.0) < 0.5:
            reasons.append("is not acoustic")
        
        if lookup("preferred_energy_level") and song.get("energy_level") == lookup("preferred_energy_level"):
            reasons.append("matches your energy level category")
        if lookup("preferred_danceability_tier") and song.get("danceability_tier") == lookup("preferred_danceability_tier"):
            reasons.append("matches your danceability tier")
        if lookup("preferred_tempo_category") and song.get("tempo_category") == lookup("preferred_tempo_category"):
            reasons.append("matches your tempo category")
        
        if reasons:
            return f"This song {', '.join(reasons)}."
        else:
            return "This song was recommended based on overall similarity."

    scored = [(song, score_song(song), explain(song)) for song in songs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


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
            songs.append({
                "id": to_int(row.get("id")) or 0,
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "genre": row.get("genre", ""),
                "mood": row.get("mood", ""),
                "energy": to_float(row.get("energy")),
                "tempo_bpm": to_float(row.get("tempo_bpm") or row.get("tempo")),
                "valence": to_float(row.get("valence")),
                "danceability": to_float(row.get("danceability")),
                "acousticness": to_float(row.get("acousticness")),
                "popularity": to_int(row.get("popularity")),
                "detailed_mood": row.get("detailed_mood"),
                "energy_level": row.get("energy_level"),
                "danceability_tier": row.get("danceability_tier"),
                "tempo_category": row.get("tempo_category"),
            })
    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    Uses the same weighted Scoring Rule and Ranking Rule as the OOP version.
    """
    def score_song(song: Dict) -> float:
        score = 0.0
        if song['genre'] == user_prefs.get('favorite_genre'):
            score += user_prefs.get('weight_genre', 10.0)
        if song['mood'] == user_prefs.get('favorite_mood'):
            score += user_prefs.get('weight_mood', 5.0)
        
        energy_diff = abs(song['energy'] - user_prefs.get('target_energy', 0.0))
        score += user_prefs.get('weight_energy', 5.0) * (1 - energy_diff)
        
        tempo_diff = abs(song['tempo_bpm'] - user_prefs.get('target_tempo', 0.0))
        score += user_prefs.get('weight_tempo', 0.0) * (1 - min(tempo_diff / 140, 1))
        
        valence_diff = abs(song['valence'] - user_prefs.get('target_valence', 0.0))
        score += user_prefs.get('weight_valence', 0.0) * (1 - valence_diff)
        
        danceability_diff = abs(song['danceability'] - user_prefs.get('target_danceability', 0.0))
        score += user_prefs.get('weight_danceability', 0.0) * (1 - danceability_diff)
        
        likes_acoustic = user_prefs.get('likes_acoustic', False)
        if likes_acoustic:
            acoustic_score = song['acousticness']
        else:
            acoustic_score = 1 - song['acousticness']
        score += user_prefs.get('weight_acousticness', 5.0) * acoustic_score
        
        return score

    def explain(song: Dict) -> str:
        reasons = []
        if song['genre'] == user_prefs.get('favorite_genre'):
            reasons.append("matches your favorite genre")
        if song['mood'] == user_prefs.get('favorite_mood'):
            reasons.append("matches your favorite mood")
        
        energy_diff = abs(song['energy'] - user_prefs.get('target_energy', 0.0))
        if energy_diff < 0.2:
            reasons.append("has similar energy level")
        
        tempo_diff = abs(song['tempo_bpm'] - user_prefs.get('target_tempo', 0.0))
        if tempo_diff < 20:
            reasons.append("has similar tempo")
        
        valence_diff = abs(song['valence'] - user_prefs.get('target_valence', 0.0))
        if valence_diff < 0.2:
            reasons.append("has similar valence")
        
        danceability_diff = abs(song['danceability'] - user_prefs.get('target_danceability', 0.0))
        if danceability_diff < 0.2:
            reasons.append("has similar danceability")
        
        likes_acoustic = user_prefs.get('likes_acoustic', False)
        if likes_acoustic and song['acousticness'] > 0.5:
            reasons.append("is acoustic")
        elif not likes_acoustic and song['acousticness'] < 0.5:
            reasons.append("is not acoustic")
        
        if reasons:
            return f"This song {', '.join(reasons)}."
        else:
            return "This song was recommended based on overall similarity."

    scored = [(song, score_song(song), explain(song)) for song in songs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
