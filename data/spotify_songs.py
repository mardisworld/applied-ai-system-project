import pandas as pd

INPUT_FILE  = "dataset.csv"
OUTPUT_FILE = "songs_dataset_full.csv"

# ── Mood derivation (simple) ──────────────────────────
MOOD_MAP = {
    "high_val_high_e": "happy",
    "high_val_low_e":  "relaxed",
    "low_val_high_e":  "intense",
    "low_val_low_e":   "melancholic",
    "mid_val_high_e":  "energetic",
    "mid_val_low_e":   "chill",
    "high_val_mid_e":  "uplifting",
    "low_val_mid_e":   "moody",
}

def get_mood(valence, energy):
    v = "high" if valence >= 0.65 else ("low" if valence < 0.40 else "mid")
    e = "high" if energy  >= 0.65 else ("low" if energy  < 0.40 else "mid")
    return MOOD_MAP.get(f"{v}_val_{e}_e", "chill")

# ── Detailed mood using valence + energy + danceability + acousticness ────
def get_detailed_mood(valence, energy, danceability, acousticness):
    if valence >= 0.75 and energy >= 0.75 and danceability >= 0.70:
        return "euphoric"
    elif valence >= 0.65 and energy >= 0.65 and danceability >= 0.60:
        return "upbeat"
    elif valence >= 0.65 and acousticness >= 0.60:
        return "nostalgic"
    elif valence >= 0.65 and energy < 0.40:
        return "peaceful"
    elif valence >= 0.55 and energy >= 0.50:
        return "happy"
    elif valence < 0.35 and energy >= 0.75:
        return "aggressive"
    elif valence < 0.35 and energy >= 0.50:
        return "angry"
    elif valence < 0.35 and acousticness >= 0.60:
        return "somber"
    elif valence < 0.35 and energy < 0.40:
        return "melancholic"
    elif valence < 0.50 and energy >= 0.60 and danceability >= 0.65:
        return "intense"
    elif valence >= 0.50 and energy < 0.40 and acousticness >= 0.50:
        return "dreamy"
    elif valence >= 0.45 and energy < 0.45:
        return "chill"
    elif valence >= 0.45 and energy >= 0.45:
        return "groovy"
    else:
        return "moody"

# ── Energy level ──────────────────────────────────────
def get_energy_level(energy):
    if energy < 0.25:
        return "Low"
    elif energy < 0.50:
        return "Medium"
    elif energy < 0.75:
        return "High"
    else:
        return "Intense"

# ── Danceability tier ─────────────────────────────────
def get_danceability_tier(danceability):
    if danceability < 0.25:
        return "Non-danceable"
    elif danceability < 0.50:
        return "Moderate"
    elif danceability < 0.75:
        return "Danceable"
    else:
        return "Club-ready"

# ── Tempo category ────────────────────────────────────
def get_tempo_category(tempo):
    if tempo < 70:
        return "Slow"
    elif tempo < 111:
        return "Mid"
    elif tempo < 141:
        return "Upbeat"
    else:
        return "Fast"

# ── Load ──────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"  Loaded {len(df):,} rows.")

# ── Clean ─────────────────────────────────────────────
required = ["track_name", "artists", "track_genre", "popularity",
            "energy", "tempo", "valence", "danceability", "acousticness"]
df = df.dropna(subset=required)
df = df.drop_duplicates(subset=["track_name", "artists"])
df = df.reset_index(drop=True)

# ── Build output dataframe ────────────────────────────
out = pd.DataFrame()

# Original columns (in original order)
out["id"]           = df.index + 1
out["title"]        = df["track_name"].str.replace('"', "'")
out["artist"]       = df["artists"].str.replace('"', "'")
out["genre"]        = df["track_genre"]
out["mood"]         = df.apply(lambda r: get_mood(r["valence"], r["energy"]), axis=1)
out["energy"]       = df["energy"].round(2)
out["tempo_bpm"]    = df["tempo"].astype(int)
out["valence"]      = df["valence"].round(2)
out["danceability"] = df["danceability"].round(2)
out["acousticness"] = df["acousticness"].round(2)

# New columns (appended after originals)
out["popularity"]        = df["popularity"].astype(int)
out["detailed_mood"]     = df.apply(lambda r: get_detailed_mood(r["valence"], r["energy"], r["danceability"], r["acousticness"]), axis=1)
out["energy_level"]      = df["energy"].apply(get_energy_level)
out["danceability_tier"] = df["danceability"].apply(get_danceability_tier)
out["tempo_category"]    = df["tempo"].apply(get_tempo_category)

# ── Save ──────────────────────────────────────────────
out.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Done! {len(out):,} songs saved to '{OUTPUT_FILE}'")
print(f"   Genres represented:        {out['genre'].nunique()}")
print(f"   Moods represented:         {out['mood'].nunique()}")
print(f"   Detailed moods represented:{out['detailed_mood'].nunique()}")