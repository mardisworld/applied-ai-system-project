# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

---

## How The System Works

## Real World Recommendation Systems
Collaborative filtering and content-based filtering represent two fundamentally different philosophies for predicting what someone will enjoy. Collaborative filtering is entirely social in nature — it ignores the content itself and focuses purely on patterns of human behavior. 

The system looks at what you've listened to, skipped, saved, or replayed, then finds other users whose behavior closely mirrors yours. From there, it recommends things those "taste twins" loved that you haven't encountered yet. The underlying assumption is that if two people agreed on hundreds of songs in the past, they'll probably agree on the next one too. This makes it remarkably good at serendipitous, cross-genre discovery — but it struggles with new users who have no history, and new songs that have no listeners yet.

Content-based filtering sidesteps that problem entirely by analyzing the intrinsic attributes of the content itself rather than relying on other people's opinions. For music, this means breaking a song down into measurable features — tempo, key, loudness, energy, danceability, emotional valence, and even the instruments present — and building a kind of sonic fingerprint for it. 

The system then recommends songs whose fingerprints closely match those of tracks you've already enjoyed.
Because it reasons from the music's own attributes, it works even for obscure or brand-new tracks with no listening data behind them. The tradeoff, however, is that it can trap you in a narrow stylistic bubble, surfacing music that sounds similar to what you know without ever pushing you toward something genuinely surprising. 

In practice, the most sophisticated recommendation systems layer both approaches together — using collaborative filtering for broad discovery and content-based filtering to fine-tune the match — supplemented by natural language processing, engagement signals, and deep learning to capture nuances that neither method alone can fully address.

## System Design Explanation: Music Recommender Simulation
This is a weighted-score music recommendation system designed for educational purposes to understand how AI recommenders match user preferences to song attributes.

## The Core Idea
The system takes a user's music taste profile and finds songs from a catalog that match it best. Instead of guessing randomly, it assigns a numerical score to each song based on how well it aligns with what the user likes. Higher-scoring songs are recommended first.

Some prompts to answer:

- What features does each `Song` use in your system?
The features used in each 'Song' are genre and mood (categorical), energy,  valence, danceability, and acousticness(numeric 0-1 scale), and tempo_bpm (beats per minute). It also incorporates "derived categories (from Spotify data processing) such as detailed_mood, energy_level, danceability_tier, tempo_category, and popularity.

- What information does your `UserProfile` store? The `UserProfile` stores three categories of attributes for each user. First, there are categorical preferences (exact matches). These include favorite_genre,  favorite_mood, target_energy, favorite_detailed_mood, and likes_acoustic (as a Boolean true/false value). 

Next, there are numerical targets which are similarity based (on the user's preferences). These include target_tempo (preferred bpm), target_valence, target_danceability(numerical 0-1), and target_popularity(numerical 0-100), preferred_energy_level, preferred_danceability_tier, and preferred_tempo_category. 
 
Finally, we have weights "importance knobs", wherein each attributes gets a weight (default: genre=10, mood=5, others = 0 - 3). Increasing a weight makes that attribute matter more in recommendations. A weight of 0 means "ignore this attribute". The weights, or importance knobs, include weight_genre, weight_mood, weight_energy, weight_tempo, weight_valence, weight_danceability, weight_acousticness, weight_detailed_mood, weight_energy_level,  weight_danceability_tier, weight_tempo_category and
weight_popularity.

- How does your `Recommender` compute a score for each song?

The `Recommender` computes a score for each song by adding up points from multiple attributes, weighted by user preferences. Here's how it works:

**Scoring Formula Overview:**
- **Total Score = Sum of all weighted matches and similarities**

**Breakdown by Attribute Type:**

- **Categorical Matches (Exact Matches):**
  - If song.genre == user.favorite_genre → add `weight_genre` (default: 10.0)
  - If song.mood == user.favorite_mood → add `weight_mood` (default: 5.0)
  - If song.detailed_mood == user.favorite_detailed_mood → add `weight_detailed_mood` (default: 0.0)
  - If song.energy_level == user.preferred_energy_level → add `weight_energy_level` (default: 0.0)
  - If song.danceability_tier == user.preferred_danceability_tier → add `weight_danceability_tier` (default: 0.0)
  - If song.tempo_category == user.preferred_tempo_category → add `weight_tempo_category` (default: 0.0)

- **Numerical Similarities (Fuzzy Matches):**
  - Energy: `weight_energy * (1 - |song.energy - user.target_energy|)` (default weight: 5.0)
  - Tempo: `weight_tempo * (1 - min(|song.tempo_bpm - user.target_tempo| / 140, 1))` (normalized for BPM range; default weight: 0.0)
  - Valence: `weight_valence * (1 - |song.valence - user.target_valence|)` (default weight: 0.0)
  - Danceability: `weight_danceability * (1 - |song.danceability - user.target_danceability|)` (default weight: 0.0)
  - Popularity: `weight_popularity * (1 - min(|song.popularity - user.target_popularity| / 100, 1))` (normalized 0-100; default weight: 0.0)

- **Acousticness (Special Case):**
  - If user.likes_acoustic: `weight_acousticness * song.acousticness`
  - Else: `weight_acousticness * (1 - song.acousticness)` (default weight: 5.0)

**Example Calculation:**
For a user who loves "pop" genre (weight_genre=10), "happy" mood (weight_mood=5), high energy (target_energy=0.8, weight_energy=5), and dislikes acoustic (likes_acoustic=False, weight_acousticness=5):

- Pop song with energy=0.75, acousticness=0.1: Score = 10 (genre) + 5 (mood) + 5*(1-0.05)=4.75 (energy) + 5*(1-0.1)=4.5 (acoustic) = **24.25**

**Diagram (Simple Flow):**
```
User Profile (preferences + weights)
    ↓
Song Attributes (genre, mood, energy, etc.)
    ↓
Compute Matches & Similarities
    ↓
Weighted Sum → Total Score
    ↓
Sort Songs by Score (descending) → Top k Recommendations
```

- How do you choose which songs to recommend?
After scoring all songs in the catalog, the system sorts them by total score in descending order (highest scores first). It then returns the top k songs (default: 5) as recommendations, along with their scores and human-readable explanations of why they matched the user's profile.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on mood from 10.0 to 1.0 with favorite_mood='chill' and favorite_genre='emo'?
When favorite_mood was set to 'chill' and weight_mood=10.0 (higher than weight_genre=5.0), the top songs were all chill emo tracks (e.g., "Mover Awayer" at 44.56).

When weight_mood was reduced to 1.0, the top songs switched to happy emo tracks (e.g., "This Is Why" at 39.21), showing how mood weight affects ranking when the mood preference varies among songs.

- What happened when you added tempo or valence to the score?
When tempo and valence weights were set to 0 (effectively removing them from scoring), song scores dropped significantly (from ~39 to ~27), and recommendations shifted to different tracks like "Du riechst so gut" and "My Paradise" that matched other attributes but not necessarily tempo/valence.

With tempo weight at 10.0 and valence at 2.0, scores increased by 10-12 points for songs with similar tempo (target: 120 BPM) and valence (target: 0.75), prioritizing upbeat, positive songs. This demonstrates how numerical similarity attributes can dramatically alter rankings when weighted heavily, allowing fine-tuning of recommendations beyond categorical matches.

- How did your system behave for different types of users?
For users preferring chill, low-energy, acoustic music (e.g., favorite_mood='chill', target_energy=0.5, likes_acoustic=True): The system recommended diverse acoustic songs from various cultures and languages (e.g., Japanese "ただ声一つ", Portuguese "Exu"), prioritizing mood and acoustic matches over genre.

For users liking happy, high-energy, non-acoustic emo (e.g., favorite_genre='emo', favorite_mood='happy', target_energy=0.8, likes_acoustic=False): It recommended upbeat emo tracks (e.g., "This Is Why", "Carry Me Away"), matching genre and mood with high scores due to multiple attribute alignments.

The system adapts recommendations based on weighted preferences, favoring songs that match categorical preferences (genre, mood) and numerical similarities (energy, acousticness), leading to personalized results for different taste profiles.

---

## Limitations and Risks

Summarize some limitations of your recommender.

- **Limited to metadata-only analysis**: The system only considers numeric features (tempo, energy, valence) and categorical tags (genre, mood) extracted from Spotify data. It cannot understand lyrics, instrumentation, vocal characteristics, or artistic intent, missing nuanced distinctions between songs that sound similar but feel different.
- **Dataset bias toward mainstream/English music**: With over 100 genres but with study and black-metal dominating the dataset, and most songs in English, the recommender is biased toward Spotify's algorithmic amplification of popular tracks. Niche, non-Western, and emerging artists are severely underrepresented, perpetuating mainstream music dominance.
- **Content-based filter bubble risk**: The purely content-based approach (no collaborative filtering) means the system recommends songs similar to user preferences without introducing serendipity. Users preferring "emo" will stay in emo, never discovering how their taste might align with adjacent genres.
- **No diversity mechanism**: The recommender greedily picks the top-5 highest-scoring songs rather than balancing exploration and exploitation. A user might receive 5 very similar songs instead of a diverse set that covers different moods or tempos within their preferences.
- **Weight configuration burden**: Users must manually tune 12+ weights to match their taste. Default weights (e.g., weight_genre=0.5, weight_tempo=10.0) may not suit all users, and poorly calibrated weights can lead to irrelevant recommendations or overwhelming a user's actual preferences.
- **No temporal or contextual awareness**: The system cannot adapt recommendations based on time of day, user mood, or listening context (e.g., workout vs. studying). A user who loves study genre music might not want study recommendations at 11 PM.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)


Write 1 to 2 paragraphs here about what you learned:

- How do recommenders turn data into predictions?

Recommender systems turn data into predictions by translating user preferences and item attributes into a common language—numbers. For content-based recommenders like this project, each song is represented by a set of features (genre, mood, energy, tempo, etc.), and each user has a profile describing their ideal values for those features. 

The system computes a score for every song by measuring how closely its features match the user's preferences, often using weighted sums or similarity functions. The highest-scoring items are predicted to be the best matches and are recommended first. In collaborative filtering, predictions are made by finding users with similar tastes and recommending items those users liked, even if the features are unknown. In both cases, the system uses patterns in the data—either content or behavior—to estimate what the user will enjoy next.

- Where can bias or unfairness show up in systems like this?

Bias and unfairness can enter recommender systems at many stages. If the dataset is skewed—like this one, which overrepresents mainstream and English-language music—then recommendations will favor those genres and artists, making it harder for niche or non-Western music to be discovered. 
The choice of features also matters: if important aspects like lyrics, cultural context, or artist identity are missing, the system can't recommend based on those dimensions, which can disadvantage certain groups or styles. 
Content-based recommenders can trap users in "filter bubbles," repeatedly surfacing similar songs and limiting exposure to new or diverse music. Collaborative filtering can reinforce popularity bias, amplifying what is already popular and ignoring minority tastes. Finally, default weights and system design choices can unintentionally privilege some users' preferences over others, making fairness an ongoing challenge that requires careful attention.

---

