# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

I am designing a music recommender that uses 

---

## How The System Works

System Design Explanation: Music Recommender Simulation
This is a weighted-score music recommendation system designed for educational purposes to understand how AI recommenders match user preferences to song attributes.

The Core Idea
The system takes a user's music taste profile and finds songs from a catalog that match it best. Instead of guessing randomly, it assigns a numerical score to each song based on how well it aligns with what the user likes. Higher-scoring songs are recommended first.

Some prompts to answer:

- What features does each `Song` use in your system?
  - The features used in each 'Song' are genre and mood (categorical), energy,  valence, danceability, and acousticness(numeric 0-1 scale), and tempo_bpm (beats per minute). It also incorporates "derived categories (from Spotify data processing) such as detailed_mood, energy_level, danceability_tier, tempo_category, and popularity.

- What information does your `UserProfile` store? The `UserProfile` stores three categories of attributes for each user. First, there are categorical preferences (exact matches). These include favorite_genre,  favorite_mood, target_energy, favorite_detailed_mood, and likes_acoustic (as a Boolean true/false value). 

Next, there are numerical targets which are similarity based (on the user's preferences). These include target_tempo (preferred bpm), target_valence, target_danceability(numerical 0-1), and target_popularity(numerical 0-100), preferred_energy_level, preferred_danceability_tier, and preferred_tempo_category. 
 
 
Finally, we have weights "importance knobs", wherein each attributes gets a weight (default: genre=10, mood=5, others = 0 - 3). Increasing a weight makes that attribute matter more in recommendations. A weight of 0 means "ignore this attribute". The weights, or importance knobs, include weight_genre, weight_mood, weight_energy, weight_tempo, weight_valence, weight_danceability, weight_acousticness, weight_detailed_mood, weight_energy_level,  weight_danceability_tier, weight_tempo_category and
weight_popularity.

- How does your `Recommender` compute a score for each song

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

- How do you choose which songs to recommend

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

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions

- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

