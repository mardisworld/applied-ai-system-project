# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **Musixdex 1.0**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

- What kind of recommendations does it generate
  - The system generates a ranked list of 5 song recommendations from a catalog of ~84,000 tracks, each with a numerical score (0-50 range) and human-readable explanations of why it matches the user's preferences. Recommendations are based on weighted matching of song attributes like genre, mood, energy, tempo, valence, danceability, acousticness, and popularity.

- What assumptions does it make about the user
  - Assumes users have well-defined music preferences that can be expressed as categorical choices (e.g., favorite genre, mood) and numerical targets (e.g., preferred energy level, tempo). It assumes users are willing to tune 12+ weights to customize how much each attribute matters, and that they want content-based recommendations rather than collaborative or social suggestions.

- Is this for real users or classroom exploration
  - This is strictly for classroom exploration and educational purposes, not for real users. It's designed to teach concepts in recommender systems, bias, and AI ethics through hands-on experimentation with a simplified music recommendation model.  

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.). I answered this in the README.md. The features used in each 'Song' are genre and mood (categorical), energy,  valence, danceability, and acousticness(numeric 0-1 scale), and tempo_bpm (beats per minute). It also incorporates "derived categories (from Spotify data processing) such as detailed_mood, energy_level, danceability_tier, tempo_category, and popularity.  
- What user preferences are considered?
  - **Categorical preferences**: Favorite genre (e.g., 'emo'), favorite mood (e.g., 'happy'), favorite detailed mood (e.g., 'upbeat'), preferred energy level (e.g., 'High'), preferred danceability tier (e.g., 'Danceable'), preferred tempo category (e.g., 'Upbeat'), and likes acoustic (true/false).
  - **Numerical targets**: Target energy (0-1 scale), target tempo (BPM), target valence (emotional positivity, 0-1), target danceability (0-1), and target popularity (0-100).
  - **Importance weights**: 12 weights controlling how much each attribute matters (e.g., weight_genre=0.5, weight_mood=5.0, weight_tempo=10.0), allowing users to prioritize certain preferences over others.  
- How does the model turn those into a score?
  - The model adds up points from different types of matches between your preferences and each song's features. For exact matches like genre or mood, it adds the full weight if they match (e.g., if you like 'emo' and the song is emo, add 5 points). For numerical similarities like energy or tempo, it gives partial points based on how close they are (e.g., if your target tempo is 120 BPM and the song is 115 BPM, it adds most of the tempo weight). Acousticness works differently—if you like acoustic, it adds points based on how acoustic the song is; if you don't, it adds points based on how non-acoustic it is. The total score is the sum of all these weighted matches, and higher scores mean better matches.  


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

---

## 4. Data  

Describe the dataset the model uses.  

- How many songs are in the catalog  - How many songs are in `data/songs.csv`? 10
- Did you add or remove any songs? Yes, I went and found a dataset from Kaggle with Spotify data that included 84,000 real songs, then had Claude write me a script (spotify_songs.py) to convert that dataset into a large csv file (84,000 real songs) of real music (songs_dataset_full.csv) so I could get more representative data of actual music. 
- What kinds of genres or moods are represented?
  - Genres include a wide variety: study (most common), black-metal, comedy, heavy-metal, bluegrass, forro, grindcore, malay, idm, iranian, and many others including pop (299 songs). The dataset has over 100 unique genres.
  - Moods include: intense (most common), happy, energetic, melancholic, chill, moody, uplifting, and relaxed. These are derived from Spotify's valence and energy features.
- Whose taste does this data mostly reflect?
  - The data reflects the collective listening habits of Spotify users worldwide. Since it's derived from Spotify's track database, it represents a global audience but is biased toward popular, mainstream music (e.g., pop, hip-hop, electronic) and English-language songs. It may underrepresent niche, regional, or non-Western genres, and the popularity scores favor tracks that have been streamed more, often reflecting algorithmic amplification on the platform.
- Are there parts of musical taste missing in the dataset?
  - **Yes, significant dimensions are missing.** The dataset captures only surface-level audio features (energy, tempo, valence) and broad categorical labels (genre, mood), but misses:
    - **Lyrical content**: No lyrics analysis, themes, storytelling depth, or poetic quality—two "happy" songs could say completely different things.
    - **Instrumentation & timbre**: No information about specific instruments (strings vs. synth), vocal style (male/female/effects), or acoustic vs. electric character.
    - **Artist identity**: No artist nationality, gender, career stage, or cultural origin—missing the social and identity dimensions of musicians.
    - **Production & era**: No information about release date, decade, production style, or sound quality—missing nostalgia and historical context.
    - **Structural complexity**: No song structure (verses, bridges, key changes) or harmonic depth—missing how musically "interesting" a song is.
    - **Subgenre precision**: Broad genre labels (e.g., "emo") mask distinct subgenres and communities (e.g., emo-pop vs. screamo vs. deathcore).
    - **Personal resonance**: No account for songs with emotional significance to individual users—a generic "pop" song might matter because of a memory attached to it.
  - These missing dimensions mean the recommender captures only the "sonic fingerprint" but not the "human meaning" of music.  

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- For which user types does it give reasonable results?
  - **Genre-specific users with clear mood preferences**: The system excels for users who know exactly what they want (e.g., "I like emo music that's upbeat and energetic"). When favorite_genre='emo' and favorite_mood='happy', the top recommendations matched both attributes with high scores, creating coherent playlists.
  - **Users preferring niche or acoustic-driven tastes**: Chill, acoustic-focused users (e.g., favorite_mood='chill', likes_acoustic=True) received unexpectedly diverse, high-quality recommendations spanning multiple cultures and languages (Japanese, Portuguese, Indian), because low genre weight allowed mood/acoustic to override genre bias.
  - **Users with consistent multi-attribute alignment**: When multiple attributes align (genre match + mood match + energy match + acoustic preference), the recommender delivers highly relevant results due to cumulative scoring. "This Is Why" scored 39.71 against a happy/energetic emo profile because it matched 4+ attributes.
  - **Users with strong, focused preferences**: The system works best for users who weight a few key attributes heavily (e.g., mood + acousticness) rather than trying to balance 12+ weights equally. This reduces noise and makes recommendations decisive.

- Any patterns you think your scoring captures correctly?
  - **Mood as an emotional anchor**: Mood weight strongly influences results when songs in the dataset have varying moods. The shift from chill emo (44.56) to happy emo (39.21) when changing mood weight from 10.0 to 1.0 shows the system correctly prioritizes emotional tone.
  - **Acoustic vs. electric trade-off**: The likes_acoustic boolean correctly filters recommendations. Chill users who like acoustic get intimate, minimal production; non-acoustic users get polished, produced tracks.
  - **Tempo as a discoverer of sub-styles**: With weight_tempo=10.0, the system found upbeat emo tracks (120 BPM target) over slower alternatives, showing tempo is an effective proxy for sub-genre differentiation.

- Cases where the recommendations matched your intuition
  - Chill/acoustic profile → diverse international folk/world music (expected for low genre weight)
  - Happy/energetic emo → upbeat emo pop (expected for high genre+mood weights)
  - Removing tempo/valence → broader but less nuanced results (expected, showed these weights refine rankings)  

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Which features does it not consider?
  - **Lyrical & semantic content**: No analysis of lyrics, themes, storytelling, or message. Two "happy" songs may contradict each other (celebration vs. ironic happiness).
  - **Instrumentation & timbre**: No information about specific instruments, vocal characteristics, production techniques, or sonic texture beyond the broad energy/valence labels.
  - **Artist metadata**: No artist nationality, gender, career stage, or cultural origin—missing identity and social dimensions of musical taste.
  - **Temporal & contextual factors**: No time of day, season, listening context (workout vs. studying), or user mood awareness. Study genre might be unwanted at 11 PM.
  - **Recency & novelty**: No preference for new releases vs. classics, or discovery-versus-comfort balance. System recommends similarly regardless of release date.
  - **User history & behavior**: No past listening patterns, skip rates, replay counts, or explicit feedback. Cold-start problem: new users get generic recommendations.
  - **Diversity & serendipity**: Greedy ranking (top-5 highest scores) means all 5 recommendations may be nearly identical. No built-in exploration vs. exploitation trade-off.
  - **Collaborative signals**: Pure content-based filtering, no "users like you also enjoyed..." signals. Misses cross-genre discovery through shared audiences.
  - **Subgenre nuance**: Broad genre labels (e.g., "emo") mask important distinctions (emo-pop vs. screamo vs. post-hardcore) that listeners care deeply about.

- Which genres or moods were underrepresented?
  - The dataset is heavily biased toward **study (996 songs) and black-metal (991 songs)**, while **pop (299 songs)** is underrepresented relative to real-world streaming.
  - **Non-English music** is significantly underrepresented; English-language pop, hip-hop, and rock dominate due to Spotify's user base skew.
  - **Regional and traditional music** (e.g., classical Indian, African, Middle Eastern) are sparse, limiting recommendations for users interested in those cultures.
  - **Moods derived from valence/energy** oversimplify emotion; nuanced moods like "nostalgic," "introspective," or "party" are collapsed into broad categories (intense, happy, chill).

- Cases where the system overfits to one preference
  - **Genre dominance**: When weight_genre is high (e.g., 10.0), the system almost exclusively recommends songs from that genre, ignoring potentially better matches in adjacent genres. An emo fan looking for upbeat energy might miss excellent indie-pop songs.
  - **Mood lock-in**: High weight_mood combined with a large dataset of songs with that mood (e.g., intense: 15,794 songs) means recommendations rarely escape the mood. A user who sometimes wants intense music will get *only* intense songs.
  - **Acoustic trap**: If a user sets likes_acoustic=True and weight_acousticness=5.0, the system nearly always recommends acoustic songs, even if a produced alternative might be more energetic or engaging.
  - **Dataset popularity bias**: High-streamed songs (higher popularity scores) naturally score higher when weight_popularity > 0, amplifying Spotify's algorithmic echo chamber and suppressing niche artists.

- Ways the scoring might unintentionally favor some users
  - **Mainstream user advantage**: Users who like popular genres (pop, hip-hop, electronic) get many high-quality matches from the dataset; niche-genre users (e.g., grindcore, iranian) have fewer catalog options and lower diversity.
  - **Default weight bias**: Default weights (genre=10.0, mood=5.0, energy=5.0, tempo=10.0) were tuned to one person's initial preferences; other users may need completely different weights but won't know to adjust them.
  - **High-energy bias**: Many weights favor high-energy attributes; a user who prefers mellow, low-energy music might struggle to get good results without heavily customizing weights.
  - **English-language & Western assumption**: The system assumes English-language music preferences and Western genre categories, disadvantaging non-Western users or those interested in music outside Spotify's dominant markets.  

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested
  - **Profile 1: Happy/Energetic Emo Fan** (favorite_genre='emo', favorite_mood='happy', target_energy=0.8, likes_acoustic=False, weight_genre=0.5, weight_mood=5.0, weight_tempo=10.0)
    - Results: Top song "This Is Why" scored 39.71, all recommendations were upbeat emo tracks (e.g., "Carry Me Away", "Somebody - Edit")
  - **Profile 2: Chill/Acoustic Listener** (favorite_mood='chill', target_energy=0.5, likes_acoustic=True, low genre weight)
    - Results: Diverse international recommendations (Japanese "ただ声一つ", Portuguese "Exu", Indian "Idhayathai Kolluriyeh"), scores ~35
  - **Profile 3: Pop/Happy User** (favorite_genre='pop', favorite_mood='happy', target_energy=0.8, likes_acoustic=False)
    - Results: Same recommendations as Profile 1 (emo songs scored higher), showing genre weight of 0.5 had minimal effect
  - **Profile 4: Weight Sensitivity Test** (same as Profile 1, but tested with weight_tempo=0, weight_valence=0)
    - Results: Scores dropped from 39 to ~27, recommendations shifted to different tracks, showing numerical attributes dramatically affect rankings

- What did you look for in the recommendations?
  - **Attribute match accuracy**: Did recommendations match stated preferences (genre, mood, energy)?
  - **Weight responsiveness**: Did changing a weight actually change the top-5 rankings, or were results unchanged?
  - **Diversity**: Were all 5 recommendations similar, or were there variations in sub-style, language, or production?
  - **Coherence**: Did explanations accurately reflect why songs scored highly? (e.g., "matches your favorite genre" only appeared for genre matches)
  - **Comparative behavior**: How did recommendations differ when mood weight changed from 10.0 to 1.0, or when tempo/valence were disabled?

- What surprised you?
  - **Pop genre didn't override emo scores**: When favorite_genre='pop' with low weight (0.5), the system still recommended emo songs. This showed that mood weight (5.0) and tempo weight (10.0) dominated genre matching, contrary to intuition.
  - **Chill + acoustic revealed diverse global catalog**: Removing genre as a filter (low weight) unlocked remarkably diverse, international recommendations (Japanese, Portuguese, Indian songs), suggesting the dataset's diversity was hidden by genre bias.
  - **Numerical attributes can dominate categorical ones**: With weight_tempo=10.0, tempo similarity (±5 BPM) contributed more points than matching favorite_genre. This flipped expectations about what "matters most."
  - **Very little difference in scores**: Top 5 recommendations ranged from 39.19 to 39.71 (0.52 point spread), making rankings fragile—small weight changes could reorder songs dramatically.

- Any simple tests or comparisons you ran
  - **Weight ablation test**: Disabled weight_tempo and weight_valence to measure their contribution (10-12 point score impact, ~27% of total score).
  - **Mood weight sweep**: Changed weight_mood from 10.0 → 1.0 to observe ranking changes in detail.
  - **Genre filter test**: Changed favorite_genre from 'emo' to 'pop' to test if high-weight attributes could override it.
  - **Comparison to real recommenders**: Informally compared results to Spotify's actual recommendation behavior and noted the system captured "sonic similarity" but missed serendipity and cultural context that real systems sometimes provide.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences
  - **Lyrical themes and keywords**: Allow users to specify preferred topics (e.g., "love songs," "breakup anthems," "nature imagery") or disliked words/phrases, using basic NLP to scan lyrics.
  - **Instrumentation preferences**: Add options for instrument emphasis (e.g., "guitar-heavy," "synth-driven," "percussion-focused") or vocal style (e.g., "male vocals," "female harmonies," "rap delivery").
  - **Artist identity filters**: Include preferences for artist nationality, gender, career stage (e.g., "emerging artists," "veteran musicians"), or cultural background to promote diversity or focus on specific communities.
  - **Era and production style**: Add decade preferences (e.g., "80s synth-pop," "90s grunge") or production quality (e.g., "lo-fi," "studio-polished," "live recordings").
  - **Song structure complexity**: Preferences for musical complexity (e.g., "simple verse-chorus," "complex arrangements," "instrumental tracks").
  - **Contextual preferences**: Time-of-day or activity-based settings (e.g., "morning upbeat," "workout high-energy," "evening chill").
  - **Mood evolution**: Allow playlists to evolve moods (e.g., start energetic, end mellow) rather than static preferences.
  - **Collaborative signals**: Basic user similarity based on shared preferences, or "users who liked X also liked Y" suggestions from simulated user data.
  - **Subgenre precision**: Expand genre options to subgenres (e.g., "emo-pop" vs. "screamo") with user-defined custom categories.
  - **Personal resonance**: Memory-based preferences (e.g., "songs from my childhood era," "tracks associated with happy memories") using self-reported data.

- Better ways to explain recommendations
  - **Detailed score breakdowns**: Instead of bullet points, show a visual bar chart or table with each attribute's contribution (e.g., "Genre match: +5.0 points," "Tempo similarity: +8.5 points") and percentage of total score.
  - **Natural language summaries**: Generate conversational explanations like "This energetic pop song fits your upbeat mood perfectly, though it's a bit faster than your ideal tempo—still a great match overall!"
  - **Comparative context**: Explain why this song was chosen over similar ones (e.g., "Preferred over 'Song B' because of better mood alignment, even though 'Song B' had higher popularity").
  - **User-customizable explanations**: Allow users to choose explanation style (e.g., "technical" with formulas, "casual" with stories, "visual" with charts).
  - **Context-aware narratives**: Tie explanations to user context (e.g., "Perfect for your morning routine—high energy to start the day, with acoustic elements you enjoy").
  - **Feedback loops**: Let users rate explanation helpfulness and use that to improve future explanations (e.g., highlight attributes users care about most).
  - **Counterfactuals**: Show what would change if weights were adjusted (e.g., "If you increased mood weight, this song would score even higher").
  - **Multi-song playlist narratives**: For playlists, explain the overall flow (e.g., "Starts energetic, transitions to mellow for a complete listening experience").  
- Improving diversity among the top results
  - **Re-ranking with diversity penalties**: Use Maximal Marginal Relevance (MMR) to penalize songs too similar to already-selected ones, balancing relevance vs. novelty (e.g., reduce score by 20% for songs in same subgenre as top pick).
  - **Diversity constraints**: Enforce minimum variety thresholds (e.g., at least 3 different genres in top 5, or no more than 2 songs from same artist/decade).
  - **Clustering-based selection**: Group songs by attributes (genre, mood, tempo range) and select proportionally from each cluster to ensure balanced representation.
  - **User-controlled diversity slider**: Add a "diversity vs. relevance" knob where users trade off perfect matches for variety (e.g., 0% = all top-scoring, 100% = maximum spread).
  - **Temporal and cultural spread**: Prioritize songs from different eras (decades) and regions/languages to avoid over-representing mainstream English tracks.
  - **Subgenre and style variation**: Within a genre, promote different sub-styles (e.g., for rock: mix punk, indie, classic) and production types (studio vs. live, acoustic vs. electric).
  - **Artist diversity**: Limit recommendations from same artist and favor emerging vs. established artists to surface new discoveries.
  - **Mood evolution in playlists**: Structure top results as a "journey" with gradual mood shifts (e.g., start energetic, transition to mellow) rather than static ranking.  
- Handling more complex user tastes
  - **Contextual profiles**: Allow multiple preference sets for different scenarios (e.g., "workout" profile: high energy/tempo, "study" profile: low energy/acoustic, "party" profile: danceable/high valence).
  - **Preference hierarchies**: Support primary/secondary/tertiary preferences with cascading weights (e.g., must-match genre first, then optimize for mood/energy).
  - **Anti-preferences and exclusions**: Let users specify disliked attributes (e.g., "no rap vocals," "avoid slow tempos," "exclude specific artists/genres") with negative scoring.
  - **Dynamic preference evolution**: Track listening history to gradually adjust weights (e.g., if user skips high-energy songs, reduce energy weight over time).
  - **Mood-aware recommendations**: Infer user mood from time/context and adjust preferences accordingly (e.g., morning: uplifting, evening: mellow).
  - **Hybrid collaborative-content filtering**: Combine personal preferences with "users like you also liked" signals from simulated user data.
  - **Ambiguous taste handling**: Support fuzzy preferences (e.g., "I like most genres but hate country," "tempo around 120±20 BPM") with flexible matching.
  - **Personality-based matching**: Tie preferences to personality traits (e.g., introverts: acoustic/chill, extroverts: energetic/social) using self-reported profiles.
  - **Social and group preferences**: Allow blending preferences from multiple users for group listening (e.g., average weights, compromise on conflicting tastes).
  - **Feedback-driven refinement**: Use explicit ratings (thumbs up/down) and implicit signals (skip rates, replay counts) to fine-tune recommendations over sessions.  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems
Building this system taught me that recommender systems are fundamentally about balancing competing priorities: relevance (matching user preferences) vs. diversity (avoiding echo chambers), and that even simple weighted scoring can reveal surprising dynamics like how numerical attributes (tempo, energy) can dominate categorical ones (genre, mood) in ways that defy intuition.
I learned that dataset bias is inescapable—our Spotify-derived data amplified mainstream English-language tracks and underrepresented niche genres, showing how real-world recommenders inherit and amplify platform biases.
Most importantly, I discovered that content-based filtering captures only the "sonic fingerprint" of music but misses the "human meaning"—lyrics, cultural context, personal memories—that make recommendations truly resonant, highlighting why hybrid approaches and human curation remain essential.  
- Something unexpected or interesting you discovered
The most surprising discovery was how dataset diversity was completely hidden by genre bias—when I lowered the genre weight for a chill/acoustic user, the system suddenly recommended songs from Japan, Portugal, and India, revealing a rich global catalog that was otherwise invisible behind the dominant "study" and "black-metal" genres.
I was surprised that numerical attributes like tempo could dominate categorical ones like genre; with weight_tempo=10.0, a 5 BPM difference contributed more points than matching the favorite_genre, flipping my intuition about what "matters most" in music preferences.
Another interesting finding was the fragility of rankings: the top 5 songs often scored within 0.5 points of each other, meaning tiny weight adjustments could completely reorder recommendations, showing how sensitive recommender systems are to parameter tuning.  
- How has this changed the way you think about music recommendation apps?
Building this system completely shifted my perspective on apps like Spotify and Apple Music—I now see them as delicate balancing acts between algorithmic precision and human serendipity, where the "perfect" recommendation might not be the most enjoyable one, and diversity is as important as relevance to avoid creating echo chambers.
I used to think recommendation algorithms were mostly about sophisticated machine learning, but now I realize that even simple weighted matching can work surprisingly well for basic preferences, yet real apps need hybrid approaches combining content, collaborative filtering, and human curation to capture the emotional and cultural layers that pure data can't touch.
This experience also made me appreciate the ethical responsibility of these systems—they're not neutral tools but powerful shapers of musical discovery that can amplify biases, limit exposure to new cultures, or reinforce existing tastes, reminding me that human judgment in design and oversight is irreplaceable.  

