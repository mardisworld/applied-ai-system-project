# Confidence Scoring

This document records an AI-generated assessment of how confidently the music recommender system works correctly across each of its major components. Scores are based on test coverage, observed runtime behavior, and known gaps.

Last updated: April 22, 2026

---

## Component Scores

| ---------Component-------------- |-Confidence-|-Evidence-|
|----------------------------------|------------|----------|
| **Prompt parsing** (`parse_query_signals`) | **90%** | 14 passing unit + edge-case tests cover mood, activity, seed artist, all seed-artist patterns, and no-signal queries. Minor gap: behavior when mood and activity signals semantically conflict is untested. |
| **Content-based scoring / ranking strategies** | **88%** | 12 passing tests including regression profiles, all 4 strategies verified to rank differently, diversity penalty confirmed. Risk: weights are hand-tuned heuristics with no user-satisfaction ground truth. |
| **RAG retrieval pipeline** | **85%** | 14 passing retrieval tests cover artist profile building, similar-artist matching, context serialization, LLM prompt rendering, and user-preference bridging. Risk: the local dataset covers a limited artist catalog, so many seed-artist queries won't find an exact match and fall back to genre/mood signals. |
| **End-to-end CLI pipeline** | **88%** | 14 passing integration tests run the full pipeline via the CLI in `retrieval-only` and `hybrid` modes, including all 4 ranking strategies, seed-artist queries, JSON output, injection blocking, and the `--persona` flag. Tests confirm output shape, score ordering, and required fields without requiring an API key. |
| **LLM grounding** | **75%** | 12 passing unit tests cover prompt construction, response parsing, hallucination guard, and the persona specialization layer. The actual LLM call is not exercised in automated tests (requires a live API key). Structured logging records model name, response length, active persona, and full error details (HTTP status + body) before re-raising. Output quality depends on the configured model and key availability; the system degrades gracefully. |
| **LLM persona specialization** | **92%** | 12 passing tests cover the `PERSONAS` registry structure, `build_persona_messages()` message ordering, few-shot injection for all three specialist personas, unknown-persona fallback to baseline, and vocabulary assertions on critic/DJ example turns. Structural differences are verified without a live model. Behavioral output differences (measured via sentence length, exclamation count, ALL-CAPS words, formal/casual vocab hits) are documented in `COMPARE_PERSONAS.md` and confirmed experimentally when an API key is present. |
| **Agentic workflow** (`src/agent.py`) | **90%** | 30 passing tests cover `AgentStep`, `AgentTrace`, the signal-based strategy planner, the full 6-step reasoning chain (sanitize → parse → plan → retrieve → rank → generate), injection blocking at the agent boundary, strategy overrides, pre-sanitized queries, JSON serializability, score ordering, and per-step decision recording. The `--agent` CLI flag is wired into the existing CLI and compatible with `--json`, `--strategy`, and `--persona`. Gap: the agent's LLM `generate` step is only tested in the skipped/error path (no live API key in CI). |
| **Spotify playlist creation** | **74%** | 27 unit tests cover credentials, auth URL, token exchange, track search, playlist creation, and `build_playlist_from_recommendations`. All HTTP calls are mocked. Structured logging captures HTTP error codes (including 403) with response body excerpts, improving debuggability of the developer-app restriction issue observed in manual testing. OAuth browser flow still not covered by automated tests. |

---

## Overall System Confidence: ~87%

The core recommendation loop — prompt → retrieval → scoring → ranked output — is well-tested and works reliably. Application-wide structured logging (via `src/logger.py` and Python's `logging` stdlib) now surfaces seed-artist resolution outcomes, candidate counts, LLM call details (including active persona), and Spotify API errors at configurable verbosity levels.

The persona specialization layer (critic, dj, friend, baseline) is fully tested structurally. Few-shot examples are injected into the `messages` array before each grounded query, steering model output style without any weight updates. Behavioral differences are documented in `COMPARE_PERSONAS.md`.

The agentic workflow (`src/agent.py`) wraps the full pipeline in a multi-step reasoning chain with observable intermediate steps. Each step records its name, decision, reasoning, duration, and status, making the system's behavior inspectable at every stage. The `--agent` CLI flag exposes this via both human-readable console output and structured JSON.

The main remaining risks are:

1. **Spotify playlist creation** (74%): the 403 error observed in manual testing is not reproduced by unit tests (which mock HTTP), so it may still affect real usage due to Spotify developer app restrictions. Logging records the full response body at ERROR level, aiding diagnosis.
2. **LLM layer** (75%): depends on runtime API key availability; no automated test covers a live LLM call. Log output shows model name, persona, and error details before propagating exceptions.
3. **Dataset coverage**: the enriched local dataset is limited in size, which reduces seed-artist match rate for less common artists.

---

## Test Suite Summary

| --------------File----------- | Tests| All Pass? |
|-------------------------------|------|------- ----|
| `tests/test_agent.py`                | 30 | ✅ |
| `tests/test_integration.py`          | 14 | ✅ |
| `tests/test_llm_client.py`           | 12 | ✅ |
| `tests/test_personas.py`             | 12 | ✅ |
| `tests/test_recommender.py`          | 12 | ✅ |
| `tests/test_retrieval.py`            | 14 | ✅ |
| `tests/test_spotify_api.py`          | 27 | ✅ |
| `tests/test_spotify_enrichment.py`   | 7  | ✅ |
| **Total** | **128** | **✅ 128 / 128**    | 

Run all tests with:

```bash
python -m pytest tests/ -v
```
