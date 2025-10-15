# TeamXtreme Language Learning Assistant

Enthusiastic conversational language practice assistant (Russell-style) with:
- Conversation memory + adaptive system prompt
- User proficiency analysis (background)
- Feedback-driven personalization
- Semantic search + contextual feedback
- Optional TTS (OpenAI first, gTTS fallback)

## Tech Stack
- Backend: Flask
- AI: OpenAI Responses + TTS
- Embeddings / Vector search: Pinecone (via pine_store)
- Speech: OpenAI tts-1-hd (fallback gTTS)
- Env management: python-dotenv
- Async background tasks: threading

## Folder Structure (partial)
```
/TeamXtreme
  routes.py        # Flask API
  pine_store.py    # (not shown) vector + feedback storage helpers
  conv_manager.py  # (not shown) in‑memory conversation manager
  .env             # environment variables
  tts_out/         # generated mp3 files (auto-created)
```

## Prerequisites
- Python 3.10+
- pip
- OpenAI API key
- Pinecone API key + existing index (name must match PINECONE_INDEX)
- (Optional) Node/React app consuming REACT_APP_API_BASE

## Environment Variables (.env)
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pc-...
PINECONE_INDEX=hackathon
REACT_APP_API_BASE=http://localhost:8000
```
Notes:
- Ensure index exists in Pinecone (dimension + model must align with embeddings you use in pine_store).
- Never commit real keys.

## Setup (Backend)
```
git clone <repo>
cd TeamXtreme
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install flask flask-cors openai gTTS python-dotenv requests pinecone-client
```
(Adjust pinecone package name if using new SDK: pip install pinecone-client or pinecone)

## Run
```
cp .env.example .env   # if you create one; else edit existing
# fill keys
python routes.py
```
Server: http://localhost:8000/api/health → {"status":"ok"}

## Key Endpoints
- POST /api/converse  body: { text, lang, session_id, tts } -> AI reply + optional audio_url
- POST /api/analyze/user  (internal async call) proficiency JSON
- POST /api/user/feedback  store learner feedback (too_easy, too_hard, confused, etc.)
- GET  /api/progress/<session_id>  aggregated progress metrics
- GET  /api/analytics/<session_id> conversation-level analysis
- GET  /api/search?session_id=...&q=... semantic search
- GET  /api/audio/<fname> served mp3

## Conversation Personalization Flow
1. First user message triggers building a personalized system prompt.
2. Cached feedback (progress + learning_experience) adjusts difficulty.
3. Each converse request:
   - Stores user + assistant messages
   - Runs contextual semantic search for similar past feedback
   - Spawns background proficiency analysis
   - Optionally generates TTS

## TTS
Primary: OpenAI tts-1-hd (voice=alloy).  
Fallback: gTTS (language inferred from lang param prefix).

## Pinecone Setup (High Level)
1. Create index (e.g., name: hackathon).
2. Dimension must match embedding model (check pine_store implementation).
3. Set PINECONE_API_KEY + PINECONE_INDEX.
4. Ensure region/environment matches client initialization in pine_store.py.

## Frontend (If React)
Use REACT_APP_API_BASE to call:
```
POST ${API}/api/converse
fetch('/api/audio/<file>') for playback
```

## Session IDs
Use a stable session_id per learner/device to persist adaptive behavior.

## Error Handling
- Missing OPENAI_API_KEY raises at startup.
- TTS failure gracefully omits audio_url.
- Semantic search failures return empty arrays silently where possible.

## Troubleshooting
| Issue | Fix |
|-------|-----|
| 401 / OpenAI error | Verify OPENAI_API_KEY + network |
| Pinecone not found | Confirm index name + region |
| Audio not returned | Check logs for TTS error; disable tts flag |
| JSON parsing errors in analysis | Usually malformed model output; logged silently |

## Security Notes
- Do not expose raw keys to frontend.
- Consider rate limiting / auth before production.
- Move in-memory conv_manager to persistent store (Redis/Postgres) for scale.

## Extending
- Add embeddings generation route if not inside pine_store.
- Swap TTS voice by changing voice param.
- Add streaming responses (currently disabled for simplicity).

## Cleaning Up
To remove old audio:
```
find tts_out -type f -mtime +1 -delete
```


## Next Ideas
- Improved avatar to show boddy langauge while replying
- Multi-turn correction summaries
- CEFR adaptive goals
- More insightful user feedback on their conversation
- Frontend transcript + highlighted corrections
