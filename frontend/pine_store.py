import os, time, uuid, json
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)  # Changed: add override=True to match routes.py

EMBED_MODEL = "text-embedding-3-small"
_openai = OpenAI()

# Lazy Pinecone
_PINECONE_KEY = os.getenv("PINECONE_API_KEY")
_PINECONE_INDEX = os.getenv("PINECONE_INDEX", "conversations")
_pc = None
_index = None
_init = False

def _lazy():
    global _pc, _index, _init
    if _init:
        return
    _init = True
    if not _PINECONE_KEY:
        print("Pinecone disabled (no key).")
        return
    try:
        from pinecone import Pinecone
        _pc = Pinecone(api_key=_PINECONE_KEY)
        _index = _pc.Index(_PINECONE_INDEX)
        print(f"Pinecone connected: {_PINECONE_INDEX}")
    except Exception as e:
        print("Pinecone init failed:", e)

def _embed(text: str) -> List[float]:
    r = _openai.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding

def store_message(session_id: str, role: str, content: str):
    _lazy()
    if not _index:
        return
    try:
        vec = _embed(content)
        print(f"[Pinecone] storing {role} message for session {session_id}")
        _index.upsert(vectors=[{
            "id": f"{session_id}-{int(time.time()*1000)}-{role}-{uuid.uuid4().hex[:8]}",
            "values": vec,
            "metadata": {"session_id": session_id, "role": role, "content": content}
        }], namespace=session_id)
        print(f"[Pinecone] stored successfully")
    except Exception as e:
        print("Pinecone store error:", e)

# Add feedback storage function
def store_feedback(session_id: str, message_pair_id: str, feedback_type: str, feedback_data: dict):
    """Store feedback about AI responses for continuous learning"""
    _lazy()
    if not _index:
        return
    try:
        # Create embedding from the feedback context
        context = f"{feedback_type}: {feedback_data.get('issue', '')} {feedback_data.get('suggestion', '')}"
        vec = _embed(context)
        
        vid = f"feedback-{session_id}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        
        # FIXED: Flatten complex data to strings for Pinecone metadata
        meta = {
            "type": "feedback",
            "session_id": session_id,
            "message_pair_id": message_pair_id,
            "feedback_type": feedback_type,
            "feedback_data_json": json.dumps(feedback_data),  # Convert to JSON string
            "timestamp": time.time()
        }
        
        # Extract key fields as separate metadata for easier querying
        if isinstance(feedback_data, dict):
            if "grammar_score" in feedback_data:
                meta["grammar_score"] = feedback_data["grammar_score"]
            if "estimated_level" in feedback_data:
                meta["estimated_level"] = feedback_data["estimated_level"]
            if "vocabulary_level" in feedback_data:
                meta["vocabulary_level"] = feedback_data["vocabulary_level"]
        
        _index.upsert(vectors=[{"id": vid, "values": vec, "metadata": meta}], namespace="feedback")
        print(f"[Feedback] stored {feedback_type} feedback")
    except Exception as e:
        print("Feedback store error:", e)

def get_feedback_patterns(feedback_type: str = None, limit: int = 50):
    """Retrieve feedback patterns for analysis"""
    _lazy()
    if not _index:
        return []
    try:
        # Query feedback namespace
        query_filter = {"type": "feedback"}
        if feedback_type:
            query_filter["feedback_type"] = feedback_type
            
        # Use a dummy vector for metadata-only search
        dummy_vec = [0.0] * 1536
        res = _index.query(
            vector=dummy_vec,
            top_k=limit,
            namespace="feedback",
            include_metadata=True,
            filter=query_filter
        )
        
        # Parse JSON data back to objects
        results = []
        for match in res.get("matches", []):
            meta = match["metadata"].copy()
            # Parse JSON string back to dict
            if "feedback_data_json" in meta:
                try:
                    import json
                    meta["feedback_data"] = json.loads(meta["feedback_data_json"])
                    del meta["feedback_data_json"]  # Remove the JSON string version
                except:
                    meta["feedback_data"] = {}
            results.append(meta)
        
        return results
    except Exception as e:
        print("Feedback retrieval error:", e)
        return []

def semantic_search(session_id: str, query: str, top_k: int = 5):
    _lazy()
    if not _index:
        return []
    try:
        qv = _embed(query)
        res = _index.query(vector=qv, top_k=top_k, namespace=session_id, include_metadata=True)
        return [
            {
                "score": m.get("score"),
                "role": m["metadata"].get("role"),
                "content": m["metadata"].get("content")
            } for m in res.get("matches", [])
        ]
    except Exception as e:
        print("Pinecone search error:", e)
        return []