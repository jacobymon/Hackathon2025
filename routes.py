import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gtts import gTTS
from openai import OpenAI
from conv_manager import conv_manager
from dotenv import load_dotenv  # added
# Update this import line at the top
from pine_store import store_message, semantic_search, store_feedback, get_feedback_patterns
import threading
import time
import requests


app = Flask(__name__)
CORS(app)

load_dotenv(override=True)  # load variables from .env if present

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key)
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "tts_out")
os.makedirs(AUDIO_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful conversational partner helping the user practice target language. Respond concisely."

@app.route("/api/health")
def health():
    return {"status": "ok"}

@app.route("/api/audio/<path:fname>")
def get_audio(fname):
    return send_from_directory(AUDIO_DIR, fname, mimetype="audio/mpeg", as_attachment=False)

@app.route("/api/converse", methods=["POST"])
def converse():
    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    target_lang = data.get("lang", "en")
    session_id = data.get("session_id") or "default"
    tts = bool(data.get("tts", True))

    if not user_text:
        return jsonify(error="Empty text"), 400

    # Add system prompt once
    if not conv_manager.get_history(session_id):
        conv_manager.append(session_id, "system", SYSTEM_PROMPT + f" Target language: {target_lang}.")

    # User message
    conv_manager.append(session_id, "user", user_text)
    try:
        store_message(session_id, "user", user_text)
    except Exception as e:
        print("Store user error:", e)

    messages = conv_manager.get_history(session_id)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Much faster than gpt-5 (~0.5s vs 2s)
            messages=messages,
        )
        
        reply = completion.choices[0].message.content.strip()
        # Remove the streaming loop - just get reply directly
    except Exception as e:
        print("OpenAI error:", e)
        return jsonify(error=str(e)), 500
    
    # Store messages
    conv_manager.append(session_id, "assistant", reply)
    try:
        store_message(session_id, "assistant", reply)
    except Exception as e:
        print("Store assistant error:", e)

    # Background analysis of BOTH messages
    def background_analysis():
        try:
            # Only analyze user's language proficiency (not AI quality)
            user_analysis_data = {
                "user_message": user_text,
                "session_id": session_id,
                "target_lang": target_lang
            }
            requests.post("http://localhost:8000/api/analyze/user", 
                            json=user_analysis_data, timeout=5)
        except:
            pass  # Silent fail

    threading.Thread(target=background_analysis, daemon=True).start()

    audio_filename = None
    if tts:
        try:
            audio_filename = f"{uuid.uuid4().hex}.mp3"
            # Use faster TTS settings
            tts_lang = target_lang.split('-')[0]  # Use base language code
            tts_obj = gTTS(reply, lang=tts_lang, slow=False)  # Ensure fast speech
            tts_obj.save(os.path.join(AUDIO_DIR, audio_filename))
        except Exception as e:
            print("TTS error:", e)
            audio_filename = None

    return jsonify(
        session_id=session_id,
        reply=reply,
        audio_url=f"/api/audio/{audio_filename}" if audio_filename else None
    )

@app.route("/api/search", methods=["GET"])
def search():
    session_id = request.args.get("session_id")
    q = request.args.get("q", "").strip()
    if not session_id or not q:
        return jsonify(error="session_id and q required"), 400
    try:
        results = semantic_search(session_id, q)
        return jsonify(results=results)
    except Exception as e:
        return jsonify(error=str(e)), 500
    
# @app.route("/api/analyze/response", methods=["POST"])
# def analyze_response():
#     """Analyze AI response quality and store feedback"""
#     data = request.get_json()
#     session_id = data.get("session_id")
#     user_message = data.get("user_message")
#     ai_response = data.get("ai_response")
    
#     if not all([session_id, user_message, ai_response]):
#         return jsonify(error="Missing required fields"), 400
    
#     try:
#         # Analyze response quality with GPT
#         analysis_prompt = f"""
#         Analyze this AI language learning response for quality and suggest improvements:
        
#         User: {user_message}
#         AI: {ai_response}
        
#         Evaluate and return JSON:
#         {{
#             "grammar_accuracy": 0-10,
#             "cultural_appropriateness": 0-10,
#             "engagement_level": 0-10,
#             "pedagogical_value": 0-10,
#             "issues": ["list of specific issues"],
#             "improvements": ["specific suggestions"]
#         }}
#         """
        
#         analysis = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": analysis_prompt}]
#         )
        
#         # Parse and store feedback
#         message_pair_id = f"{session_id}-{int(time.time()*1000)}"
        
#         # Store analysis as feedback in background
#         def store_analysis():
#             try:
#                 import json
#                 feedback_data = json.loads(analysis.choices[0].message.content)
#                 store_feedback(session_id, message_pair_id, "quality_analysis", feedback_data)
#             except:
#                 pass  # Silent fail for background task
        
#         threading.Thread(target=store_analysis, daemon=True).start()
        
#         return jsonify(analysis=analysis.choices[0].message.content)
        
#     except Exception as e:
#         return jsonify(error=str(e)), 500

# @app.route("/api/improve/suggestions", methods=["GET"])
# def get_improvement_suggestions():
#     """Get aggregated suggestions for improving AI responses"""
#     try:
#         # Get recent feedback patterns
#         feedback_data = get_feedback_patterns(limit=100)
        
#         if not feedback_data:
#             return jsonify(suggestions=[])
        
#         # Analyze patterns with GPT
#         feedback_summary = "\n".join([
#             f"Issue: {f.get('feedback_data', {}).get('issues', [])} | "
#             f"Improvements: {f.get('feedback_data', {}).get('improvements', [])}"
#             for f in feedback_data[-20:]  # Last 20 feedback items
#         ])
        
#         improvement_prompt = f"""
#         Based on this conversation feedback data, suggest 5 specific improvements 
#         for the AI language learning assistant:
        
#         {feedback_summary}
        
#         Return JSON: {{"improvements": ["improvement 1", "improvement 2", ...]}}
#         """
        
#         suggestions = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": improvement_prompt}]
#         )
        
#         return jsonify(
#             feedback_count=len(feedback_data),
#             suggestions=suggestions.choices[0].message.content
#         )
        
#     except Exception as e:
#         return jsonify(error=str(e)), 500

@app.route("/api/user/feedback", methods=["POST"])
def user_learning_feedback():
    """Store user's learning experience feedback"""
    data = request.get_json()
    session_id = data.get("session_id")
    learning_feedback = data.get("learning_feedback")
    ai_response = data.get("ai_response")
    user_message = data.get("user_message")
    
    try:
        # Store learning experience
        feedback_data = {
            "learning_feedback": learning_feedback,
            "ai_response": ai_response,
            "user_message": user_message,
            "timestamp": time.time()
        }
        
        store_feedback(session_id, f"learning-{int(time.time()*1000)}", 
                      "learning_experience", feedback_data)
        
        return jsonify(success=True)
        
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/analytics/<session_id>", methods=["GET"])
def get_analytics(session_id):
    try:
        # Get conversation history from memory
        messages = conv_manager.get_history(session_id)
        
        # Extract user messages for analysis
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        
        if not user_messages:
            return jsonify(error="No user messages found"), 404
            
        # Use GPT to analyze conversation for improvement areas
        analysis_prompt = """
        Analyze the following user messages for language learning insights. 
        Return JSON with: {"grammar_issues": [], "vocabulary_suggestions": [], "fluency_notes": []}
        
        User messages: """ + "\n".join(user_messages)
        
        analysis = client.chat.completions.create(
            model="gpt-4o-mini",  # Much faster than gpt-5
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        # Search for similar past conversations for context
        recent_searches = []
        for msg in user_messages[-3:]:  # Last 3 messages
            similar = semantic_search(session_id, msg, top_k=3)
            recent_searches.extend(similar)
        
        return jsonify(
            session_id=session_id,
            message_count=len(user_messages),
            analysis=analysis.choices[0].message.content,
            similar_conversations=recent_searches[:5]  # Top 5 similar
        )
        
    except Exception as e:
        return jsonify(error=str(e)), 500
    
# @app.route("/api/dashboard/metrics", methods=["GET"])
# def get_metrics():
#     """Get conversation quality metrics"""
#     try:
#         feedback = get_feedback_patterns(limit=200)
        
#         # Calculate metrics
#         total_conversations = len(set(f.get('session_id') for f in feedback))
#         avg_quality = sum(f.get('feedback_data', {}).get('grammar_accuracy', 0) for f in feedback) / max(len(feedback), 1)
        
#         common_issues = {}
#         for f in feedback:
#             issues = f.get('feedback_data', {}).get('issues', [])
#             for issue in issues:
#                 common_issues[issue] = common_issues.get(issue, 0) + 1
        
#         return jsonify(
#             total_conversations=total_conversations,
#             total_feedback=len(feedback),
#             avg_quality_score=round(avg_quality, 2),
#             top_issues=sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:10]
#         )
#     except Exception as e:
#         return jsonify(error=str(e)), 500
    
@app.route("/api/analyze/user", methods=["POST"])
def analyze_user_response():
    """Analyze user's language proficiency and progress"""
    data = request.get_json()
    session_id = data.get("session_id")
    user_message = data.get("user_message")
    target_lang = data.get("target_lang", "en")
    
    try:
        # Analyze user's language skills
        user_analysis_prompt = f"""
        Analyze this language learner's message for proficiency assessment:
        
        Target language: {target_lang}
        User message: "{user_message}"
        
        Evaluate and return JSON:
        {{
            "grammar_score": 0-10,
            "vocabulary_level": "beginner|intermediate|advanced",
            "fluency_indicators": ["natural phrases used", "complex structures"],
            "errors": [
                {{"type": "grammar", "error": "specific mistake", "correction": "suggested fix"}},
                {{"type": "vocabulary", "error": "word choice", "suggestion": "better word"}}
            ],
            "strengths": ["what they did well"],
            "focus_areas": ["what to practice next"],
            "estimated_level": "A1|A2|B1|B2|C1|C2"
        }}
        """
        
        analysis = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_analysis_prompt}]
        )
        
        # Store user progress data
        def store_user_progress():
            try:
                import json
                progress_data = json.loads(analysis.choices[0].message.content)
                store_feedback(session_id, f"user-{int(time.time()*1000)}", "user_progress", progress_data)
            except Exception as e:
                print("Store user progress error:", e)
        
        threading.Thread(target=store_user_progress, daemon=True).start()
        
        return jsonify(analysis=analysis.choices[0].message.content)
        
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/progress/<session_id>", methods=["GET"])
def get_user_progress(session_id):
    """Track user's learning progress over time"""
    try:
        # Get user progress feedback
        user_feedback = [f for f in get_feedback_patterns(feedback_type="user_progress", limit=50) 
                        if f.get('session_id') == session_id]
        
        if not user_feedback:
            return jsonify(
                session_id=session_id,
                total_messages_analyzed=0,
                message="No progress data yet"
            )
        
        # Calculate progress trends (with safe defaults)
        grammar_scores = []
        levels = []
        all_errors = []
        
        for f in user_feedback:
            feedback_data = f.get('feedback_data', {})
            
            # Safely extract grammar score
            if 'grammar_score' in feedback_data:
                grammar_scores.append(feedback_data['grammar_score'])
                
            # Safely extract level
            if 'estimated_level' in feedback_data:
                levels.append(feedback_data['estimated_level'])
                
            # Safely extract errors
            errors = feedback_data.get('errors', [])
            all_errors.extend([err.get('type', 'unknown') for err in errors if isinstance(err, dict)])
        
        # Count error patterns
        error_patterns = {}
        for error_type in all_errors:
            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        return jsonify(
            session_id=session_id,
            total_messages_analyzed=len(user_feedback),
            grammar_trend=grammar_scores,
            current_level=levels[-1] if levels else "Not assessed",
            level_progression=levels,
            common_error_types=sorted(error_patterns.items(), key=lambda x: x[1], reverse=True),
            avg_grammar_score=sum(grammar_scores) / len(grammar_scores) if grammar_scores else None
        )
        
    except Exception as e:
        return jsonify(error=str(e)), 500
    
if __name__ == "__main__":
    # export OPENAI_API_KEY=...
    app.run(host="0.0.0.0", port=8000, debug=True)
