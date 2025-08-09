import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gtts import gTTS
from openai import OpenAI
from conv_manager import conv_manager
from dotenv import load_dotenv  # added


app = Flask(__name__)
CORS(app)

load_dotenv()  # load variables from .env if present

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

    # Initialize history with system prompt once
    if not conv_manager.get_history(session_id):
        conv_manager.append(session_id, "system", SYSTEM_PROMPT + f" Target language: {target_lang}.")

    conv_manager.append(session_id, "user", user_text)

    messages = conv_manager.get_history(session_id)

    try:
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            temperature=0.7,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        return jsonify(error=str(e)), 500

    conv_manager.append(session_id, "assistant", reply)

    audio_filename = None
    if tts:
        try:
            audio_filename = f"{uuid.uuid4().hex}.mp3"
            gTTS(reply, lang=target_lang.split('-')[0]).save(os.path.join(AUDIO_DIR, audio_filename))
        except Exception:
            audio_filename = None  # Fail silently for TTS

    return jsonify(
        session_id=session_id,
        reply=reply,
        audio_url=f"/api/audio/{audio_filename}" if audio_filename else None
    )

if __name__ == "__main__":
    # export OPENAI_API_KEY=...
    app.run(host="0.0.0.0", port=8000, debug=True)
