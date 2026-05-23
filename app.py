import os
import base64
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mahanth-ai-secret-2024")
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

SYSTEM_PROMPT = """You are Mahanth AI — a superintelligent, friendly, and highly capable AI assistant created by Mahanth.
You combine the best of all AI assistants: you can search knowledge like Perplexity, reason like ChatGPT, 
be creative like Gemini, and be thoughtful like Claude.

Key traits:
- Answer ANY question thoroughly and accurately
- Analyze images when provided
- Provide well-structured responses with clear formatting using markdown
- Be conversational yet precise
- Support all languages naturally — reply in the same language the user writes in
- Always be helpful

When analyzing images: describe what you see in detail and answer any questions about them.
Format responses with markdown (bold, lists, code blocks) when it helps clarity."""


def call_gemini(messages, image_data=None, image_type=None):
    """Call Gemini API with conversation history and optional image."""
    
    # Build contents array from history
    contents = []
    
    # Add conversation history
    for msg in messages[:-1]:  # All except last
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    
    # Build the last user message (may include image)
    last_parts = []
    
    if image_data:
        # Strip data URL prefix
        if "," in image_data:
            image_data = image_data.split(",")[1]
        last_parts.append({
            "inline_data": {
                "mime_type": image_type or "image/jpeg",
                "data": image_data
            }
        })
    
    last_msg = messages[-1]["content"]
    if last_msg:
        last_parts.append({"text": last_msg})
    elif not image_data:
        last_parts.append({"text": "Hello"})
    
    if not last_parts:
        last_parts.append({"text": "Analyze this image."})
    
    contents.append({
        "role": "user",
        "parts": last_parts
    })
    
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 2048,
        }
    }
    
    response = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=60
    )
    
    if response.status_code != 200:
        raise Exception(f"Gemini API error {response.status_code}: {response.text}")
    
    data = response.json()
    
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise Exception("Unexpected response format from Gemini")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    image_data = data.get("image")
    image_type = data.get("image_type", "image/jpeg")
    history = data.get("history", [])

    if not user_message and not image_data:
        return jsonify({"error": "No message or image provided"}), 400

    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set in .env file"}), 500

    # Build messages list
    messages = []
    for h in history[-8:]:  # Last 8 turns for context
        messages.append({"role": h["role"], "content": h["content"]})
    
    # Add current user message
    msg_text = user_message if user_message else ("Please analyze this image." if image_data else "")
    messages.append({"role": "user", "content": msg_text})

    try:
        reply = call_gemini(messages, image_data, image_type)
        return jsonify({"reply": reply, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    status = "ready" if GEMINI_API_KEY else "missing API key"
    return jsonify({"status": "Mahanth AI is running!", "api": status})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🤖 Mahanth AI starting on http://localhost:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)
