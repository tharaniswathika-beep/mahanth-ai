import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mahanth-ai-secret-2024")
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

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
    contents = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    last_parts = []
    if image_data:
        if "," in image_data:
            image_data = image_data.split(",")[1]
        last_parts.append({"inline_data": {"mime_type": image_type or "image/jpeg", "data": image_data}})

    last_msg = messages[-1]["content"]
    if last_msg:
        last_parts.append({"text": last_msg})
    elif not image_data:
        last_parts.append({"text": "Hello"})
    if not last_parts:
        last_parts.append({"text": "Analyze this image."})

    contents.append({"role": "user", "parts": last_parts})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 2048}
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
        raise Exception("Unexpected response from Gemini")


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
        return jsonify({"error": "No message provided"}), 400
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 500

    messages = []
    for h in history[-8:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message or "Analyze this image."})

    try:
        reply = call_gemini(messages, image_data, image_type)
        return jsonify({"reply": reply, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "Mahanth AI is running!"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🤖 Mahanth AI running on http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
