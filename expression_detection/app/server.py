import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import detect_emotion
import base64

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/emotion", methods=["POST"])
def emotion_api():
    data = request.json
    
    if "image_base64" not in data:
        return jsonify({"error": "image_base64 is required"}), 400

    # Decode Base64 image
    img_data = base64.b64decode(data["image_base64"])
    image_path = os.path.join(UPLOAD_FOLDER, "input.jpg")

    with open(image_path, "wb") as f:
        f.write(img_data)

    try:
        emotions = detect_emotion(image_path)
        return jsonify({
            "status": "success",
            "emotions": emotions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "DeepFace Emotion Detection API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)

