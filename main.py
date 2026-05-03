import os
import io
import time
import shutil
import base64
import uuid
import logging
import sys
import requests
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Wellness Tracker ML APIs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HF Token (only used for optional transcription)
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL_WHISPER = "https://api-inference.huggingface.co/models/openai/whisper-base"

# Lazy load DeepFace to save RAM on startup
deepface_detect_emotion = None

def get_deepface():
    global deepface_detect_emotion
    if deepface_detect_emotion is None:
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "expression_detection", "app"))
            from utils import detect_emotion
            deepface_detect_emotion = detect_emotion
            logger.info("DeepFace loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load DeepFace: {e}")
            def fallback(img): return {"error": "Face detection unavailable"}
            deepface_detect_emotion = fallback
    return deepface_detect_emotion

# ==========================================
# HEALTH CHECK
# ==========================================

@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "running", "mode": "local-processing + optional-hf"}

@app.get("/check_ffmpeg")
def check_ffmpeg():
    return {"ffmpeg_found": shutil.which("ffmpeg") is not None}

# ==========================================
# 1. VOICE EMOTION - Local librosa analysis
# (No HF API needed - uses audio features)
# ==========================================

@app.post("/predict_emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    """Analyze voice emotion using local librosa audio features."""
    import librosa

    content = await audio_file.read()
    temp_path = f"/tmp/voice_{uuid.uuid4()}.wav"

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        # Load audio with librosa
        try:
            y, sr = librosa.load(temp_path, sr=16000, duration=10.0)
        except Exception as e:
            logger.warning(f"Librosa load failed: {e}")
            return _neutral_voice_scores()

        if len(y) == 0:
            return _neutral_voice_scores()

        # Extract audio features
        energy = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) if not isinstance(tempo, float) else tempo

        # Compute spectral features
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfccs[0]))

        # Rule-based emotion mapping from audio features
        scores = _audio_features_to_emotions(energy, zcr, tempo, spec_centroid, mfcc_mean)
        logger.info(f"Voice analysis complete: {scores}")
        return scores

    except Exception as e:
        logger.error(f"Voice emotion analysis error: {e}")
        return _neutral_voice_scores()
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass


def _neutral_voice_scores():
    return {"neutral": 60.0, "calm": 20.0, "happy": 0.0, "sad": 0.0,
            "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0}


def _audio_features_to_emotions(energy, zcr, tempo, spec_centroid, mfcc_mean):
    """Map raw audio features to emotion scores using heuristic rules."""
    scores = {"neutral": 0.0, "calm": 0.0, "happy": 0.0, "sad": 0.0,
              "angry": 0.0, "fearful": 0.0, "disgust": 0.0, "surprised": 0.0}

    # High energy + high ZCR + fast tempo → angry/stressed
    # Low energy + low ZCR + slow tempo → sad/calm
    # Medium energy + medium tempo → neutral/happy

    if energy > 0.1 and zcr > 0.15 and tempo > 120:
        scores["angry"] = 45.0
        scores["fearful"] = 25.0
        scores["neutral"] = 20.0
        scores["surprised"] = 10.0
    elif energy < 0.03 and tempo < 80:
        scores["sad"] = 40.0
        scores["calm"] = 35.0
        scores["neutral"] = 25.0
    elif energy > 0.06 and tempo > 100:
        scores["happy"] = 40.0
        scores["neutral"] = 35.0
        scores["calm"] = 25.0
    else:
        # Default neutral/calm
        scores["neutral"] = 50.0
        scores["calm"] = 30.0
        scores["happy"] = 10.0
        scores["sad"] = 10.0

    return scores


# ==========================================
# 2. TRANSCRIPTION - Optional HF API
# ==========================================

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Try HF Whisper API, return empty on failure (non-critical)."""
    content = await audio_file.read()

    if not HF_TOKEN:
        return {"success": False, "text": ""}

    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(API_URL_WHISPER, headers=headers, data=content, timeout=20)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict):
                text = result.get("text", "")
            elif isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", result[0].get("text", ""))
            else:
                text = ""
            return {"success": bool(text.strip()), "text": text.strip()}
        else:
            logger.warning(f"Whisper API returned {response.status_code}")
            return {"success": False, "text": ""}
    except Exception as e:
        logger.warning(f"Transcription failed (non-critical): {e}")
        return {"success": False, "text": ""}


# ==========================================
# 3. CHAT - Keyword-based sentiment (Reliable)
# ==========================================

NEGATIVE_SEVERE = ['always', 'never', 'hopeless', 'worthless', 'impossible',
                   'every day', 'constantly', 'exhausted', 'can\'t go on', 'give up']
NEGATIVE_MODERATE = ['sometimes', 'often', 'difficult', 'hard', 'tired', 'sad',
                     'worried', 'anxious', 'struggle', 'depressed', 'lonely']
NEGATIVE_MILD = ['a bit', 'slightly', 'little', 'occasionally', 'not great']
POSITIVE = ['good', 'great', 'fine', 'well', 'happy', 'hopeful', 'energetic',
            'better', 'improving', 'positive', 'wonderful', 'excited']

FOLLOW_UPS = [
    "How have you been feeling emotionally?",
    "Have you been feeling tired or lacking energy lately?",
    "How well have you been sleeping recently?",
    "Have you been able to enjoy activities you usually like?",
    "How has your appetite been? Any changes?",
    "How have you been feeling about yourself lately?",
    "Have you been able to concentrate on things?",
    "Have you noticed changes in how you move or speak?",
    "How do you feel about the future?"
]

RESPONSES = {
    "negative": ["I hear that you're going through a difficult time. {q}",
                 "It sounds like things have been challenging. {q}"],
    "positive": ["That's good to hear! {q}", "I'm glad things are going okay. {q}"],
    "neutral":  ["I understand. {q}", "Thank you for sharing. {q}"]
}

def _keyword_sentiment(text: str):
    t = text.lower()
    sev = sum(1 for w in NEGATIVE_SEVERE if w in t)
    mod = sum(1 for w in NEGATIVE_MODERATE if w in t)
    mild = sum(1 for w in NEGATIVE_MILD if w in t)
    pos = sum(1 for w in POSITIVE if w in t)

    if sev >= 1 or mod >= 2: return "sadness", 0.8
    if mod >= 1 or mild >= 1: return "sadness", 0.5
    if pos >= 1: return "joy", 0.7
    return "neutral", 0.5

class ChatRequest(BaseModel):
    message: str
    question_index: int = 0

@app.post("/chat")
async def chat(request: ChatRequest):
    import random
    sentiment, score = _keyword_sentiment(request.message)

    dep_score = 0
    if sentiment == "sadness":
        dep_score = 2 if score > 0.6 else 1
    elif sentiment == "joy":
        dep_score = 0

    templates = RESPONSES.get("negative" if sentiment == "sadness" else
                              "positive" if sentiment == "joy" else "neutral")
    template = random.choice(templates)
    next_q_idx = min(request.question_index + 1, len(FOLLOW_UPS) - 1)
    reply = template.format(q=FOLLOW_UPS[next_q_idx])

    return {
        "reply": reply,
        "sentiment": sentiment,
        "sentiment_score": score,
        "depression_score": dep_score,
        "is_complete": request.question_index >= 8
    }


# ==========================================
# 4. FACE EMOTION - DeepFace (Local, Working)
# ==========================================

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/api/emotion")
async def face_emotion_api(payload: dict = Body(...)):
    if "image_base64" not in payload:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    try:
        img_data = base64.b64decode(payload["image_base64"])
        image_path = os.path.join(UPLOAD_FOLDER, f"input_{uuid.uuid4()}.jpg")
        with open(image_path, "wb") as f:
            f.write(img_data)
        detector = get_deepface()
        emotions = detector(image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        return {"status": "success", "emotions": emotions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
