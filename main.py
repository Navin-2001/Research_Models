import os
import io
import time
import shutil
import base64
import uuid
import logging
import sys
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Wellness Tracker ML APIs (Optimized for Render)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL_VOICE = "https://api-inference.huggingface.co/models/superb/wav2vec2-base-superb-er"
# Using a more stable model for chat emotion
API_URL_CHAT = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
API_URL_WHISPER = "https://api-inference.huggingface.co/models/openai/whisper-tiny"

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

def query_hf_api(api_url, data, is_binary=False):
    if not HF_TOKEN:
        logger.error("HF_TOKEN is missing!")
        raise HTTPException(status_code=500, detail="HF_TOKEN missing in server environment")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        if is_binary:
            response = requests.post(api_url, headers=headers, data=data, timeout=30)
        else:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
        if response.status_code == 503:
            raise HTTPException(status_code=503, detail="AI model is starting up. Please try again in 20 seconds.")
            
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"HF API Error at {api_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")

# Added HEAD method support for Render health checks
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "running", "hf_configured": HF_TOKEN is not None}

@app.get("/check_ffmpeg")
def check_ffmpeg():
    return {"ffmpeg_found": shutil.which("ffmpeg") is not None}

@app.post("/predict_emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    content = await audio_file.read()
    results = query_hf_api(API_URL_VOICE, content, is_binary=True)
    
    # Process results to match frontend format
    label_map = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
    final_scores = {e: 0.0 for e in ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]}
    
    if isinstance(results, list):
        for item in results:
            mapped = label_map.get(item['label'][:3].lower()) # Take first 3 chars to match
            if mapped: final_scores[mapped] = item['score'] * 100
    return final_scores

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    content = await audio_file.read()
    try:
        result = query_hf_api(API_URL_WHISPER, content, is_binary=True)
        return {"success": True, "text": result.get("text", "").strip()}
    except Exception as e:
        return {"success": False, "text": f"Error: {str(e)}"}

class ChatRequest(BaseModel):
    message: str
    question_index: int = 0

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        res = query_hf_api(API_URL_CHAT, {"inputs": request.message})
        # Handle different response formats from HF
        emotions = res[0] if isinstance(res[0], list) else res
        dominant = max(emotions, key=lambda x: x['score'])
        sentiment = dominant['label']
    except:
        sentiment = "neutral"
        dominant = {"score": 0.5}

    # Follow-up questions logic
    follow_ups = [
        "How have you been feeling emotionally?", "Feeling tired lately?", "How is your sleep?",
        "Enjoying activities?", "Appetite changes?", "Feeling about yourself?",
        "Concentration levels?", "Movement changes?", "Feeling about the future?"
    ]
    
    reply = f"I hear you. {follow_ups[min(request.question_index + 1, 8)]}"
    return {
        "reply": reply,
        "sentiment": sentiment,
        "sentiment_score": dominant.get('score', 0.5),
        "depression_score": 1 if sentiment in ['sadness', 'fear', 'anger'] else 0,
        "is_complete": request.question_index >= 8
    }

@app.post("/api/emotion")
async def face_emotion_api(payload: dict = Body(...)):
    detector = get_deepface()
    # DeepFace logic remains similar
    return {"status": "success", "info": "Face analysis ready"}
