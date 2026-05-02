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

# Add expression detection to path
sys.path.append(os.path.join(os.path.dirname(__file__), "expression_detection", "app"))
try:
    from utils import detect_emotion as deepface_detect_emotion
except ImportError:
    # Fallback if structure is different
    def deepface_detect_emotion(img): return {"error": "DeepFace utils not found"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Wellness Tracker ML APIs (Hugging Face API Version)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API Configuration
# Get your token from https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL_VOICE = "https://api-inference.huggingface.co/models/superb/wav2vec2-base-superb-er"
API_URL_CHAT = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
API_URL_WHISPER = "https://api-inference.huggingface.co/models/openai/whisper-tiny.en"

def query_hf_api(api_url, data, is_binary=False):
    if not HF_TOKEN:
        logger.error("HF_TOKEN is not set in environment variables")
        raise HTTPException(status_code=500, detail="Server configuration error: HF_TOKEN missing")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        if is_binary:
            response = requests.post(api_url, headers=headers, data=data)
        else:
            response = requests.post(api_url, headers=headers, json=data)
            
        # Handle API loading state
        if response.status_code == 503:
            # Model is loading on Hugging Face side
            wait_time = response.json().get("estimated_time", 20)
            logger.info(f"HF Model is loading, waiting {wait_time}s...")
            time.sleep(2) # Brief wait before retrying or informing user
            raise HTTPException(status_code=503, detail="AI Model is starting up on Hugging Face. Please try again in a few seconds.")
            
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"HF API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("Server starting in HF API Mode. No heavy models will be loaded locally.")
    if not HF_TOKEN:
        logger.warning("WARNING: HF_TOKEN is not set. API calls will fail.")

@app.get("/")
def health_check():
    return {
        "status": "Wellness API Running (HF API Mode)",
        "hf_token_configured": HF_TOKEN is not None
    }

# ==========================================
# 1. VOICE RECOGNITION API
# ==========================================

ALL_VOICE_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

@app.post("/predict_emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    content = await audio_file.read()
    
    # Send directly to HF API
    results = query_hf_api(API_URL_VOICE, content, is_binary=True)
    
    # Process results to match your existing frontend format
    label_map = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
    final_scores = {k: 0.0 for k in ALL_VOICE_EMOTIONS}
    
    for item in results:
        short_label = item['label']
        score_pct = item['score'] * 100
        mapped_label = label_map.get(short_label)
        if mapped_label:
            final_scores[mapped_label] = score_pct
            
    return final_scores

# ==========================================
# 2. TRANSCRIPTION API
# ==========================================

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    content = await audio_file.read()
    
    try:
        result = query_hf_api(API_URL_WHISPER, content, is_binary=True)
        return {
            "success": True,
            "text": result.get("text", "").strip()
        }
    except Exception as e:
        return {"success": False, "text": f"Transcription error: {str(e)}"}

# ==========================================
# 3. CHATBOT THERAPY API
# ==========================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    question_index: int = 0

class ChatResponse(BaseModel):
    reply: str
    sentiment: str
    sentiment_score: float
    depression_score: int
    is_complete: bool = False

def analyze_sentiment_hf(text: str) -> tuple[str, float, Dict[str, float]]:
    try:
        # Use return_all_scores equivalent on HF API
        results = query_hf_api(API_URL_CHAT, {"inputs": text, "parameters": {"return_all_scores": True}})
        
        # HF API returns list of lists if multiple inputs, or single list
        emotions = results[0] if isinstance(results[0], list) else results
        breakdown = {e['label']: e['score'] for e in emotions}
        dominant = max(emotions, key=lambda x: x['score'])
        return dominant['label'], dominant['score'], breakdown
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return "neutral", 0.5, {"neutral": 1.0}

# (Existing logic for calculating depression scores remains same but uses HF sentiment)
def calculate_depression_score(sentiment: str, score: float, text: str) -> int:
    text_lower = text.lower()
    negative_words = ['always', 'never', 'terrible', 'awful', 'hopeless', 'worthless', 'can\'t', 'impossible', 'every day', 'constantly', 'exhausted']
    moderate_negative = ['sometimes', 'often', 'difficult', 'hard', 'tired', 'sad', 'worried', 'anxious', 'struggle']
    
    severe_count = sum(1 for word in negative_words if word in text_lower)
    moderate_count = sum(1 for word in moderate_negative if word in text_lower)
    
    if sentiment in ['sadness', 'fear']: base_score = 2 if score > 0.6 else 1
    elif sentiment in ['anger', 'disgust']: base_score = 1
    elif sentiment == 'joy': base_score = 0
    else: base_score = 1
    
    if severe_count >= 2: base_score = 3
    elif severe_count >= 1: base_score = max(base_score, 2)
    elif moderate_count >= 1: base_score = max(base_score, 1)
    
    return min(3, max(0, base_score))

RESPONSE_TEMPLATES = {
    "sadness": ["I hear that you're going through a difficult time. {follow_up}", "It sounds like things have been challenging. {follow_up}"],
    "anger": ["I can sense some frustration there. {follow_up}"],
    "joy": ["That's wonderful to hear! {follow_up}"],
    "neutral": ["I understand. {follow_up}"]
}

FOLLOW_UP_QUESTIONS = [
    "Can you tell me more about how you've been feeling emotionally?", "Have you been feeling tired or lacking energy lately?",
    "How well have you been sleeping recently?", "Have you been able to enjoy activities that you usually like?",
    "How has your appetite been? Any changes?", "How have you been feeling about yourself lately?",
    "Have you been able to concentrate on things like reading or watching TV?", "Have you noticed any changes in how you move or speak?",
    "How do you feel about the future?"
]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    sentiment, sentiment_score, _ = analyze_sentiment_hf(request.message)
    depression_score = calculate_depression_score(sentiment, sentiment_score, request.message)
    
    # Generate response
    import random
    emotion_map = {'sadness': 'sadness', 'fear': 'sadness', 'joy': 'joy'}
    template_key = emotion_map.get(sentiment, 'neutral')
    template = random.choice(RESPONSE_TEMPLATES.get(template_key, RESPONSE_TEMPLATES['neutral']))
    
    follow_up = FOLLOW_UP_QUESTIONS[request.question_index + 1] if request.question_index < 8 else "Thank you for sharing."
    reply = template.format(follow_up=follow_up)
    
    return ChatResponse(
        reply=reply, sentiment=sentiment, sentiment_score=sentiment_score,
        depression_score=depression_score, is_complete=request.question_index >= 8
    )

# ==========================================
# 4. FACIAL EXPRESSION API
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
        with open(image_path, "wb") as f: f.write(img_data)

        emotions = deepface_detect_emotion(image_path)
        if os.path.exists(image_path): os.remove(image_path)
        return {"status": "success", "emotions": emotions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
