import os
import io
import time
import shutil
import base64
import uuid
import logging
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

# Additional ML Libraries
import librosa
import numpy as np

# Add expression detection to path so DeepFace utils loads
sys.path.append(os.path.join(os.path.dirname(__file__), "expression_detection", "app"))
from utils import detect_emotion as deepface_detect_emotion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Wellness Tracker ML APIs (Voice, Chat, Face)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
voice_classifier = None
chat_emotion_classifier = None

# ==========================================
# 1. SHARED SETUP & FASTAPI EVENTS
# ==========================================

@app.on_event("startup")
async def load_models():
    """Load all ML models into memory to unify the endpoint."""
    global voice_classifier, chat_emotion_classifier
    
    logger.info("Starting up Unified ML Server...")
    logger.info("Loading Models... This will take a moment.")
    
    try:
        from transformers import pipeline
        
        # Load Voice Model
        logger.info("[Voice] Loading Wav2Vec2 audio classification...")
        voice_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        
        # Load Chat Emotion Model
        logger.info("[Chat] Loading distilroberta text classification...")
        chat_emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        
        logger.info("All Hugging Face models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load Hugging Face models: {e}")
        logger.info("Service will still run, but specific endpoints may fail.")

    # Setup FFmpeg for audio processing
    setup_ffmpeg()

from pathlib import Path

def setup_ffmpeg():
    """Locate FFmpeg and FFprobe for pydub to handle .m4a audio conversions."""
    ffmpeg_exe = shutil.which("ffmpeg")
    ffprobe_exe = shutil.which("ffprobe")
    
    if not (ffmpeg_exe and ffprobe_exe):
        common_ffmpeg_paths = [
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",       # Homebrew on Apple Silicon
            "/usr/bin/ffmpeg",
            str(Path.home() / "bin" / "ffmpeg"),
            r"C:\ffmpeg\bin\ffmpeg.exe",      # Windows Fallbacks
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        ]
        
        for path in common_ffmpeg_paths:
            if os.path.exists(path):
                if not ffmpeg_exe:
                    ffmpeg_exe = path
                    
                # Setup Environment PATH
                os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_exe)
                
                # Guess ffprobe in same dir
                if not ffprobe_exe:
                    base_name = "ffprobe.exe" if path.endswith(".exe") else "ffprobe"
                    ffprobe_guess = os.path.join(os.path.dirname(ffmpeg_exe), base_name)
                    if os.path.exists(ffprobe_guess):
                        ffprobe_exe = ffprobe_guess
                break

    if ffmpeg_exe:
        logger.info(f"FFmpeg ready at: {ffmpeg_exe} (m4a support encoded via subprocess)")
    else:
        logger.warning(f"WARNING: FFmpeg not found. Decoding WILL fail.")


@app.get("/")
def health_check():
    """Unified Health check endpoint."""
    return {
        "status": "Unified Wellness Models Running",
        "voice_model_loaded": voice_classifier is not None,
        "chat_model_loaded": chat_emotion_classifier is not None,
    }

# ==========================================
# 2. VOICE RECOGNITION API
# ==========================================

ALL_VOICE_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

@app.get("/check_ffmpeg")
def check_ffmpeg():
    ffmpeg_exe = shutil.which("ffmpeg")
    return {
        "ffmpeg_found": ffmpeg_exe is not None,
        "path": ffmpeg_exe,
        "advice": "FFmpeg is ready." if ffmpeg_exe else "Install FFmpeg or point the server to its directory."
    }

@app.post("/predict_emotion")
async def predict_emotion(audio_file: UploadFile = File(...)):
    """Voice emotion detection imported from vocal_emotion_hf."""
    if not voice_classifier:
        raise HTTPException(status_code=503, detail="Voice model is still loading")

    file_ext = os.path.splitext(audio_file.filename)[1] if audio_file.filename else '.m4a'
    temp_filename = f"temp_{uuid.uuid4()}{file_ext}"
    
    try:
        logger.info(f"Receiving audio file: {audio_file.filename}")
        
        content = await audio_file.read()
        with open(temp_filename, "wb") as buffer:
            buffer.write(content)
            
        speech, sr = None, None
        is_wav = temp_filename.lower().endswith('.wav') or (audio_file.filename and audio_file.filename.lower().endswith('.wav'))

        try:
            if is_wav:
                import soundfile as sf
                with open(temp_filename, 'rb') as f:
                    speech, sr_org = sf.read(f)
                    if speech.ndim > 1:
                        speech = speech.mean(axis=1)
                    if sr_org != 16000:
                        speech = librosa.resample(speech, orig_sr=sr_org, target_sr=16000)
                    sr = 16000
            else:
                try:
                    speech, sr = librosa.load(temp_filename, sr=16000, duration=10.0)
                except Exception as lib_err:
                    logger.warning(f"Librosa direct load failed: {lib_err}")
                    import subprocess
                    import soundfile as sf
                    
                    wav_filename = f"{temp_filename}.wav"
                    ffmpeg_cmd = [
                        shutil.which("ffmpeg") or "ffmpeg", 
                        "-y", "-i", temp_filename, 
                        "-ar", "16000", "-ac", "1", 
                        wav_filename
                    ]
                    
                    try:
                        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        speech, sr_org = sf.read(wav_filename)
                        sr = 16000
                    except Exception as ffmpeg_err:
                        raise Exception(f"Subprocess FFmpeg conversion failed: {ffmpeg_err}")
                    finally:
                        try:
                            if os.path.exists(wav_filename):
                                os.remove(wav_filename)
                        except:
                            pass
        except Exception as err:
            err_msg = str(err)
            logger.error(f"====== AUDIO LOAD CRASH DETAILED ======\n{err_msg}\n=======================================")
            if "find the file specified" in err_msg or "[WinError 2]" in err_msg:
                raise HTTPException(status_code=400, detail="FFmpeg is not installed on the server.")
            raise HTTPException(status_code=400, detail=f"Could not load audio file: {err_msg}")
        
        if speech is None or len(speech) == 0:
            logger.error("speech variable is None or empty length after processing")
            raise HTTPException(status_code=400, detail="Audio file is empty or invalid")
        
        labels = voice_classifier(speech, top_k=None) 
        
        label_map = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
        final_scores = {k: 0.0 for k in ALL_VOICE_EMOTIONS}
        
        for item in labels:
            short_label = item['label']
            score_pct = item['score'] * 100
            mapped_label = label_map.get(short_label)
            if mapped_label:
                final_scores[mapped_label] = score_pct
        
        return final_scores

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            await audio_file.close()
            if 'speech' in locals(): del speech
            if 'audio' in locals(): del audio
            
            for _ in range(3):
                try:
                    os.remove(temp_filename)
                    break
                except Exception:
                    time.sleep(0.1)


# ==========================================
# 3. CHATBOT THERAPY API
# ==========================================

PHQ9_TOPICS = [
    "overall mood and feelings", "energy levels and fatigue", "sleep quality",
    "interest in activities", "appetite changes", "self-perception and self-worth",
    "concentration and focus", "physical restlessness or slowness", "hope for the future"
]

RESPONSE_TEMPLATES = {
    "sadness": ["I hear that you're going through a difficult time. {follow_up}", "It sounds like things have been challenging. {follow_up}", "Thank you for sharing that with me. {follow_up}"],
    "anger": ["I can sense some frustration there. {follow_up}", "It's understandable to feel that way. {follow_up}"],
    "fear": ["It takes courage to share these feelings. {follow_up}", "Those concerns are valid. {follow_up}"],
    "joy": ["That's wonderful to hear! {follow_up}", "I'm glad things are going well in that area. {follow_up}"],
    "neutral": ["I understand. {follow_up}", "Thank you for sharing. {follow_up}"]
}

FOLLOW_UP_QUESTIONS = [
    "Can you tell me more about how you've been feeling emotionally?", "Have you been feeling tired or lacking energy lately?",
    "How well have you been sleeping recently?", "Have you been able to enjoy activities that you usually like?",
    "How has your appetite been? Any changes?", "How have you been feeling about yourself lately?",
    "Have you been able to concentrate on things like reading or watching TV?", "Have you noticed any changes in how you move or speak?",
    "How do you feel about the future?"
]

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

class AnalyzeRequest(BaseModel):
    messages: List[ChatMessage]

class AnalyzeResponse(BaseModel):
    total_score: int
    mood_level: str
    summary: str
    emotion_breakdown: Dict[str, float]

def analyze_sentiment(text: str) -> tuple[str, float, Dict[str, float]]:
    if not chat_emotion_classifier:
        return "neutral", 0.5, {"neutral": 1.0}
    
    try:
        results = chat_emotion_classifier(text[:512])
        if results and len(results) > 0:
            emotions = results[0] if isinstance(results[0], list) else results
            breakdown = {e['label']: e['score'] for e in emotions}
            dominant = max(emotions, key=lambda x: x['score'])
            return dominant['label'], dominant['score'], breakdown
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
    return "neutral", 0.5, {"neutral": 1.0}

def calculate_depression_score(sentiment: str, score: float, text: str) -> int:
    text_lower = text.lower()
    negative_words = ['always', 'never', 'terrible', 'awful', 'hopeless', 'worthless', 'can\'t', 'impossible', 'every day', 'constantly', 'exhausted']
    moderate_negative = ['sometimes', 'often', 'difficult', 'hard', 'tired', 'sad', 'worried', 'anxious', 'struggle']
    mild_negative = ['occasionally', 'a bit', 'slightly', 'little']
    positive_words = ['good', 'great', 'fine', 'well', 'happy', 'hopeful', 'energetic', 'better', 'improving', 'positive']
    
    severe_count = sum(1 for word in negative_words if word in text_lower)
    moderate_count = sum(1 for word in moderate_negative if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    if sentiment in ['sadness', 'fear']: base_score = 2 if score > 0.6 else 1
    elif sentiment in ['anger', 'disgust']: base_score = 1
    elif sentiment == 'joy': base_score = 0
    else: base_score = 1
    
    if severe_count >= 2: base_score = 3
    elif severe_count >= 1: base_score = max(base_score, 2)
    elif moderate_count >= 2: base_score = max(base_score, 2)
    elif moderate_count >= 1: base_score = max(base_score, 1)
    
    if positive_count >= 2: base_score = max(0, base_score - 1)
    return min(3, max(0, base_score))

def generate_response(sentiment: str, question_index: int) -> str:
    import random
    emotion_map = {'sadness': 'sadness', 'fear': 'fear', 'anger': 'anger', 'disgust': 'anger', 'joy': 'joy', 'surprise': 'neutral', 'neutral': 'neutral'}
    template_key = emotion_map.get(sentiment, 'neutral')
    templates = RESPONSE_TEMPLATES.get(template_key, RESPONSE_TEMPLATES['neutral'])
    template = random.choice(templates)
    
    if question_index < len(FOLLOW_UP_QUESTIONS) - 1:
        follow_up = FOLLOW_UP_QUESTIONS[question_index + 1]
    else:
        follow_up = "Thank you for sharing all of this with me."
    return template.format(follow_up=follow_up)

def get_mood_level(score: int) -> str:
    if score < 5: return "Minimal"
    elif score < 10: return "Mild"
    elif score < 15: return "Moderate"
    else: return "Severe"

def generate_summary(mood_level: str, emotions: Dict[str, float]) -> str:
    dominant_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
    emotion_names = [e[0] for e in dominant_emotions]
    
    summaries = {
        "Minimal": f"Based on our conversation, you seem to be in a relatively positive headspace. The predominant emotions detected were {' and '.join(emotion_names)}. Keep up the good work maintaing your mental wellbeing!",
        "Mild": f"Our chat suggests you may be experiencing some mild stress or low mood. The emotions {' and '.join(emotion_names)} came through in your responses. Consider practicing self-care and reaching out to loved ones.",
        "Moderate": f"The conversation indicates you may be going through a challenging time. I noticed {' and '.join(emotion_names)} in your responses. Please consider speaking with a trusted friend, family member, or counselor.",
        "Severe": f"Based on what you've shared, it seems like you're facing significant difficulties. I detected {' and '.join(emotion_names)} throughout our conversation. Please reach out to a mental health professional or call a support helpline."
    }
    return summaries.get(mood_level, summaries["Mild"])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        sentiment, sentiment_score, _ = analyze_sentiment(request.message)
        depression_score = calculate_depression_score(sentiment, sentiment_score, request.message)
        reply = generate_response(sentiment, request.question_index)
        is_complete = request.question_index >= 8
        
        return ChatResponse(
            reply=reply, sentiment=sentiment, sentiment_score=sentiment_score,
            depression_score=depression_score, is_complete=is_complete
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        total_score = 0
        all_emotions: Dict[str, float] = {}
        user_messages = [m for m in request.messages if m.role == "user"]
        
        for msg in user_messages:
            sentiment, score, breakdown = analyze_sentiment(msg.content)
            dep_score = calculate_depression_score(sentiment, score, msg.content)
            total_score += dep_score
            for emotion, value in breakdown.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + value
        
        if user_messages:
            for emotion in all_emotions:
                all_emotions[emotion] /= len(user_messages)
        
        total_score = min(27, total_score)
        mood_level = get_mood_level(total_score)
        summary = generate_summary(mood_level, all_emotions)
        
        return AnalyzeResponse(total_score=total_score, mood_level=mood_level, summary=summary, emotion_breakdown=all_emotions)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 4. FACIAL EXPRESSION API
# ==========================================

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class FaceEmotionRequest(BaseModel):
    image_base64: str

@app.post("/api/emotion")
async def face_emotion_api(payload: dict = Body(...)):
    """Converted from Flask DeepFace routing."""
    if "image_base64" not in payload:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    try:
        # Decode Base64 image
        img_data = base64.b64decode(payload["image_base64"])
        image_path = os.path.join(UPLOAD_FOLDER, f"input_{uuid.uuid4()}.jpg")

        with open(image_path, "wb") as f:
            f.write(img_data)

        # DeepFace Processing
        emotions = deepface_detect_emotion(image_path)
        
        # Cleanup
        try:
            os.remove(image_path)
        except:
            pass
            
        return {
            "status": "success",
            "emotions": emotions
        }
    except Exception as e:
        logger.error(f"Face emotion analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
