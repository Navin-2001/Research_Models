"""
Chatbot Emotion Analysis Service (Hugging Face)

Uses local transformers for:
- Emotion detection from user messages
- Empathetic response generation
- Depression severity scoring (PHQ-9 inspired, 0-27 scale)

Port: 5008
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot Emotion Analysis Service")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models will be loaded on startup
emotion_classifier = None
response_generator = None

# PHQ-9 inspired question topics
PHQ9_TOPICS = [
    "overall mood and feelings",
    "energy levels and fatigue", 
    "sleep quality",
    "interest in activities",
    "appetite changes",
    "self-perception and self-worth",
    "concentration and focus",
    "physical restlessness or slowness",
    "hope for the future"
]

# Empathetic response templates based on detected emotion
RESPONSE_TEMPLATES = {
    "sadness": [
        "I hear that you're going through a difficult time. {follow_up}",
        "It sounds like things have been challenging. {follow_up}",
        "Thank you for sharing that with me. {follow_up}",
    ],
    "anger": [
        "I can sense some frustration there. {follow_up}",
        "It's understandable to feel that way. {follow_up}",
    ],
    "fear": [
        "It takes courage to share these feelings. {follow_up}",
        "Those concerns are valid. {follow_up}",
    ],
    "joy": [
        "That's wonderful to hear! {follow_up}",
        "I'm glad things are going well in that area. {follow_up}",
    ],
    "neutral": [
        "I understand. {follow_up}",
        "Thank you for sharing. {follow_up}",
    ]
}

# Follow-up questions for each PHQ-9 topic
FOLLOW_UP_QUESTIONS = [
    "Can you tell me more about how you've been feeling emotionally?",
    "Have you been feeling tired or lacking energy lately?",
    "How well have you been sleeping recently?",
    "Have you been able to enjoy activities that you usually like?",
    "How has your appetite been? Any changes?",
    "How have you been feeling about yourself lately?",
    "Have you been able to concentrate on things like reading or watching TV?",
    "Have you noticed any changes in how you move or speak?",
    "How do you feel about the future?",
]

# Request/Response models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
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


@app.on_event("startup")
async def load_models():
    """Load emotion classification model on startup."""
    global emotion_classifier
    
    logger.info("Loading emotion classification model...")
    
    try:
        from transformers import pipeline
        
        # Use a lighter emotion model for faster inference
        # This model detects: anger, disgust, fear, joy, neutral, sadness, surprise
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        
        logger.info("Emotion model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("Service will still run but with fallback responses")


def analyze_sentiment(text: str) -> tuple[str, float, Dict[str, float]]:
    """Analyze sentiment of text using emotion classifier."""
    if not emotion_classifier:
        return "neutral", 0.5, {"neutral": 1.0}
    
    try:
        results = emotion_classifier(text[:512])  # Limit text length
        
        if results and len(results) > 0:
            # results is a list of lists of dicts
            emotions = results[0] if isinstance(results[0], list) else results
            
            # Build emotion breakdown
            breakdown = {e['label']: e['score'] for e in emotions}
            
            # Find dominant emotion
            dominant = max(emotions, key=lambda x: x['score'])
            return dominant['label'], dominant['score'], breakdown
            
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
    
    return "neutral", 0.5, {"neutral": 1.0}


def calculate_depression_score(sentiment: str, score: float, text: str) -> int:
    """
    Calculate depression score (0-3) for a single response.
    Based on PHQ-9 scoring: 0=Not at all, 1=Several days, 2=More than half, 3=Nearly every day
    """
    text_lower = text.lower()
    
    # Negative indicators
    negative_words = ['always', 'never', 'terrible', 'awful', 'hopeless', 'worthless', 
                      'can\'t', 'impossible', 'every day', 'constantly', 'exhausted']
    moderate_negative = ['sometimes', 'often', 'difficult', 'hard', 'tired', 'sad',
                        'worried', 'anxious', 'struggle']
    mild_negative = ['occasionally', 'a bit', 'slightly', 'little']
    positive_words = ['good', 'great', 'fine', 'well', 'happy', 'hopeful', 'energetic',
                     'better', 'improving', 'positive']
    
    # Count indicators
    severe_count = sum(1 for word in negative_words if word in text_lower)
    moderate_count = sum(1 for word in moderate_negative if word in text_lower)
    mild_count = sum(1 for word in mild_negative if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    # Base score on emotion
    if sentiment in ['sadness', 'fear']:
        base_score = 2 if score > 0.6 else 1
    elif sentiment in ['anger', 'disgust']:
        base_score = 1
    elif sentiment == 'joy':
        base_score = 0
    else:
        base_score = 1
    
    # Adjust based on word analysis
    if severe_count >= 2:
        base_score = 3
    elif severe_count >= 1:
        base_score = max(base_score, 2)
    elif moderate_count >= 2:
        base_score = max(base_score, 2)
    elif moderate_count >= 1:
        base_score = max(base_score, 1)
    
    # Positive words reduce score
    if positive_count >= 2:
        base_score = max(0, base_score - 1)
    
    return min(3, max(0, base_score))


def generate_response(sentiment: str, question_index: int) -> str:
    """Generate empathetic response with follow-up question."""
    import random
    
    # Map emotions to template categories
    emotion_map = {
        'sadness': 'sadness',
        'fear': 'fear', 
        'anger': 'anger',
        'disgust': 'anger',
        'joy': 'joy',
        'surprise': 'neutral',
        'neutral': 'neutral'
    }
    
    template_key = emotion_map.get(sentiment, 'neutral')
    templates = RESPONSE_TEMPLATES.get(template_key, RESPONSE_TEMPLATES['neutral'])
    template = random.choice(templates)
    
    # Get next question if available
    if question_index < len(FOLLOW_UP_QUESTIONS) - 1:
        follow_up = FOLLOW_UP_QUESTIONS[question_index + 1]
    else:
        follow_up = "Thank you for sharing all of this with me."
    
    return template.format(follow_up=follow_up)


def get_mood_level(score: int) -> str:
    """Convert total score to mood level."""
    if score < 5:
        return "Minimal"
    elif score < 10:
        return "Mild"
    elif score < 15:
        return "Moderate"
    else:
        return "Severe"


def generate_summary(mood_level: str, emotions: Dict[str, float]) -> str:
    """Generate a summary based on the assessment."""
    dominant_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
    emotion_names = [e[0] for e in dominant_emotions]
    
    summaries = {
        "Minimal": f"Based on our conversation, you seem to be in a relatively positive headspace. "
                   f"The predominant emotions detected were {' and '.join(emotion_names)}. "
                   f"Keep up the good work maintaining your mental wellbeing!",
        "Mild": f"Our chat suggests you may be experiencing some mild stress or low mood. "
                f"The emotions {' and '.join(emotion_names)} came through in your responses. "
                f"Consider practicing self-care and reaching out to loved ones.",
        "Moderate": f"The conversation indicates you may be going through a challenging time. "
                    f"I noticed {' and '.join(emotion_names)} in your responses. "
                    f"Please consider speaking with a trusted friend, family member, or counselor.",
        "Severe": f"Based on what you've shared, it seems like you're facing significant difficulties. "
                  f"I detected {' and '.join(emotion_names)} throughout our conversation. "
                  f"Please reach out to a mental health professional or call a support helpline."
    }
    
    return summaries.get(mood_level, summaries["Mild"])


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "Chatbot Emotion Service Running (Port 5008)",
        "model_loaded": emotion_classifier is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user message and return empathetic response with analysis."""
    try:
        # Analyze sentiment
        sentiment, sentiment_score, _ = analyze_sentiment(request.message)
        
        # Calculate depression score for this response
        depression_score = calculate_depression_score(
            sentiment, sentiment_score, request.message
        )
        
        # Generate response
        reply = generate_response(sentiment, request.question_index)
        
        # Check if complete (9 questions for PHQ-9)
        is_complete = request.question_index >= 8
        
        return ChatResponse(
            reply=reply,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            depression_score=depression_score,
            is_complete=is_complete
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze full conversation and return final assessment."""
    try:
        total_score = 0
        all_emotions: Dict[str, float] = {}
        
        # Analyze each user message
        user_messages = [m for m in request.messages if m.role == "user"]
        
        for i, msg in enumerate(user_messages):
            sentiment, score, breakdown = analyze_sentiment(msg.content)
            
            # Add to total score
            dep_score = calculate_depression_score(sentiment, score, msg.content)
            total_score += dep_score
            
            # Aggregate emotions
            for emotion, value in breakdown.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + value
        
        # Normalize emotion scores
        if user_messages:
            for emotion in all_emotions:
                all_emotions[emotion] /= len(user_messages)
        
        # Cap score at 27 (PHQ-9 max)
        total_score = min(27, total_score)
        
        # Get mood level and summary
        mood_level = get_mood_level(total_score)
        summary = generate_summary(mood_level, all_emotions)
        
        return AnalyzeResponse(
            total_score=total_score,
            mood_level=mood_level,
            summary=summary,
            emotion_breakdown=all_emotions
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
