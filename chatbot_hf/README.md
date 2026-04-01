# Chatbot Emotion Analysis Service (Hugging Face)

This service provides AI-powered conversational depression assessment using local Hugging Face models.

## Setup

1. **Create Virtual Environment:**
   ```bash
   py -3.12 -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: First run downloads models (~500MB)*

## Running the Service

```bash
python -m uvicorn model:app --host 0.0.0.0 --port 5008 --reload
```

The service runs on Port **5008**.

## API Endpoints

### POST /chat
Send a message and get AI response with sentiment analysis.

**Request:**
```json
{
    "message": "I've been feeling very tired lately",
    "history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "question_index": 0
}
```

**Response:**
```json
{
    "reply": "I hear you. Feeling exhausted can really affect your day...",
    "sentiment": "sadness",
    "sentiment_score": 0.75,
    "depression_score": 2
}
```

### POST /analyze
Get final depression assessment from conversation.

**Request:**
```json
{
    "messages": [...]
}
```

**Response:**
```json
{
    "total_score": 12,
    "mood_level": "Moderate",
    "summary": "Based on our conversation...",
    "emotion_breakdown": {...}
}
```
