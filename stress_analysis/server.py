from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Social Media Stress Analysis API", description="AI analysis of social media patterns and insights")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DailyUsage(BaseModel):
    duration_minutes: int
    frequency: int
    late_night_logins: int

class UsageData(BaseModel):
    patterns: List[DailyUsage]

@app.post("/analyze/patterns")
def analyze_patterns(data: UsageData):
    """
    Analyzes raw usage patterns (time, duration, frequency)
    and returns a behavioral risk assessment.
    """
    if not data.patterns:
        return {"error": "No data provided"}
        
    total_duration = sum(d.duration_minutes for d in data.patterns)
    avg_duration = total_duration / len(data.patterns)
    avg_freq = sum(d.frequency for d in data.patterns) / len(data.patterns)
    
    risk_level = "Low"
    if avg_duration > 180 or avg_freq > 30:
        risk_level = "High"
    elif avg_duration > 120 or avg_freq > 15:
        risk_level = "Medium"

    return {
        "analysis_type": "Pattern Behavior",
        "risk_level": risk_level,
        "average_duration": round(avg_duration, 1),
        "average_frequency": round(avg_freq, 1),
        "recommendation": "Try implementing 20-minute screen breaks." if risk_level == "High" else "Your usage patterns are within healthy boundaries."
    }

@app.post("/analyze/insights")
def analyze_insights(data: UsageData):
    """
    Provides deep insights into how social media usage 
    might be affecting stress and sleep.
    """
    if not data.patterns:
        return {"error": "No data provided"}

    late_nights = sum(d.late_night_logins for d in data.patterns)
    total_days = len(data.patterns)
    
    sleep_impact = "None"
    if late_nights > total_days: 
        sleep_impact = "High"
    elif late_nights > 0:
        sleep_impact = "Moderate"

    return {
        "analysis_type": "Wellbeing Insights",
        "sleep_disruption_level": sleep_impact,
        "late_night_logins_total": late_nights,
        "psychological_insight": "Late-night social media engagement heavily disrupts circadian rhythms." if sleep_impact == "High" else "Good job avoiding screens before bed."
    }

if __name__ == "__main__":
    # Running on a completely separate port (5001) as requested
    print("Starting Social Media Stress Analysis API on port 5001...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
