from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import pytesseract
from PIL import Image, ImageDraw
import io
import speech_recognition as sr
import tempfile
import os
import uuid
import json
import soundfile as sf
import matplotlib.pyplot as plt
import base64
from datetime import datetime
import hashlib
import time
import re
from collections import Counter
import requests
import asyncio
from urllib.parse import urlparse

app = FastAPI(
    title="MultiSense API",
    description="Advanced sentiment analysis with multi-modal input support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update as per your system
MAX_BATCH_SIZE = 5
ENABLE_CACHING = True
CACHE_TIMEOUT = 3600  # 1 hour
WEBHOOK_TIMEOUT = 10  # seconds

# Set up Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# In-memory storage
session_history = {}
analysis_cache = {}
registered_webhooks = {}
pending_tasks = {}

# Initialize models
sentiment_models = {
    "en": pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis"),
    "multilingual": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
}

# Initialize emotion detection model
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

# Initialize text summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize content moderation model
content_filter = pipeline("text-classification", model="michellejieli/inappropriate_text_classifier")

# Data models
class AnalysisRequest(BaseModel):
    text: str
    include_emotions: bool = False
    detect_language: bool = True
    generate_summary: bool = False
    session_id: Optional[str] = None

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., max_items=MAX_BATCH_SIZE)
    include_emotions: bool = False
    detect_language: bool = True
    generate_summary: bool = False
    session_id: Optional[str] = None

class WebhookRegistration(BaseModel):
    callback_url: str
    secret_token: Optional[str] = None
    events: List[str] = ["text", "image", "audio", "batch"]

class EmotionAnalysis(BaseModel):
    primary_emotion: str
    confidence: float
    secondary_emotions: List[Dict[str, float]]

class SentimentAnalysis(BaseModel):
    label: str
    score: float
    explanation: str
    emotions: Optional[EmotionAnalysis] = None
    language: Optional[str] = None
    summary: Optional[str] = None
    content_flags: Optional[Dict[str, float]] = None
    word_count: int
    processing_time: float

# Helper functions
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English if detection fails

def explain_sentiment(label: str, score: float) -> str:
    """Generate a human-readable explanation of the sentiment."""
    if label == "POS" or label == "positive" or label.startswith("5") or label.startswith("4"):
        if score > 0.9:
            return "The text expresses extremely positive sentiment with very high confidence."
        elif score > 0.7:
            return "The text expresses strongly positive sentiment."
        else:
            return "The text leans positive, but with some ambiguity."
    elif label == "NEU" or label == "neutral" or label.startswith("3"):
        return "The text appears to be primarily neutral or factual in nature."
    else:  # Negative
        if score > 0.9:
            return "The text expresses extremely negative sentiment with very high confidence."
        elif score > 0.7:
            return "The text expresses strongly negative sentiment."
        else:
            return "The text leans negative, but with some ambiguity."

def analyze_emotions(text: str) -> EmotionAnalysis:
    """Analyze emotions in the text."""
    results = emotion_model(text)
    
    primary = results[0]
    secondary = results[1:]
    
    return EmotionAnalysis(
        primary_emotion=primary["label"],
        confidence=primary["score"],
        secondary_emotions=[{"emotion": item["label"], "score": item["score"]} for item in secondary]
    )

def moderate_content(text: str) -> Dict[str, float]:
    """Check text for inappropriate content."""
    results = content_filter(text)
    return {
        "inappropriate": next((item["score"] for item in results if item["label"] == "inappropriate"), 0.0),
        "appropriate": next((item["score"] for item in results if item["label"] == "appropriate"), 0.0)
    }

def generate_text_summary(text: str) -> str:
    """Generate a summary of the text if it's long enough."""
    if len(text.split()) < 50:  # Don't summarize short texts
        return "Text too short for summarization"
    
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Summarization error: {str(e)}"

def get_cache_key(content: str, include_emotions: bool, detect_lang: bool, summarize: bool) -> str:
    """Generate a cache key for the request."""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"{content_hash}_{include_emotions}_{detect_lang}_{summarize}"

def should_use_cache(cache_key: str) -> bool:
    """Check if we should use the cached result."""
    if not ENABLE_CACHING or cache_key not in analysis_cache:
        return False
    
    timestamp, _ = analysis_cache[cache_key]
    return (time.time() - timestamp) < CACHE_TIMEOUT

def update_session_history(session_id: str, result: Dict[str, Any]) -> None:
    """Update the session history with the latest analysis result."""
    if not session_id:
        return
    
    if session_id not in session_history:
        session_history[session_id] = []
    
    # Add timestamp to the result
    result_with_timestamp = {
        **result,
        "timestamp": datetime.now().isoformat()
    }
    
    session_history[session_id].append(result_with_timestamp)
    
    # Keep only the last 50 entries per session
    if len(session_history[session_id]) > 50:
        session_history[session_id] = session_history[session_id][-50:]

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from an image using OCR."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        if not text.strip():
            return "No text detected in image"
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

def speech_to_text(audio_bytes: bytes, language: str = "en-US") -> str:
    """Convert speech to text from audio data."""
    try:
        # Create a BytesIO stream from the audio bytes
        audio_io = io.BytesIO(audio_bytes)
        
        # Use soundfile to read audio data and extract the sample rate
        data, sample_rate = sf.read(audio_io)
        
        # Write the audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            sf.write(temp_audio_path, data, sample_rate)
        
        # Use speech_recognition to transcribe the audio from the temporary file
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language)
        
        # Clean up the temporary file
        os.unlink(temp_audio_path)
        
        if not text.strip():
            return "No speech detected in audio"
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in speech recognition: {str(e)}")

def generate_visualization(sentiment_data: Dict[str, Any]) -> str:
    """Generate a base64 encoded visualization of sentiment data."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Sentiment score visualization
        plt.subplot(1, 2, 1)
        sentiment_score = sentiment_data["score"]
        sentiment_label = sentiment_data["label"]
        
        # Determine color based on sentiment
        if sentiment_label in ["POS", "positive", "5 stars", "4 stars"]:
            color = 'green'
        elif sentiment_label in ["NEU", "neutral", "3 stars"]:
            color = 'blue'
        else:
            color = 'red'
        
        plt.bar(['Sentiment'], [sentiment_score], color=color)
        plt.title(f"Sentiment: {sentiment_label}")
        plt.ylim(0, 1)
        
        # If emotions are present, show them too
        if "emotions" in sentiment_data and sentiment_data["emotions"]:
            plt.subplot(1, 2, 2)
            emotions = sentiment_data["emotions"]
            primary = emotions["primary_emotion"]
            confidence = emotions["confidence"]
            
            labels = [primary]
            values = [confidence]
            
            # Add secondary emotions
            for emotion in emotions["secondary_emotions"][:2]:  # Only show top 2 secondary emotions
                if isinstance(emotion, dict) and "emotion" in emotion and "score" in emotion:
                    labels.append(emotion["emotion"])
                    values.append(emotion["score"])
            
            plt.bar(labels, values, color=['purple', 'lightblue', 'lightgreen'][:len(labels)])
            plt.title("Detected Emotions")
            plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Convert plot to base64 encoded string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        return f"Visualization error: {str(e)}"

async def send_webhook(url: str, data: Dict[str, Any], secret: Optional[str] = None) -> bool:
    """Send data to a webhook URL."""
    headers = {"Content-Type": "application/json"}
    
    # Add signature if secret is provided
    if secret:
        payload = json.dumps(data)
        signature = hashlib.sha256(f"{payload}{secret}".encode()).hexdigest()
        headers["X-Webhook-Signature"] = signature
    
    try:
        async with asyncio.timeout(WEBHOOK_TIMEOUT):
            response = requests.post(url, json=data, headers=headers, timeout=WEBHOOK_TIMEOUT)
            return response.status_code >= 200 and response.status_code < 300
    except Exception:
        return False

def predict_sentiment(text: str, 
                      include_emotions: bool = False, 
                      detect_lang: bool = True,
                      generate_summary: bool = False) -> Dict[str, Any]:
    """Analyze sentiment of the given text with enhanced features."""
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key(text, include_emotions, detect_lang, generate_summary)
    if should_use_cache(cache_key):
        _, result = analysis_cache[cache_key]
        return result
    
    try:
        # Detect language if requested
        language = None
        if detect_lang:
            language = detect_language(text)
        else:
            language = "en"  # Default to English
        
        # Choose appropriate model based on language
        if language in sentiment_models:
            model = sentiment_models[language]
        else:
            model = sentiment_models["multilingual"]  # Fall back to multilingual model
        
        # Analyze sentiment
        sentiment_result = model(text)[0]
        label = sentiment_result["label"]
        score = float(sentiment_result["score"])
        
        # Build result object
        result = {
            "label": label,
            "score": score,
            "text": text,
            "explanation": explain_sentiment(label, score),
            "language": language,
            "word_count": len(text.split()),
            "processing_time": 0  # Will be updated at the end
        }
        
        # Add emotion analysis if requested
        if include_emotions:
            if language == "en":  # Emotion model only works for English
                result["emotions"] = analyze_emotions(text)
            else:
                result["emotions"] = {
                    "primary_emotion": "unavailable",
                    "confidence": 0,
                    "secondary_emotions": [],
                    "note": "Emotion detection only available for English text"
                }
        
        # Generate summary if requested and text is long enough
        if generate_summary and len(text.split()) >= 50:
            result["summary"] = generate_text_summary(text)
        
        # Add content moderation flags
        result["content_flags"] = moderate_content(text)
        
        # Update processing time
        result["processing_time"] = time.time() - start_time
        
        # Cache the result
        if ENABLE_CACHING:
            analysis_cache[cache_key] = (time.time(), result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """API root - returns a simple HTML interface."""
    return """
    <html>
        <head><title>MultiSense API</title></head>
        <body>
            <h1>MultiSense API</h1>
            <p>Advanced sentiment analysis with multi-modal input support.</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><code>/api/analyze/text</code> - Analyze text sentiment</li>
                <li><code>/api/analyze/batch</code> - Batch analyze multiple texts</li>
                <li><code>/api/analyze/image</code> - Extract text from image and analyze</li>
                <li><code>/api/analyze/audio</code> - Transcribe audio and analyze</li>
                <li><code>/api/visualize</code> - Generate visualization for sentiment results</li>
                <li><code>/api/history/{session_id}</code> - Get analysis history for a session</li>
                <li><code>/api/webhooks</code> - Register webhooks for async notifications</li>
                <li><code>/api/stats</code> - Get API usage statistics</li>
                <li><code>/health</code> - API health check</li>
            </ul>
        </body>
    </html>
    """

@app.post("/api/analyze/text", response_model=SentimentAnalysis)
async def analyze_text(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    x_webhook_id: Optional[str] = Header(None)
):
    """Analyze sentiment of the provided text."""
    result = predict_sentiment(
        request.text, 
        include_emotions=request.include_emotions,
        detect_lang=request.detect_language,
        generate_summary=request.generate_summary
    )
    
    # Update session history if session_id is provided
    if request.session_id:
        update_session_history(request.session_id, result)
    
    # Send webhook notification if webhook_id is provided
    if x_webhook_id and x_webhook_id in registered_webhooks:
        webhook = registered_webhooks[x_webhook_id]
        background_tasks.add_task(
            send_webhook, 
            webhook["url"], 
            {"event": "text_analysis", "result": result}, 
            webhook.get("secret")
        )
    
    return result

@app.post("/api/analyze/batch", response_model=List[SentimentAnalysis])
async def analyze_batch(
    request: BatchTextRequest,
    background_tasks: BackgroundTasks,
    x_webhook_id: Optional[str] = Header(None)
):
    """Analyze sentiment of multiple texts in a single request."""
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")
    
    results = []
    for text in request.texts:
        result = predict_sentiment(
            text,
            include_emotions=request.include_emotions,
            detect_lang=request.detect_language,
            generate_summary=request.generate_summary
        )
        results.append(result)
        
        # Update session history if session_id is provided
        if request.session_id:
            update_session_history(request.session_id, result)
    
    # Send webhook notification if webhook_id is provided
    if x_webhook_id and x_webhook_id in registered_webhooks:
        webhook = registered_webhooks[x_webhook_id]
        background_tasks.add_task(
            send_webhook, 
            webhook["url"], 
            {"event": "batch_analysis", "results": results}, 
            webhook.get("secret")
        )
    
    return results

@app.post("/api/analyze/image", response_model=SentimentAnalysis)
async def analyze_image(
    file: UploadFile = File(...),
    include_emotions: bool = Form(False),
    detect_language: bool = Form(True),
    generate_summary: bool = Form(False),
    session_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_webhook_id: Optional[str] = Header(None)
):
    """Extract text from an image and analyze its sentiment."""
    # Read the image file
    image_bytes = await file.read()
    
    # Extract text from the image
    extracted_text = extract_text_from_image(image_bytes)
    
    # Analyze the extracted text
    result = predict_sentiment(
        extracted_text,
        include_emotions=include_emotions,
        detect_lang=detect_language,
        generate_summary=generate_summary
    )
    
    # Add extracted text to the result
    result["extracted_text"] = extracted_text
    result["source_type"] = "image"
    result["filename"] = file.filename
    
    # Update session history if session_id is provided
    if session_id:
        update_session_history(session_id, result)
    
    # Send webhook notification if webhook_id is provided
    if x_webhook_id and x_webhook_id in registered_webhooks:
        webhook = registered_webhooks[x_webhook_id]
        background_tasks.add_task(
            send_webhook, 
            webhook["url"], 
            {"event": "image_analysis", "result": result}, 
            webhook.get("secret")
        )
    
    return result

@app.post("/api/analyze/audio", response_model=SentimentAnalysis)
async def analyze_audio(
    file: UploadFile = File(...),
    language: str = Form("en-US"),
    include_emotions: bool = Form(False),
    detect_language: bool = Form(False),  # Default to False for speech - we already know the language
    generate_summary: bool = Form(False),
    session_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_webhook_id: Optional[str] = Header(None)
):
    """Transcribe audio and analyze its sentiment."""
    try:
        # Read the audio file
        audio_bytes = await file.read()
        
        # Transcribe the audio
        transcribed_text = speech_to_text(audio_bytes, language)
        
        # Analyze the transcribed text
        result = predict_sentiment(
            transcribed_text,
            include_emotions=include_emotions,
            detect_lang=detect_language,
            generate_summary=generate_summary
        )
        
        # Add transcribed text to the result
        result["transcribed_text"] = transcribed_text
        result["source_type"] = "audio"
        result["filename"] = file.filename
        
        # Update session history if session_id is provided
        if session_id:
            update_session_history(session_id, result)
        
        # Send webhook notification if webhook_id is provided
        if x_webhook_id and x_webhook_id in registered_webhooks:
            webhook = registered_webhooks[x_webhook_id]
            background_tasks.add_task(
                send_webhook, 
                webhook["url"], 
                {"event": "audio_analysis", "result": result}, 
                webhook.get("secret")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

@app.post("/api/visualize")
async def visualize_sentiment(sentiment_data: Dict[str, Any]):
    """Generate a visualization of sentiment analysis results."""
    visualization = generate_visualization(sentiment_data)
    return {"visualization": visualization}

@app.get("/api/history/{session_id}")
async def get_session_history(session_id: str, limit: int = Query(10, ge=1, le=50)):
    """Get analysis history for a specific session."""
    if session_id not in session_history:
        return {"history": []}
    
    # Return the most recent entries up to the limit
    entries = session_history[session_id][-limit:]
    
    # Calculate sentiment trends
    if len(entries) > 1:
        labels = [entry.get("label") for entry in entries]
        scores = [entry.get("score", 0) for entry in entries]
        
        trend = {
            "sentiment_shift": scores[-1] - scores[0],
            "average_score": sum(scores) / len(scores),
            "most_common_sentiment": Counter(labels).most_common(1)[0][0]
        }
    else:
        trend = {"note": "At least 2 entries needed for trend analysis"}
    
    return {
        "session_id": session_id,
        "entry_count": len(session_history[session_id]),
        "trend": trend,
        "history": entries
    }

@app.post("/api/webhooks")
async def register_webhook(registration: WebhookRegistration):
    """Register a webhook for asynchronous notifications."""
    # Validate the URL
    try:
        parsed_url = urlparse(registration.callback_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise HTTPException(status_code=400, detail="Invalid URL format")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    # Generate webhook ID
    webhook_id = str(uuid.uuid4())
    
    # Store webhook registration
    registered_webhooks[webhook_id] = {
        "url": registration.callback_url,
        "secret": registration.secret_token,
        "events": registration.events,
        "created_at": datetime.now().isoformat()
    }
    
    return {
        "webhook_id": webhook_id,
        "registered_events": registration.events,
        "message": "Webhook registered successfully. Include the webhook_id in the X-Webhook-ID header in your requests."
    }

@app.delete("/api/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str):
    """Delete a registered webhook."""
    if webhook_id not in registered_webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    del registered_webhooks[webhook_id]
    return {"message": "Webhook deleted successfully"}

@app.get("/api/models")
async def list_models():
    """List available models for sentiment analysis."""
    return {
        "sentiment_models": list(sentiment_models.keys()),
        "features": {
            "emotion_detection": True,
            "multilingual_support": True,
            "content_moderation": True,
            "text_summarization": True
        }
    }

@app.get("/api/stats")
async def get_stats():
    """Get API usage statistics."""
    return {
        "active_sessions": len(session_history),
        "cache_entries": len(analysis_cache),
        "registered_webhooks": len(registered_webhooks),
        "pending_tasks": len(pending_tasks)
    }

@app.get("/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Advanced endpoints for topic analysis and document comparison
@app.post("/api/advanced/topic")
async def analyze_topics(request: AnalysisRequest):
    """Analyze main topics in a text."""
    text = request.text
    
    # Simple topic extraction using keyword frequency
    # In a production app, you'd use a more sophisticated NLP approach
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = {"this", "that", "with", "from", "have", "some", "they", "their", "there", "about"}
    
    # Remove stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count word frequency
    word_counts = Counter(filtered_words)
    
    # Get top keywords
    top_keywords = word_counts.most_common(5)
    
    return {
        "topics": [{"word": word, "count": count} for word, count in top_keywords],
        "word_count": len(text.split()),
        "summary": generate_text_summary(text) if len(text.split()) >= 50 else None
    }

@app.post("/api/advanced/compare")
async def compare_texts(texts: List[str] = Form(...)):
    """Compare sentiment between multiple texts."""
    if len(texts) < 2:
        raise HTTPException(status_code=400, detail="At least 2 texts required for comparison")
    
    if len(texts) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 texts allowed for comparison")
    
    results = []
    for text in texts:
        result = predict_sentiment(text)
        results.append(result)
    
    # Calculate the sentiment difference
    sentiment_range = max(r["score"] for r in results) - min(r["score"] for r in results)
    
    return {
        "results": results,
        "comparison": {
            "sentiment_range": sentiment_range,
            "most_positive": max(range(len(results)), key=lambda i: results[i]["score"]),
            "most_negative": min(range(len(results)), key=lambda i: results[i]["score"]),
            "average_sentiment": sum(r["score"] for r in results) / len(results)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
