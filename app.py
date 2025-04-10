from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import pytesseract
from PIL import Image
import io
import speech_recognition as sr
import tempfile
import os
import soundfile as sf  # New import for handling audio files

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update as per your system
# Initialize sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

def predict_sentiment(text: str) -> dict:
    try:
        result = sentiment_analysis(text)[0]
        return {
            "label": result["label"],
            "score": float(result["score"]),
            "text": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        if not text.strip():
            return "No text detected in image"
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

def speech_to_text(audio_bytes: bytes) -> str:
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
            text = recognizer.recognize_google(audio_data)
        
        # Clean up the temporary file
        os.unlink(temp_audio_path)
        
        if not text.strip():
            return "No speech detected in audio"
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in speech recognition: {str(e)}")

@app.post("/submit-text/")
async def submit_text(text: str = Form(...)):
    # Process the text directly
    result = predict_sentiment(text)
    return result

@app.post("/submit-image/")
async def submit_image(file: UploadFile = File(...)):
    # Read the file content
    image_bytes = await file.read()
    
    # Extract text from the image
    extracted_text = extract_text_from_image(image_bytes)
    
    # Predict sentiment from the extracted text
    result = predict_sentiment(extracted_text)
    result["extracted_text"] = extracted_text
    
    return result

@app.post("/submit-audio/")
async def submit_audio(file: UploadFile = File(...)):
    try:
        # Read the uploaded file content directly as bytes
        audio_bytes = await file.read()
        
        # Use the new speech_to_text function to transcribe the audio
        transcribed_text = speech_to_text(audio_bytes)
        
        # Predict sentiment from the transcribed text
        result = predict_sentiment(transcribed_text)
        result["transcribed_text"] = transcribed_text
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
