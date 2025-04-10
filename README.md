# Sentiment Analysis Backend Service

A FastAPI-based backend service that performs sentiment analysis on text input from multiple sources: direct text, images (OCR), and audio (speech-to-text).

## Features

- 🎯 Sentiment Analysis using BERTweet model
- 📷 OCR (Optical Character Recognition) for extracting text from images
- 🎤 Speech-to-Text conversion for audio files
- 🚀 Fast and efficient processing
- 🔄 CORS support for cross-origin requests

## Prerequisites

- Python 3.8+
- Tesseract OCR installed on your system
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shruti1238/backend-sentiment.git
cd backend-sentiment
```

2. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Update the Tesseract path in `app.py` if necessary:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust path as needed
```

## Usage

1. Start the server:
```bash
python app.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### 1. Text Analysis
```http
POST /submit-text/
Content-Type: application/x-www-form-urlencoded

text=Your text here
```

#### 2. Image Analysis
```http
POST /submit-image/
Content-Type: multipart/form-data

file=@your_image.jpg
```

#### 3. Audio Analysis
```http
POST /submit-audio/
Content-Type: multipart/form-data

file=@your_audio.wav
```

### Example Response

```json
{
    "label": "POS",
    "score": 0.9876543,
    "text": "I love this product!",
    "extracted_text": "Sample extracted text",  // For image uploads
    "transcribed_text": "Sample transcribed text"  // For audio uploads
}
```

## Dependencies

- FastAPI - Web framework
- Transformers - Sentiment analysis model
- Pytesseract - OCR processing
- SpeechRecognition - Audio transcription
- Pillow - Image processing
- soundfile - Audio file handling

## Error Handling

The API includes comprehensive error handling for:
- Invalid file formats
- Processing errors
- Empty or unreadable inputs
- Server-side errors

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [BERTweet](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) for the sentiment analysis model
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction
- [Google Speech Recognition](https://cloud.google.com/speech-to-text) for audio transcription
