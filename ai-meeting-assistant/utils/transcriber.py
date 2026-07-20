import os
import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()

def transcribe_whisper(audio_path: str, model_name: str = "base") -> dict:
    """
    Transcribes a local WAV file using a local Whisper model.
    Returns a dictionary containing 'text' and a list of 'segments' with timestamps.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    print(f"[Whisper] Loading local model '{model_name}'...")
    model = whisper.load_model(model_name)
    
    print(f"[Whisper] Transcribing file: {audio_path}...")
    result = model.transcribe(audio_path)
    
    # Standardize result structure
    text = result.get("text", "").strip()
    segments = []
    
    for s in result.get("segments", []):
        segments.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": s.get("text", "").strip()
        })
        
    return {
        "text": text,
        "segments": segments
    }

def transcribe_sarvam(audio_path: str, api_key: str = None, language_code: str = "hi-IN") -> dict:
    """
    Transcribes a local WAV file using Sarvam AI's speech-to-text API (optimized for Hindi/Hinglish).
    Returns a dictionary containing 'text' and a list of 'segments' with timestamps.
    """
    # Grab API key from argument or environment
    sarvam_key = api_key or os.getenv("SARVAM_API_KEY")
    if not sarvam_key:
        raise ValueError("Sarvam AI API key is missing. Please set the SARVAM_API_KEY environment variable or pass it to the function.")
        
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": sarvam_key
    }
    
    print(f"[SarvamAI] Sending request to transcribe {audio_path} (Language: {language_code})...")
    
    with open(audio_path, "rb") as f:
        files = {
            "file": (os.path.basename(audio_path), f, "audio/wav")
        }
        data = {
            "model": "saaras:v3",
            "language_code": language_code,
            "mode": "transcribe"
        }
        
        response = requests.post(url, headers=headers, files=files, data=data)
        
    if response.status_code != 200:
        raise RuntimeError(f"[SarvamAI] API request failed (Status Code {response.status_code}): {response.text}")
        
    result = response.json()
    transcript = result.get("transcript", "").strip()
    raw_segments = result.get("segments", [])
    
    segments = []
    for s in raw_segments:
        segments.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": s.get("text", "").strip()
        })
        
    # If API does not provide segments but returns transcript, treat whole chunk as a single segment
    if not segments and transcript:
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_sec = len(audio) / 1000.0
        except Exception:
            duration_sec = 25.0  # default fallback chunk size
            
        segments.append({
            "start": 0.0,
            "end": duration_sec,
            "text": transcript
        })
        
    return {
        "text": transcript,
        "segments": segments
    }

def transcribe(
    audio_path: str,
    backend: str = "whisper",
    api_key: str = None,
    language_code: str = "hi-IN",
    whisper_model_name: str = "base"
) -> dict:
    """
    Unified transcriber entry point that routes the request to either local Whisper or Sarvam AI.
    """
    backend_lower = backend.lower()
    if backend_lower == "whisper":
        return transcribe_whisper(audio_path, model_name=whisper_model_name)
    elif backend_lower == "sarvam":
        return transcribe_sarvam(audio_path, api_key=api_key, language_code=language_code)
    else:
        raise ValueError(f"Unsupported transcription backend: {backend}. Use 'whisper' or 'sarvam'.")
