import os
import urllib.parse
import yt_dlp

from pydub import AudioSegment
def is_url(source: str) -> bool:
    """
    Checks if the source string is a valid URL.
    """
    try:
        result = urllib.parse.urlparse(source)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def download_youtube_audio(url: str, output_dir: str) -> str:
    """
    Downloads audio from a YouTube URL and extracts it as a WAV file.
    Suppresses verbose terminal logging via yt-dlp's quiet option.
    Returns the absolute path to the downloaded file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define options for yt-dlp to download best audio and postprocess to wav
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        
        # Postprocessor converts to .wav, so adjust extension
        base, _ = os.path.splitext(filename)
        wav_path = base + ".wav"
        
        # Fallback verification in case of unexpected naming behavior
        if not os.path.exists(wav_path):
            video_id = info.get('id')
            for f in os.listdir(output_dir):
                if f.startswith(video_id) and f.endswith('.wav'):
                    wav_path = os.path.join(output_dir, f)
                    break
                    
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Failed to find downloaded and converted WAV file for video ID {info.get('id')}")
            
        return os.path.abspath(wav_path)

def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Converts any input audio/video format to mono, 16kHz WAV.
    Handles common formats like MP3, MP4, M4A, etc.
    Returns the absolute path to the converted WAV file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_normalized.wav"
        
    # Standardize path formatting
    output_path = os.path.abspath(output_path)
    
    # Load audio stream (handles video container format audio tracks as well if ffmpeg exists)
    audio = AudioSegment.from_file(input_path)
    
    # Normalize: set frame rate to 16000Hz (16kHz) and channel count to 1 (mono)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Export as WAV
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_length_sec: int = 600, output_dir: str = None) -> list[str]:
    """
    Splits a WAV file into smaller segments of configurable chunk_length_sec.
    Returns a list of absolute paths to the chunk files.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
        
    audio = AudioSegment.from_file(wav_path, format="wav")
    duration_sec = len(audio) / 1000.0
    
    # If the file is shorter than or equal to the target chunk length, return it directly
    if duration_sec <= chunk_length_sec:
        return [os.path.abspath(wav_path)]
        
    if output_dir is None:
        output_dir = os.path.dirname(wav_path)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    chunk_paths = []
    
    # Split audio based on chunk length in milliseconds
    chunk_length_ms = chunk_length_sec * 1000
    
    for i, start_ms in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start_ms : start_ms + chunk_length_ms]
        chunk_file_name = f"{base_name}_chunk_{i:03d}.wav"
        chunk_file_path = os.path.join(output_dir, chunk_file_name)
        
        chunk.export(chunk_file_path, format="wav")
        chunk_paths.append(os.path.abspath(chunk_file_path))
        
    return chunk_paths

def process_input(source: str, output_dir: str = None, chunk_length_sec: int = 600) -> list[str]:
    """
    Unified input handler that detects whether source is a URL or local file,
    routes it to download/convert as necessary, normalizes to mono 16kHz WAV,
    and returns a list of segmented chunks.
    """
    if output_dir is None:
        # Default to a 'downloads' directory relative to the current working directory
        output_dir = os.path.abspath(os.path.join(os.getcwd(), 'downloads'))
        
    os.makedirs(output_dir, exist_ok=True)
    
    if is_url(source):
        print(f"[AudioProcessor] URL detected. Downloading audio from: {source}")
        raw_audio_path = download_youtube_audio(source, output_dir)
        print(f"[AudioProcessor] Download completed: {raw_audio_path}")
        
        # Convert to 16kHz, mono WAV for final Whisper processing
        normalized_filename = os.path.splitext(os.path.basename(raw_audio_path))[0] + "_normalized.wav"
        normalized_path = os.path.join(output_dir, normalized_filename)
        
        print("[AudioProcessor] Normalizing audio to 16kHz mono WAV...")
        final_wav_path = convert_to_wav(raw_audio_path, normalized_path)
        
        # Clean up the raw downloaded file if it is different from the normalized path
        if raw_audio_path != final_wav_path and os.path.exists(raw_audio_path):
            try:
                os.remove(raw_audio_path)
            except Exception as e:
                print(f"[AudioProcessor] Warning: Could not remove temporary raw download: {e}")
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Local file does not exist: {source}")
            
        print(f"[AudioProcessor] Local file detected: {source}")
        normalized_filename = os.path.splitext(os.path.basename(source))[0] + "_normalized.wav"
        normalized_path = os.path.join(output_dir, normalized_filename)
        
        print("[AudioProcessor] Normalizing audio to 16kHz mono WAV...")
        final_wav_path = convert_to_wav(source, normalized_path)
        
    # Segment the normalized WAV audio
    print(f"[AudioProcessor] Splitting {final_wav_path} into {chunk_length_sec}-second chunks...")
    chunks = chunk_audio(final_wav_path, chunk_length_sec=chunk_length_sec, output_dir=output_dir)
    print(f"[AudioProcessor] Processing complete. Created {len(chunks)} chunk(s).")
    
    return chunks
