import os
import sys
from utils.audio_processor import process_input, is_url
from pydub import AudioSegment

def test_pipeline():
    # Target URL: Me at the zoo (19 seconds)
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    downloads_dir = os.path.abspath("downloads")
    
    print("--- Testing is_url Utility ---")
    assert is_url(test_url) is True, "Expected test_url to be recognized as a valid URL"
    assert is_url("not_a_url") is False, "Expected simple text to not be recognized as a URL"
    assert is_url("c:\\path\\to\\file") is False, "Expected path to not be recognized as a URL"
    print("is_url check passed!")
    
    print("\n--- Running process_input (Download + Normalize + Chunk) ---")
    # We set chunk length to 5 seconds to force chunking on the 19-second video
    chunk_len = 5
    try:
        chunks = process_input(test_url, output_dir=downloads_dir, chunk_length_sec=chunk_len)
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please verify that FFmpeg is installed and added to your system's PATH.")
        sys.exit(1)
        
    print(f"\nExecution successful! Created {len(chunks)} chunks:")
    for path in chunks:
        print(f" - {path}")
        
    # Verification assertions
    assert len(chunks) > 0, "No chunks were generated!"
    
    # Check properties of the first chunk
    first_chunk_path = chunks[0]
    assert os.path.exists(first_chunk_path), f"Chunk file does not exist: {first_chunk_path}"
    
    audio = AudioSegment.from_file(first_chunk_path)
    print(f"\n--- Verifying First Chunk Properties ---")
    print(f"Format: WAV")
    print(f"Channels: {audio.channels} (Expected: 1 for mono)")
    print(f"Frame Rate: {audio.frame_rate} Hz (Expected: 16000 Hz)")
    print(f"Duration: {len(audio)/1000.0} seconds (Expected: <= {chunk_len} seconds)")
    
    assert audio.channels == 1, f"Expected 1 channel, got {audio.channels}"
    assert audio.frame_rate == 16000, f"Expected 16000 frame rate, got {audio.frame_rate}"
    assert len(audio) <= chunk_len * 1000 + 100, f"Expected chunk <= {chunk_len}s, got {len(audio)/1000.0}s"
    
    print("\nAll pipeline assertions passed successfully!")

if __name__ == "__main__":
    test_pipeline()
