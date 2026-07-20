import os
import argparse
import sys
from dotenv import load_dotenv
from utils.audio_processor import process_input
from utils.transcriber import transcribe
from utils.summarise import generate_summary

# Load environment variables
load_dotenv()

def format_timestamp(seconds: float) -> str:
    """
    Formats a float duration in seconds into a MM:SS string.
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def run_integration_pipeline():
    parser = argparse.ArgumentParser(description="AI Meeting Assistant Integration Test Runner")
    parser.add_argument("--url", type=str, help="YouTube URL to download and transcribe")
    parser.add_argument("--file", type=str, help="Path to a local audio/video file")
    parser.add_argument("--backend", type=str, choices=["whisper", "sarvam"], default="whisper",
                        help="Transcription backend: 'whisper' (local) or 'sarvam' (Hindi/Hinglish API)")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size to load locally (e.g., tiny, base, small)")
    parser.add_argument("--chunk-sec", type=int, default=None,
                        help="Override chunk duration in seconds (defaults: 600s for Whisper, 25s for Sarvam)")
    parser.add_argument("--language-code", type=str, default="hi-IN",
                        help="Language code for Sarvam AI backend (default: hi-IN)")
    parser.add_argument("--summarize", action="store_true",
                        help="Flag to enable LangChain and Mistral AI summarization")
    parser.add_argument("--mistral-model", type=str, default="open-mistral-7b",
                        help="Mistral AI model to use for summarization (default: open-mistral-7b)")
    
    args = parser.parse_args()
    
    # Verify inputs
    if not args.url and not args.file:
        print("Error: You must provide either a --url or a --file input.")
        parser.print_help()
        sys.exit(1)
        
    source = args.url if args.url else args.file
    
    # 1. Determine chunk duration dynamically
    if args.chunk_sec is not None:
        chunk_length_sec = args.chunk_sec
    else:
        # Sarvam AI has a 30s limit on the short REST API, so we use 25s chunks.
        # Whisper can handle much larger chunks (e.g., 10 minutes / 600s).
        chunk_length_sec = 25 if args.backend == "sarvam" else 600
        
    print(f"\n=== Configuration ===")
    print(f"Source: {source}")
    print(f"Backend: {args.backend.upper()}")
    if args.backend == "whisper":
        print(f"Whisper Model: {args.whisper_model}")
    else:
        print(f"Sarvam Language: {args.language_code}")
    print(f"Chunk Duration: {chunk_length_sec} seconds")
    print(f"Run Summarizer: {args.summarize}")
    print(f"=====================\n")
    
    # 2. Run ingestion pipeline (download/convert to WAV and chunk)
    downloads_dir = os.path.abspath("downloads")
    try:
        print("[Pipeline] Ingesting and chunking source input...")
        chunks = process_input(source, output_dir=downloads_dir, chunk_length_sec=chunk_length_sec)
        print(f"[Pipeline] Successfully generated {len(chunks)} audio chunk(s).")
    except Exception as e:
        print(f"[Pipeline] Ingestion failed: {e}")
        sys.exit(1)
        
    # 3. Transcribe each chunk and aggregate timestamps
    full_transcript_text = []
    aggregated_segments = []
    
    for i, chunk_path in enumerate(chunks):
        print(f"\n[Pipeline] Processing chunk {i+1}/{len(chunks)}: {os.path.basename(chunk_path)}")
        
        # Calculate time offset for this chunk
        time_offset = i * chunk_length_sec
        
        try:
            # Call unified transcriber
            chunk_result = transcribe(
                audio_path=chunk_path,
                backend=args.backend,
                language_code=args.language_code,
                whisper_model_name=args.whisper_model
            )
            
            # Aggregate text
            chunk_text = chunk_result.get("text", "").strip()
            if chunk_text:
                full_transcript_text.append(chunk_text)
                
            # Aggregate segments, offsetting their timestamps
            for seg in chunk_result.get("segments", []):
                aggregated_segments.append({
                    "start": seg["start"] + time_offset,
                    "end": seg["end"] + time_offset,
                    "text": seg["text"].strip()
                })
        except Exception as e:
            print(f"[Pipeline] Failed to transcribe chunk {i+1}: {e}")
            sys.exit(1)
            
    # Combine full text
    combined_transcript = " ".join(full_transcript_text).strip()
    
    # Print transcript with aggregated timestamps
    print("\n" + "="*30)
    print("      AGGREGATED TRANSCRIPT")
    print("="*30)
    if not aggregated_segments:
        print("[No segments transcribed]")
    else:
        for seg in aggregated_segments:
            start_str = format_timestamp(seg["start"])
            end_str = format_timestamp(seg["end"])
            print(f"[{start_str} - {end_str}] {seg['text']}")
    print("="*30 + "\n")
    
    # 4. Generate structured summary if enabled
    if args.summarize:
        print("[Pipeline] Generating meeting summary via LangChain + Mistral AI...")
        try:
            summary = generate_summary(
                transcript=combined_transcript,
                model_name=args.mistral_model
            )
            print("\n" + "="*30)
            print("        MEETING SUMMARY")
            print("="*30)
            print(summary)
            print("="*30 + "\n")
        except Exception as e:
            print(f"[Pipeline] Summarization failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_integration_pipeline()
