import os
import sys
import argparse
from dotenv import load_dotenv

# Import utilities
from utils.audio_processor import process_input
from utils.transcriber import transcribe
from utils.summarise import generate_summary
from utils.extractor import extract_meeting_elements
from utils.history import save_meeting, get_meeting
from utils.vector_store import index_transcript
from utils.rag_engine import query_meeting

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="AI Meeting Assistant E2E Integration Pipeline Test")
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=jNQXAC9IVRw",
                        help="YouTube URL to process (default is 'Me at the zoo')")
    parser.add_argument("--backend", type=str, default="whisper", choices=["whisper", "sarvam"],
                        help="Transcription backend: 'whisper' or 'sarvam'")
    parser.add_argument("--whisper-model", type=str, default="tiny",
                        help="Local Whisper model size (default: tiny for speed)")
    parser.add_argument("--question", type=str, default="What animal is being shown in the video?",
                        help="Question to query the RAG engine (default: 'What animal is being shown in the video?')")
    
    args = parser.parse_args()
    
    print("\n==============================================")
    # Validate API keys if summarization/extraction/RAG are active
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        print("[Error] MISTRAL_API_KEY is not configured in your .env file!")
        print("Please configure your .env file with valid credentials before running tests.")
        sys.exit(1)
        
    print("Starting E2E Integration Test Pipeline...")
    print(f"Target Source: {args.url}")
    print(f"Backend: {args.backend.upper()} (Model: {args.whisper_model})")
    print(f"Test Question: '{args.question}'")
    print("==============================================\n")
    
    downloads_dir = os.path.abspath("downloads")
    chunk_len_sec = 25 if args.backend == "sarvam" else 600
    
    # 1. Ingest and split audio
    print("[E2E Test] Step 1: Processing input source (Download + Normalize + Chunk)...")
    try:
        chunks = process_input(args.url, output_dir=downloads_dir, chunk_length_sec=chunk_len_sec)
        print(f" -> Generated {len(chunks)} audio chunk(s).")
    except Exception as e:
        print(f"[E2E Test] Failed Step 1 (Ingestion): {e}")
        sys.exit(1)
        
    # 2. Transcribe chunks
    print("\n[E2E Test] Step 2: Transcribing audio segments...")
    full_transcript_text = []
    aggregated_segments = []
    
    for i, chunk_path in enumerate(chunks):
        print(f" -> Transcribing chunk {i+1}/{len(chunks)}: {os.path.basename(chunk_path)}...")
        time_offset = i * chunk_len_sec
        try:
            result = transcribe(
                audio_path=chunk_path,
                backend=args.backend,
                whisper_model_name=args.whisper_model
            )
            text = result.get("text", "").strip()
            if text:
                full_transcript_text.append(text)
                
            for seg in result.get("segments", []):
                aggregated_segments.append({
                    "start": seg["start"] + time_offset,
                    "end": seg["end"] + time_offset,
                    "text": seg["text"].strip()
                })
        except Exception as e:
            print(f"[E2E Test] Failed Step 2 (Transcription): {e}")
            sys.exit(1)
            
    combined_transcript = " ".join(full_transcript_text).strip()
    print(f" -> Merged Transcript length: {len(combined_transcript)} characters.")
    print(f" -> Merged Segments count: {len(aggregated_segments)}.")
    
    if not combined_transcript:
        print("[E2E Test] Error: Merged transcript is empty. Skipping further steps.")
        sys.exit(1)
        
    # 3. Generate Summary
    print("\n[E2E Test] Step 3: Generating meeting summary via Mistral AI...")
    try:
        summary = generate_summary(combined_transcript)
        print(" -> Summary generated successfully.")
    except Exception as e:
        print(f"[E2E Test] Failed Step 3 (Summarization): {e}")
        sys.exit(1)
        
    # 4. Extract Action Items, Decisions, Questions
    print("\n[E2E Test] Step 4: Extracting structured action items, decisions, and open questions...")
    try:
        analysis = extract_meeting_elements(combined_transcript)
        print(f" -> Extracted {len(analysis.get('action_items', []))} action items.")
        print(f" -> Extracted {len(analysis.get('decisions', []))} decisions.")
        print(f" -> Extracted {len(analysis.get('open_questions', []))} open questions.")
    except Exception as e:
        print(f"[E2E Test] Failed Step 4 (Extraction): {e}")
        sys.exit(1)
        
    # 5. Persist to SQLite Database
    print("\n[E2E Test] Step 5: Storing meeting records in local database (meetings.db)...")
    try:
        meeting_id = save_meeting(
            title="E2E Integration Test Meeting",
            source=args.url,
            backend=args.backend,
            duration=len(chunks) * chunk_len_sec, # rough duration fallback
            transcript_text=combined_transcript,
            summary_markdown=summary,
            segments=aggregated_segments,
            analysis=analysis
        )
        print(f" -> Saved meeting in database with row ID: {meeting_id}")
    except Exception as e:
        print(f"[E2E Test] Failed Step 5 (Database Save): {e}")
        sys.exit(1)
        
    # 6. Index in ChromaDB Vector Store
    print("\n[E2E Test] Step 6: Chunking and indexing transcript in ChromaDB...")
    try:
        index_transcript(meeting_id, combined_transcript)
        print(" -> Chroma indexing completed successfully.")
    except Exception as e:
        print(f"[E2E Test] Failed Step 6 (Chroma Indexing): {e}")
        sys.exit(1)
        
    # 7. Query RAG Q&A Engine
    print(f"\n[E2E Test] Step 7: Querying RAG engine: '{args.question}'...")
    try:
        answer = query_meeting(meeting_id, args.question)
        print("\n" + "="*30)
        print(f"Q: {args.question}")
        print(f"A: {answer}")
        print("="*30 + "\n")
    except Exception as e:
        print(f"[E2E Test] Failed Step 7 (RAG query): {e}")
        sys.exit(1)
        
    # Verify retrieval
    print("[E2E Test] Verifying database integrity fetch...")
    meeting_verify = get_meeting(meeting_id)
    assert meeting_verify is not None, "Failed to retrieve saved meeting from DB!"
    assert meeting_verify["title"] == "E2E Integration Test Meeting", "Title mismatch on fetch!"
    assert len(meeting_verify["analysis"].get("action_items", [])) == len(analysis["action_items"]), "Action items mismatch!"
    
    print("\nE2E INTEGRATION PIPELINE TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
