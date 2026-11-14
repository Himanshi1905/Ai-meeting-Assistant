import os
import sys
import json
import time
import subprocess
from dotenv import load_dotenv
from pydub import AudioSegment
import streamlit as st
# Install Deno for yt-dlp JavaScript support
DENO_BIN = os.path.expanduser("~/.deno/bin")
os.environ["PATH"] = DENO_BIN + os.pathsep + os.environ.get("PATH", "")

if not os.path.exists(os.path.join(DENO_BIN, "deno")):
    subprocess.run(
        "curl -fsSL https://deno.land/install.sh | sh",
        shell=True,
        check=True
    )
# Install BgUtils PO Token Provider for YouTube
BGUTIL_DIR = os.path.expanduser("~/bgutil-ytdlp-pot-provider")
BGUTIL_SERVER = os.path.join(BGUTIL_DIR, "server")

if not os.path.isdir(BGUTIL_DIR):
    subprocess.run(
        [
            "git", "clone",
            "--depth", "1",
            "--branch", "2.0.2",
            "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git",
            BGUTIL_DIR
        ],
        check=True
    )

if not os.path.isdir(os.path.join(BGUTIL_SERVER, "node_modules")):
    subprocess.run(
        ["deno", "install", "--allow-scripts=npm:canvas", "--frozen"],
        cwd=BGUTIL_SERVER,
        check=True
    )
# Import utilities
from utils.audio_processor import process_input, is_url
from utils.transcriber import transcribe
from utils.summarise import generate_summary
from utils.history import save_meeting, list_meetings, get_meeting, delete_meeting, init_db
from utils.extractor import extract_meeting_elements
from utils.vector_store import index_transcript
from utils.rag_engine import query_meeting
from utils.exporter import generate_pdf_report

# Load environment variables
load_dotenv()

# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database Schema
init_db()

# --- Helper Functions ---
def format_timestamp(seconds: float) -> str:
    """
    Formats a duration in seconds into a MM:SS string.
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def get_youtube_title(url: str) -> str:
    """
    Fetch the title of a YouTube video using yt-dlp.
    """
    try:
        import yt_dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('title', 'YouTube Video')
    except Exception:
        return "YouTube Video"

# --- Inject Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Font style override */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Stylings */
    .app-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Containers */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Timeline segment visual block */
    .segment-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #FF4B4B;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .segment-box:hover {
        transform: translateX(4px);
        background: rgba(255, 255, 255, 0.05);
    }
    .segment-time {
        font-weight: 600;
        color: #FF4B4B;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .segment-text {
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    /* Custom lists styled for Action Items and Decisions */
    .action-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .action-desc {
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .action-meta {
        font-size: 0.9rem;
        color: #888888;
    }
    
    /* Status Badge styling */
    .status-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-active {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
    }
    .status-missing {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "ingest"
if "selected_meeting_id" not in st.session_state:
    st.session_state.selected_meeting_id = None

# --- SIDEBAR COMPONENT ---
with st.sidebar:
    st.markdown("### 🎙️ AI Meeting Assistant")
    st.markdown("---")
    
    # 1. API Configuration Status
    st.markdown("#### API Key Status")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")
    
    if mistral_key:
        st.markdown('<span class="status-badge status-active">Mistral AI: Configured</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-missing">Mistral AI: Key Missing</span>', unsafe_allow_html=True)
        
    if sarvam_key:
        st.markdown('<span class="status-badge status-active">Sarvam AI: Configured</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-missing">Sarvam AI: Key Missing</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Meeting History
    st.markdown("#### 📁 Past Meetings")
    history = list_meetings()
    
    if not history:
        st.info("No meetings saved yet.")
    else:
        for item in history:
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                display_title = item["title"]
                if len(display_title) > 28:
                    display_title = display_title[:25] + "..."
                    
                is_selected = st.session_state.selected_meeting_id == item["id"]
                btn_type = "primary" if is_selected else "secondary"
                
                if st.button(
                    f"📄 {display_title}", 
                    key=f"hist_app_{item['id']}", 
                    use_container_width=True, 
                    type=btn_type
                ):
                    st.session_state.selected_meeting_id = item["id"]
                    st.session_state.current_page = "view"
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_app_{item['id']}", help="Delete meeting from database"):
                    delete_meeting(item["id"])
                    if st.session_state.selected_meeting_id == item["id"]:
                        st.session_state.selected_meeting_id = None
                        st.session_state.current_page = "ingest"
                    st.success("Deleted!")
                    time.sleep(0.5)
                    st.rerun()

# --- PAGE 1: INGESTION FORM ---
if st.session_state.current_page == "ingest":
    st.markdown('<div class="app-title">Ingest & Transcribe</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Upload local audio/video or paste a YouTube URL to transcribe and summarize.</div>', unsafe_allow_html=True)
    
    source_type = st.radio(
        "Select Input Source",
        options=["YouTube Video URL", "Local Audio/Video File"],
        horizontal=True,
        key="app_source_type"
    )
    
    input_source = None
    default_title = ""
    
    if source_type == "YouTube Video URL":
        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="e.g., https://www.youtube.com/watch?v=jNQXAC9IVRw",
            help="Provide a public YouTube watch link to download and transcribe.",
            key="app_yt_url"
        )
        if youtube_url.strip():
            input_source = youtube_url.strip()
            
    else:
        uploaded_file = st.file_uploader(
            "Upload Audio or Video File",
            type=["wav", "mp3", "m4a", "mp4", "ogg", "flac"],
            help="Upload an audio or video container. The tool normalizes the stream to 16kHz WAV automatically.",
            key="app_file_uploader"
        )
        if uploaded_file is not None:
            downloads_dir = os.path.abspath("downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            temp_path = os.path.join(downloads_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            input_source = temp_path
            default_title = os.path.splitext(uploaded_file.name)[0]
            
    # Advanced Configuration Expander
    with st.expander("🛠️ Advanced Settings", expanded=False):
        custom_title = st.text_input(
            "Custom Meeting Title (Optional)",
            value=default_title,
            placeholder="e.g., Q3 Planning Session",
            key="app_custom_title"
        )
        
        col_backend, col_chunk = st.columns(2)
        with col_backend:
            backend = st.selectbox(
                "Transcription Backend",
                options=["whisper", "sarvam"],
                format_func=lambda x: "Local Whisper (OpenAI)" if x == "whisper" else "Sarvam AI REST API (Hindi/Hinglish)",
                help="Choose Whisper for local offline transcription or Sarvam AI for Hindi/Hinglish speech.",
                key="app_backend"
            )
        with col_chunk:
            override_chunk = st.number_input(
                "Chunk Size (Seconds) - Leave 0 for Auto",
                min_value=0,
                value=0,
                step=5,
                help="Sarvam AI requires short chunks (< 30s). Whisper defaults to 600s.",
                key="app_override_chunk"
            )
            
        col_whisper, col_sarvam = st.columns(2)
        with col_whisper:
            whisper_model = st.selectbox(
                "Whisper Model (Local)",
                options=["tiny", "base", "small", "medium", "large"],
                index=1,
                disabled=(backend != "whisper"),
                help="Larger models are more accurate but take longer and use more memory/CPU.",
                key="app_whisper_model"
            )
        with col_sarvam:
            language_code = st.text_input(
                "Sarvam Language Code",
                value="hi-IN",
                disabled=(backend != "sarvam"),
                help="Defaults to hi-IN. Use hinglish-colloquial for mixed speech.",
                key="app_lang_code"
            )
            
        st.markdown("---")
        col_sum_toggle, col_mistral = st.columns([1, 2])
        with col_sum_toggle:
            enable_summarize = st.checkbox("Generate Meeting Summary & Action Items", value=True, key="app_enable_sum")
        with col_mistral:
            mistral_model = st.selectbox(
                "Mistral AI Model",
                options=["open-mistral-7b", "mistral-tiny", "mistral-small", "mistral-medium"],
                index=0,
                disabled=not enable_summarize,
                key="app_mistral_model"
            )

    # Process Action
    if st.button("🚀 Process Meeting", type="primary", use_container_width=True, key="app_btn_process"):
        if not input_source:
            st.error("Please enter a valid YouTube URL or upload a file.")
        elif backend == "sarvam" and not sarvam_key:
            st.error("Sarvam AI key is missing. Add it to the .env file.")
        elif enable_summarize and not mistral_key:
            st.error("Mistral AI key is missing. Add it to the .env file to enable summarization.")
        else:
            meeting_title = custom_title.strip()
            if not meeting_title:
                if source_type == "YouTube Video URL":
                    with st.spinner("Fetching YouTube video details..."):
                        meeting_title = get_youtube_title(input_source)
                else:
                    meeting_title = default_title
                    
            if not meeting_title:
                meeting_title = "Untitled Meeting"
                
            if override_chunk > 0:
                chunk_len_sec = override_chunk
            else:
                chunk_len_sec = 25 if backend == "sarvam" else 600
                
            downloads_dir = os.path.abspath("downloads")
            
            try:
                status_container = st.empty()
                progress_bar = st.progress(0)
                
                # Phase 1: Ingest and Chunk
                status_container.info("🔄 Ingesting source audio/video. This includes downloading and converting to WAV...")
                progress_bar.progress(10)
                
                chunks = process_input(input_source, output_dir=downloads_dir, chunk_length_sec=chunk_len_sec)
                
                if not chunks:
                    raise ValueError("No audio segments generated.")
                
                progress_bar.progress(30)
                
                # Calculate total duration
                total_duration = 0.0
                for path in chunks:
                    try:
                        audio = AudioSegment.from_file(path)
                        total_duration += len(audio) / 1000.0
                    except Exception:
                        pass
                
                # Phase 2: Transcribe
                full_transcript_text = []
                aggregated_segments = []
                
                for i, chunk_path in enumerate(chunks):
                    status_container.info(f"🎙️ Transcribing chunk {i+1} of {len(chunks)} ({backend.upper()})...")
                    transcribe_progress = int(30 + (40 * (i / len(chunks))))
                    progress_bar.progress(transcribe_progress)
                    
                    time_offset = i * chunk_len_sec
                    
                    chunk_result = transcribe(
                        audio_path=chunk_path,
                        backend=backend,
                        language_code=language_code,
                        whisper_model_name=whisper_model
                    )
                    
                    chunk_text = chunk_result.get("text", "").strip()
                    if chunk_text:
                        full_transcript_text.append(chunk_text)
                        
                    for seg in chunk_result.get("segments", []):
                        aggregated_segments.append({
                            "start": seg["start"] + time_offset,
                            "end": seg["end"] + time_offset,
                            "text": seg["text"].strip()
                        })
                
                combined_transcript = " ".join(full_transcript_text).strip()
                progress_bar.progress(70)
                
                # Phase 3: Summarize & Extract Action Items
                summary = ""
                analysis = {
                    "action_items": [],
                    "decisions": [],
                    "open_questions": []
                }
                
                if enable_summarize and combined_transcript:
                    status_container.info("📝 Generating structured summary using Mistral AI...")
                    summary = generate_summary(combined_transcript, model_name=mistral_model)
                    progress_bar.progress(80)
                    
                    status_container.info("🎯 Extracting action items, decisions, and open questions...")
                    analysis = extract_meeting_elements(combined_transcript, api_key=mistral_key, model_name=mistral_model)
                elif not combined_transcript:
                    summary = "No transcription content available to summarize."
                else:
                    summary = "Summary generation disabled."
                    
                progress_bar.progress(90)
                
                # Phase 4: Save to SQLite Database
                status_container.info("💾 Saving transcript and summary to SQLite database...")
                meeting_id = save_meeting(
                    title=meeting_title,
                    source=input_source if is_url(input_source) else os.path.basename(input_source),
                    backend=backend,
                    duration=total_duration,
                    transcript_text=combined_transcript,
                    summary_markdown=summary,
                    segments=aggregated_segments,
                    analysis=analysis
                )
                
                # Phase 5: Index in ChromaDB
                status_container.info("🧠 Indexing transcript chunks into ChromaDB for Q&A...")
                index_transcript(meeting_id, combined_transcript)
                
                progress_bar.progress(100)
                status_container.success("✅ Meeting pipeline completed successfully!")
                
                # Update selected meeting and view page
                st.session_state.selected_meeting_id = meeting_id
                st.session_state.current_page = "view"
                time.sleep(1.0)
                st.rerun()
                
            except Exception as e:
                st.error(f"Pipeline error during execution: {e}")

# --- PAGE 2: VIEW RESULTS ---
elif st.session_state.current_page == "view":
    meeting = get_meeting(st.session_state.selected_meeting_id)
    
    if not meeting:
        st.error("Selected meeting record not found.")
        if st.button("⬅️ Back to Ingest", key="app_back_err"):
            st.session_state.current_page = "ingest"
            st.session_state.selected_meeting_id = None
            st.rerun()
    else:
        if st.button("⬅️ Ingest a New Meeting", key="app_btn_ingest_new"):
            st.session_state.current_page = "ingest"
            st.session_state.selected_meeting_id = None
            st.rerun()
            
        st.markdown(f'<div class="app-title">{meeting["title"]}</div>', unsafe_allow_html=True)
        
        # Meta Card
        duration_formatted = format_timestamp(meeting["duration"])
        st.markdown(f"""
            <div class="metric-card">
                <strong>Source:</strong> <code>{meeting["source"]}</code> &nbsp;|&nbsp; 
                <strong>Backend:</strong> <code>{meeting["backend"].upper()}</code> &nbsp;|&nbsp; 
                <strong>Duration:</strong> <code>{duration_formatted}</code> &nbsp;|&nbsp; 
                <strong>Ingested On:</strong> <code>{meeting["created_at"]}</code>
            </div>
        """, unsafe_allow_html=True)
        
        # Tabs including new Action Items & Decisions tab, and Chat QA tab
        sum_tab, action_tab, chat_tab, time_tab, raw_tab = st.tabs([
            "📝 Structured Summary", 
            "🎯 Action Items & Decisions",
            "💬 Ask Questions (RAG)",
            "🕒 Interactive Timeline", 
            "📜 Full Transcript"
        ])
        
        with sum_tab:
            st.markdown(meeting["summary_markdown"] if meeting["summary_markdown"] else "*No summary generated.*")
            
        with action_tab:
            analysis = meeting.get("analysis", {})
            action_items = analysis.get("action_items", [])
            decisions = analysis.get("decisions", [])
            open_questions = analysis.get("open_questions", [])
            
            st.markdown("### 🎯 Action Items")
            if not action_items:
                st.info("No action items extracted.")
            else:
                for item in action_items:
                    owner_text = f"👤 **Owner:** {item.get('owner')}" if item.get("owner") else "👤 **Owner:** Unassigned"
                    deadline_text = f"📅 **Deadline:** {item.get('deadline')}" if item.get("deadline") else "📅 **Deadline:** N/A"
                    
                    st.markdown(f"""
                        <div class="action-card">
                            <div class="action-desc">☑️ {item.get('description')}</div>
                            <div class="action-meta">{owner_text} &nbsp;|&nbsp; {deadline_text}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("---")
            st.markdown("### 💡 Key Decisions")
            if not decisions:
                st.info("No key decisions recorded.")
            else:
                for dec in decisions:
                    st.markdown(f"**- {dec.get('description')}**")
                    if dec.get("context"):
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Context: {dec.get('context')}*")
                        
            st.markdown("---")
            st.markdown("### ❓ Open Questions & Blockers")
            if not open_questions:
                st.info("No open questions or blockers recorded.")
            else:
                for q in open_questions:
                    st.markdown(f"**- {q.get('question')}**")
                    if q.get("context"):
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Details: {q.get('context')}*")
                        
        with chat_tab:
            st.markdown("### 💬 Chat with your Meeting Assistant")
            st.write("Ask questions about the discussion, action items, or specific details. Answers are grounded in the transcript.")
            
            chat_key = f"chat_history_app_{meeting['id']}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = []
                
            # Render conversation history
            for msg in st.session_state[chat_key]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    
            # User chat input
            if question := st.chat_input("e.g., What did we decide about the timeline?", key="app_chat_input"):
                with st.chat_message("user"):
                    st.write(question)
                st.session_state[chat_key].append({"role": "user", "content": question})
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing transcript..."):
                        answer = query_meeting(meeting["id"], question)
                        st.write(answer)
                st.session_state[chat_key].append({"role": "assistant", "content": answer})
            
        with time_tab:
            segments = meeting.get("segments", [])
            if not segments:
                st.info("No timestamped segments available for this meeting.")
            else:
                for seg in segments:
                    start_s = format_timestamp(seg["start"])
                    end_s = format_timestamp(seg["end"])
                    st.markdown(f"""
                        <div class="segment-box">
                            <div class="segment-time">[{start_s} - {end_s}]</div>
                            <div class="segment-text">{seg["text"]}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
        with raw_tab:
            st.text_area(
                "Plain Text Transcript",
                value=meeting["transcript_text"],
                height=400,
                disabled=True,
                key="app_raw_transcript"
            )
            
        # Export Actions
        st.markdown("### 💾 Export Results")
        col_dl_pdf, col_dl_sum, col_dl_trans = st.columns(3)
        
        with col_dl_pdf:
            with st.spinner("Preparing PDF report..."):
                try:
                    pdf_data = generate_pdf_report(meeting)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name=f"{meeting['title'].lower().replace(' ', '_')}_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="app_btn_pdf"
                    )
                except Exception as e:
                    st.button(f"PDF Export Error: {e}", disabled=True, use_container_width=True, key="app_pdf_err")
        
        with col_dl_sum:
            if meeting["summary_markdown"]:
                st.download_button(
                    label="Download Summary (.md)",
                    data=meeting["summary_markdown"],
                    file_name=f"{meeting['title'].lower().replace(' ', '_')}_summary.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="app_btn_md"
                )
            else:
                st.button("Download Summary", disabled=True, use_container_width=True, key="app_btn_md_dis")
                
        with col_dl_trans:
            st.download_button(
                label="Download Transcript (.txt)",
                data=meeting["transcript_text"],
                file_name=f"{meeting['title'].lower().replace(' ', '_')}_transcript.txt",
                mime="text/plain",
                use_container_width=True,
                key="app_btn_txt"
            )
