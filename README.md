
```markdown
# 🎙️ AI Meeting Assistant

> **Turn raw meeting recordings or YouTube links into actionable notes, structured summaries, and an interactive Q&A knowledge base.**

An end-to-end Python application that automates meeting documentation. Feed it any YouTube video or local audio/video file to generate precise transcripts, structured action items, executive summaries, and chat directly with your transcript using **Retrieval-Augmented Generation (RAG)**.

---

## 🌟 Key Features

- ** flexible Input Handling**: Direct ingestion from YouTube URLs or local file uploads (`.mp3`, `.wav`, `.m4a`, `.mp4`).
- ** Hybrid Speech-to-Text Backends**:
  - **OpenAI Whisper**: Local, offline, $0 API cost, high-accuracy transcription.
  - **Sarvam AI (`saaras:v3`)**: Specialized cloud transcription for Indian accents, regional languages, and code-mixed **Hinglish**.
- ** Structured Insight Extraction**:
  - **Executive Summaries**: High-level discussion summaries generated via **Mistral AI**.
  - **Action Item Extraction**: Leverages **Pydantic** models to parse tasks, assigned owners, deadlines, decisions, and open questions into structured JSON.
- ** Interactive RAG Chat**: Chat with your meeting transcripts in real-time. Anchored strictly to transcript context using **ChromaDB** and **Sentence-Transformers**.
- ** Dual-Layer Persistence**:
  - **SQLite**: Stores metadata, canonical transcripts, and structured extractions.
  - **ChromaDB**: Manages vector embeddings for semantic similarity searches.
- ** Export Options**: Generate and download formatted reports in **PDF**, Markdown, or plain text formats.

---

## 🏗️ Technical Architecture


```

[User Input: YouTube URL / Local File]
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Ingestion, Conversion & Chunking                │
│ • yt-dlp extracts audio stream                           │
│ • FFmpeg + PyDub normalizes to 16kHz Mono PCM WAV        │
│ • Sliced into chunks (Whisper: ~10m | Sarvam: ~25s)      │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 2: Speech-to-Text (STT) Engine                     │
│ • Local: OpenAI Whisper (PyTorch Encoder-Decoder)        │
│ • Cloud: Sarvam AI REST API (Saaras v3)                  │
│ • Timestamp Recalibration & Concatenation               │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 3: AI Processing & Analysis                        │
│ • Mistral AI + LangChain LCEL -> Markdown Summary       │
│ • Pydantic Schema Output Parser -> Structured Action Items│
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────┴───────────────────────────────┐
│ Phase 4: Storage & Indexing                              │
│ ├─► SQLite (meetings.db): Canonical Data & History       │
│ └─► ChromaDB (chroma_db/): 384-dim Embeddings (MiniLM)   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 5: RAG Q&A Engine & Export                         │
│ • Cosine similarity search over top-K chunks             │
│ • Context-grounded response generation with Mistral      │
└──────────────────────────────────────────────────────────┘

```

---

## 🛠️ Tech Stack & Dependencies

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **User Interface** | `Streamlit` | Web UI framework |
| **Ingestion** | `yt-dlp` | Audio downloading from YouTube |
| **Audio Processing**| `FFmpeg` / `PyDub` | Audio conversion, normalization, and chunking |
| **Speech-to-Text** | `OpenAI Whisper` | Offline, local speech recognition |
| | `Sarvam AI` | Cloud API for Hindi & Hinglish transcription |
| **LLM Orchestration**| `LangChain (LCEL)` | Chain construction and prompt templating |
| **LLM Engine** | `Mistral AI` | Summarization, structured extraction, and RAG Q&A |
| **Embeddings** | `sentence-transformers` | `all-MiniLM-L6-v2` local vector embeddings |
| **Vector Database**| `ChromaDB` | Embedded vector database for RAG context retrieval |
| **Relational DB** | `SQLite` | Persistent storage for history and metadata |
| **Export Engine** | `FPDF2` | Exporting meeting summaries to PDF |

---

## 📂 Project Structure


```

ai-meeting-assistant/
├── app.py                  # Main Streamlit web application interface
├── main.py                 # Alternate application entry point
├── test_pipeline.py        # CLI script for headless pipeline testing
├── requirements.txt        # Python dependency manifest
├── .env.example            # Environment variables template
├── utils/
│   ├── audio_processor.py  # Ingestion, WAV downsampling, and audio chunking
│   ├── transcriber.py     # Wrapper for Whisper and Sarvam AI backends
│   ├── summarise.py       # LangChain LCEL summarization pipeline
│   ├── extractor.py       # Pydantic-based structured extraction pipeline
│   ├── history.py         # SQLite CRUD management
│   ├── vector_store.py    # ChromaDB indexing and vector search
│   ├── rag_engine.py      # Retrieval-Augmented Generation Q&A pipeline
│   └── exporter.py        # PDF document generator
├── downloads/              # Temporary directory for intermediate WAV files
├── meetings.db             # SQLite database (created at runtime)
└── chroma_db/              # ChromaDB vector store directory (created at runtime)

```

---

## 🚦 Getting Started

### Prerequisites

- **Python**: Version `3.10` or higher.
- **FFmpeg**: Required for audio transcoding.
  - **Windows**: Install via `winget install FFmpeg` or download binaries.
  - **macOS**: Install via Homebrew: `brew install ffmpeg`
  - **Linux**: Install via APT: `sudo apt install ffmpeg`

---

### Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/ai-meeting-assistant.git](https://github.com/your-username/ai-meeting-assistant.git)
   cd ai-meeting-assistant

```

2. **Create and activate a virtual environment**:
```bash
# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate

```


3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt

```



---

### Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env

```


2. Configure your API keys inside `.env`:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here  # Optional: Required only for Sarvam AI backend

```



> **Note**: OpenAI Whisper runs entirely locally on PyTorch and does **not** require an API key.

---

## 🏃 Usage

### 1. Launching the Streamlit App

Start the web application locally:

```bash
streamlit run app.py

```

Open your browser and navigate to `http://localhost:8501`.

#### How to use:

1. Navigate to **Ingest & Transcribe**.
2. Select your input type: **YouTube URL** or **Local File**.
3. Choose your preferred transcription backend (**Whisper** or **Sarvam AI**).
4. Click **Process Meeting**.
5. View structured summaries, action items, interact via RAG Q&A, or export reports on the results page.

---

### 2. Testing via CLI (Headless Mode)

To run the end-to-end pipeline directly from the command line without launching the UI:

```bash
python test_pipeline.py --url "[https://www.youtube.com/watch?v=YOUR_VIDEO_ID](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)" --backend whisper --whisper-model tiny

```

---

## 🧠 Technical Deep-Dive

### Audio Chunking & Processing

To prevent VRAM/RAM Out-Of-Memory (OOM) errors and stay within REST API payload limits:

* Audio is normalized to **16kHz, 16-bit Mono PCM WAV** via FFmpeg.
* Audio is split into fixed time chunks:

$$\text{Chunk Size} = \begin{cases} 600\text{ seconds} & \text{for Whisper} \\ 25\text{ seconds} & \text{for Sarvam AI} \end{cases}$$


* Global segment timestamps are adjusted by adding cumulative offsets:

$$\text{Global Timestamp} = \text{Local Segment Start} + \sum \text{Previous Chunk Durations}$$



### Retrieval-Augmented Generation (RAG)

1. **Chunking**: The final transcript is split using `RecursiveCharacterTextSplitter` into **1000-character chunks** with a **200-character overlap**.
2. **Embedding**: `all-MiniLM-L6-v2` encodes each chunk into a **384-dimensional dense vector**.
3. **Storage**: Embeddings are stored in **ChromaDB** with a metadata filter tag: `{"meeting_id": current_id}`.
4. **Retrieval**: User queries are vectorized, and the **top 4 ($K=4$) closest contexts** are retrieved using cosine similarity to anchor Mistral's response strictly to the transcript.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```
