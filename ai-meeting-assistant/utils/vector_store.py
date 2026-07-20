import os
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_core.documents import Document
        
        class RecursiveCharacterTextSplitter:
            """
            A robust pure-Python fallback implementation of RecursiveCharacterTextSplitter
            that guarantees independence from varying LangChain package layouts.
            """
            def __init__(self, chunk_size=1000, chunk_overlap=200):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str) -> list[str]:
                chunks = []
                start = 0
                text_len = len(text)
                
                while start < text_len:
                    end = start + self.chunk_size
                    if end >= text_len:
                        chunks.append(text[start:])
                        break
                        
                    # Locate best boundary in overlap region
                    overlap_start = max(start, end - self.chunk_overlap)
                    search_str = text[overlap_start:end]
                    
                    best_idx = -1
                    for sep in ["\n\n", "\n", ". ", " "]:
                        idx = search_str.rfind(sep)
                        if idx != -1:
                            best_idx = idx + len(sep)
                            break
                            
                    if best_idx != -1:
                        end = overlap_start + best_idx
                        
                    chunks.append(text[start:end])
                    
                    # Advance start index
                    next_start = end - self.chunk_overlap
                    if next_start <= start:
                        start = end
                    else:
                        start = next_start
                return chunks

            def create_documents(self, texts: list[str], metadatas: list[dict] = None) -> list:
                docs = []
                for i, text in enumerate(texts):
                    meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                    for chunk in self.split_text(text):
                        docs.append(Document(page_content=chunk, metadata=meta))
                return docs



# Robust import structure for HuggingFace embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError("Could not import HuggingFaceEmbeddings. Make sure 'sentence-transformers' and 'langchain' are installed.")

# Robust import structure for Chroma DB
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    try:
        from langchain_chroma import Chroma
    except ImportError:
        raise ImportError("Could not import Chroma vectorstore. Make sure 'chromadb' and 'langchain' are installed.")

CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))

def get_vectorstore():
    """
    Initializes and returns the persistent Chroma vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="meeting_transcripts",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore

def index_transcript(meeting_id: int, transcript_text: str):
    """
    Splits the transcript text, computes embeddings, and stores them in ChromaDB.
    Cleans up any existing entries for the same meeting_id first.
    """
    if not transcript_text or not transcript_text.strip():
        print("[VectorStore] Warning: Empty transcript, skipping indexing.")
        return
        
    vectorstore = get_vectorstore()
    
    # 1. Clean up existing chunks for this meeting_id to prevent duplicates
    try:
        # Access underlying Chroma collection to delete by metadata where clause
        vectorstore._collection.delete(where={"meeting_id": meeting_id})
        print(f"[VectorStore] Cleaned up existing chunks for meeting ID: {meeting_id}")
    except Exception as e:
        # Silently skip if no collection exists yet or if delete is not supported
        print(f"[VectorStore] Note: Could not clear existing chunks (this may be a new database): {e}")
        
    # 2. Segment the text using standard chunk sizes
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.create_documents(
        texts=[transcript_text],
        metadatas=[{"meeting_id": meeting_id}]
    )
    
    # 3. Embed and index documents
    print(f"[VectorStore] Embedding and indexing {len(docs)} chunk(s) for meeting ID {meeting_id}...")
    vectorstore.add_documents(docs)
    
    # 4. Save/Persist database (newer versions auto-persist, but older versions require calling persist())
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
        
    print("[VectorStore] Indexing completed successfully.")
