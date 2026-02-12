"""
Trikhya SOP Saathi - Voice-Activated AI Supervisor
A factory floor assistant that answers worker questions based on PDF manuals.
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Optional
import tempfile

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# ============================================================================
# CONFIGURATION
# ============================================================================

MANUALS_FOLDER = "manuals"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 3
SYSTEM_PROMPT = """You are an expert Production Supervisor at Spark Minda. 
Answer strictly based on the context provided. 
Detect the user's language (Hindi, Marathi, or English) and reply in the SAME language. 
Keep answers short (under 2 sentences) and authoritative."""

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Trikhya SOP Saathi",
    page_icon="🏭",
    layout="centered",  # Better for mobile devices
    initial_sidebar_state="collapsed"  # Collapsed by default on mobile
)

# ============================================================================
# INITIALIZATION
# ============================================================================

def init_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client with API key."""
    # Try Streamlit secrets first (for cloud deployment), then fall back to .env
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found. Please set it in Streamlit secrets or .env file.")
        return None
    return OpenAI(api_key=api_key)

def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    """Load all PDF files from the specified folder."""
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        st.warning(f"📁 Folder '{folder_path}' not found. Please create it and add PDF manuals.")
        return documents
    
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        st.warning(f"📄 No PDF files found in '{folder_path}'. Please add manual PDFs.")
        return documents
    
    with st.spinner(f"📚 Loading {len(pdf_files)} PDF manual(s)..."):
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                
                # Attach filename as metadata to each chunk
                for doc in pdf_docs:
                    doc.metadata["source"] = pdf_file.name
                
                documents.extend(pdf_docs)
                st.sidebar.success(f"✅ Loaded: {pdf_file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load {pdf_file.name}: {str(e)}")
    
    return documents

def create_vector_store(documents: List[Document]) -> Optional[FAISS]:
    """Create FAISS vector store from documents."""
    if not documents:
        return None
    
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        st.sidebar.info(f"📊 Created {len(chunks)} text chunks from {len(documents)} pages")
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.vector_store_loaded = False

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def transcribe_audio(client: OpenAI, audio_bytes: bytes) -> Optional[str]:
    """Transcribe audio using OpenAI Whisper."""
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Transcribe using Whisper
        with open(tmp_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return transcript.text
    except Exception as e:
        st.error(f"❌ Transcription error: {str(e)}")
        return None

def get_answer_from_rag(query: str, vector_store: FAISS, client: OpenAI) -> tuple[str, str]:
    """Get answer using RAG pipeline."""
    try:
        # Perform similarity search
        docs = vector_store.similarity_search(query, k=TOP_K_RESULTS)
        
        if not docs:
            return "मुझे इस सवाल का जवाब मैनुअल में नहीं मिला। / I couldn't find an answer in the manuals.", "N/A"
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Extract source filename (from the first most relevant document)
        source = docs[0].metadata.get("source", "Unknown")
        
        # Generate answer using GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        answer = response.choices[0].message.content.strip()
        
        return answer, source
    except Exception as e:
        st.error(f"❌ Error generating answer: {str(e)}")
        return "क्षमा करें, एक त्रुटि हुई। / Sorry, an error occurred.", "N/A"

def text_to_speech(client: OpenAI, text: str) -> Optional[bytes]:
    """Convert text to speech using OpenAI TTS."""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text
        )
        return response.content
    except Exception as e:
        st.error(f"❌ TTS error: {str(e)}")
        return None

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render application header."""
    # Mobile-friendly CSS
    st.markdown("""
    <style>
    /* Mobile-optimized styles */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        h1 {
            font-size: 1.8rem !important;
        }
        
        /* Larger touch targets for mobile */
        button {
            min-height: 3rem !important;
            font-size: 1.1rem !important;
        }
        
        /* Better spacing for chat messages */
        .stChatMessage {
            padding: 1rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* Audio input optimization */
        audio {
            width: 100% !important;
        }
    }
    
    /* General improvements */
    .stAudio {
        margin: 1rem 0;
    }
    
    /* Make chat bubbles more readable */
    .stChatMessage {
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏭 Trikhya SOP Saathi")
    st.markdown("**Voice-Activated AI Supervisor**")
    st.caption("Ask questions about factory SOPs in Hindi, Marathi, or English")
    st.divider()

def render_sidebar():
    """Render sidebar with information."""
    with st.sidebar:
        st.header("📋 Status")
        
        if st.session_state.vector_store:
            st.success("✅ Ready")
        else:
            st.warning("⚠️ No manuals")
        
        st.divider()
        
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. 🎤 **Record** your question
        2. ⏳ **Wait** for processing
        3. 🔊 **Listen** to response
        4. 📄 **Check** source
        
        **Languages:**
        - Hindi (हिंदी)
        - Marathi (मराठी)
        - English
        """)

def render_chat_history():
    """Render chat history."""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        source = message.get("source", "")
        
        if role == "user":
            with st.chat_message("user", avatar="👷"):
                st.markdown(f"**Question:** {content}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**Answer:** {content}")
                if source and source != "N/A":
                    st.caption(f"📄 Source: {source}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic."""
    # Initialize
    initialize_session_state()
    
    # Initialize OpenAI client
    client = init_openai_client()
    if not client:
        st.stop()
    
    # Load vector store (only once)
    if not st.session_state.vector_store_loaded:
        documents = load_pdfs_from_folder(MANUALS_FOLDER)
        if documents:
            st.session_state.vector_store = create_vector_store(documents)
            st.session_state.vector_store_loaded = True
        else:
            st.session_state.vector_store_loaded = True  # Mark as attempted
    
    # Render UI
    render_header()
    render_sidebar()
    
    # Check if vector store is ready
    if not st.session_state.vector_store:
        st.error("⚠️ Please add PDF manuals to the 'manuals' folder and restart the app.")
        st.stop()
    
    # Display chat history
    render_chat_history()
    
    # Audio input
    st.markdown("### 🎤 Ask Your Question")
    st.info("👇 Tap the microphone below to record your question")
    audio_bytes = st.audio_input("Record your question")
    
    if audio_bytes:
        with st.spinner("🎧 Transcribing your question..."):
            # Transcribe audio
            question = transcribe_audio(client, audio_bytes.getvalue())
        
        if question:
            # Display transcribed question
            st.info(f"**You asked:** {question}")
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            
            with st.spinner("🧠 Searching manuals and generating answer..."):
                # Get answer from RAG
                answer, source = get_answer_from_rag(
                    question, 
                    st.session_state.vector_store, 
                    client
                )
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source": source
            })
            
            # Display answer
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**Answer:** {answer}")
                if source != "N/A":
                    st.caption(f"📄 Source: {source}")
            
            # Convert to speech and autoplay
            with st.spinner("🔊 Generating audio response..."):
                audio_response = text_to_speech(client, answer)
            
            if audio_response:
                st.audio(audio_response, format="audio/mp3", autoplay=True)
            
            # Rerun to update chat history display
            st.rerun()

if __name__ == "__main__":
    main()
