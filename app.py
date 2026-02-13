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

# Model-to-file mapping
MODEL_FILE_MAPPING = {
    "Maruti Brezza": "manuals/Maruti_Brezza_SOP.pdf",
    "Mahindra Thar": "manuals/Mahindra_Thar_SOP.pdf"
}
SYSTEM_PROMPT = """
### System Persona
You are an expert Production Supervisor at Spark Minda. Your goal is to provide precise, authoritative assembly instructions to factory workers on the shop floor.

### Operational Guidelines

1. **Input Analysis:**
   - Detect if the user is speaking English, Hindi, Marathi, or "Hinglish" (Mixed).
   - Acknowledge that workers often use technical English nouns mixed with vernacular grammar.

2. **Cognitive Process (Mental Translation):**
   - **Internal Mapping:** If the user speaks in Hindi/Marathi, internally translate their intent to the English technical keywords found in the SOP context.
     - *Example:* If user asks for "garmi" (heat), look for "Thermal" or "Overheating" in the context.
     - *Example:* If user asks for "taar" (wire), look for "Cable", "FPC", or "Harness".
   - Use this internal understanding to retrieve the most accurate technical answer from the provided English context.

3. **Precision Protocol (Anti-Hallucination):**
   - **QUOTE EXACT VALUES:** If the manual says "< 1mA", state "less than 1 milliamp". Do NOT estimate or round it to "0.5mA".
   - **EXACT TORQUE:** If the manual specifies "0.6 Nm", state "0.6 Newton Meters". Do not round down to "0.5 Nm". Accuracy is safety.
   - If the specific value is not in the context, politely inform the user in their own language (Hindi/Marathi/English) that this specific data is not mentioned in the SOP. Do not guess.

4. **Voice-Optimized Output:**
   - **Language Matching:** Reply in the SAME language structure as the user. (User speaks Hindi -> You speak Hindi).
   - **Ear-Friendly:** Use short, punchy sentences (under 2 sentences). Avoid markdown tables, bullet points, or complex lists that sound bad in Text-to-Speech.
   - **Code-Switching:** When speaking Hindi/Marathi, keep technical nouns in **English** (e.g., say "Torque," "Connector," "Probe," "Thermal Paste") so the worker understands. Do not translate technical terms into pure Hindi.

5. **Crucial Rule:**:
   - If the user mentions a new car model (e.g., switches from Brezza to Thar), the NEW model overrides the history. Do not mix attributes of two different cars.

### Response Format
[Direct Answer with Exact Value] + [Brief Consequence/Risk if ignored].
"""

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

def load_single_pdf(pdf_path: str) -> List[Document]:
    """Load a single PDF file."""
    documents = []
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        st.error(f"📄 PDF file not found: {pdf_path}")
        return documents
    
    try:
        with st.spinner(f"📚 Loading {pdf_file.name}..."):
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
    
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.vector_store_loaded = False
    
    if "last_processed_audio" not in st.session_state:
        st.session_state.last_processed_audio = None

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

def rewrite_query_with_context(client: OpenAI, original_query: str, chat_history: list) -> str:
    """
    Rewrite user query to be standalone and keyword-rich using chat history.
    This prevents hallucinations on follow-up questions like "And what if it is uneven?"
    Uses last 5 exchanges to ensure model names (Brezza/Thar) persist.
    """
    # If no chat history, return original query
    if not chat_history:
        return original_query
    
    try:
        # Get last 5 Q&A pairs (10 messages) for context
        # This ensures model names persist even after multiple follow-ups
        last_messages = chat_history[-10:] if len(chat_history) >= 10 else chat_history
        context_summary = ""
        
        for msg in last_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_summary += f"{role}: {msg['content']}\n"
        
        # Rewriting prompt
        rewrite_prompt = f"""Given this conversation context:

{context_summary}

The user now asks: "{original_query}"

Your task: Rewrite this question to be STANDALONE and SPECIFIC, explicitly mentioning the subject from the previous conversation. Include technical keywords that would appear in an SOP manual.

Rules:
1. If the question refers to "it", "that", "this", replace with the actual subject
2. Add technical keywords (e.g., "FPC cable", "display unit", "torque specification")
3. Keep the language (Hindi/English/Marathi) the same as the original
4. Make it searchable - think about what keywords would be in the manual
5. IMPORTANT: If a car model (Brezza/Thar/etc.) was mentioned in recent context, include it in the rewritten query
6. If user mentions a NEW model, use the new model (it overrides history)

Output ONLY the rewritten question, nothing else."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a query rewriting assistant for technical SOP manuals."},
                {"role": "user", "content": rewrite_prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        
        rewritten_query = response.choices[0].message.content.strip()
        
        # Debug output
        print(f"\n🔄 QUERY REWRITE:")
        print(f"   Original: {original_query}")
        print(f"   Rewritten: {rewritten_query}\n")
        
        return rewritten_query
        
    except Exception as e:
        st.warning(f"⚠️ Query rewriting failed, using original: {str(e)}")
        return original_query

def get_answer_from_rag(query: str, vector_store: FAISS, client: OpenAI) -> tuple[str, str]:
    """Get answer using RAG pipeline."""
    try:
        # Perform similarity search with scores to get most relevant documents
        docs_with_scores = vector_store.similarity_search_with_score(query, k=TOP_K_RESULTS)
        
        if not docs_with_scores:
            return "मुझे इस सवाल का जवाब मैनुअल में नहीं मिला। / I couldn't find an answer in the manuals.", "N/A"
        
        # Extract documents (already sorted by score - lower is better)
        docs = [doc for doc, score in docs_with_scores]
        
        # Prepare context with source labels
        context_parts = []
        for i, doc in enumerate(docs):
            source_name = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Source: {source_name}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        
        # Enhanced system prompt to identify source
        enhanced_prompt = SYSTEM_PROMPT + "\n\nIMPORTANT: After answering, on a new line write 'SOURCE_USED: ' followed by the exact filename of the source document you primarily used for your answer."
        
        # Generate answer using GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # Extract answer and source from response
        if "SOURCE_USED:" in full_response:
            parts = full_response.split("SOURCE_USED:")
            answer = parts[0].strip()
            source = parts[1].strip()
        else:
            # Fallback to first document if GPT doesn't specify
            answer = full_response
            source = docs[0].metadata.get("source", "Unknown")
        
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
    # Mobile-optimized CSS with compact spacing
    st.markdown("""
    <style>
    /* Reduce overall padding for compact layout */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Compact headers */
    h1 {
        margin-bottom: 0.2rem !important;
    }
    
    h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Reduce divider spacing */
    hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Mobile-optimized styles */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.5rem !important;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        h1 {
            font-size: 1.5rem !important;
            margin-bottom: 0.1rem !important;
        }
        
        h3 {
            font-size: 1.1rem !important;
            margin-top: 0.3rem !important;
            margin-bottom: 0.2rem !important;
        }
        
        /* LARGE microphone button for mobile - HERO ELEMENT */
        /* Remove default Streamlit audio input styling */
        .stAudioInput {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        .stAudioInput > div {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 1rem 0 !important;
        }
        
        /* Style the actual microphone button */
        .stAudioInput button {
            min-height: 80px !important;
            min-width: 80px !important;
            width: 80px !important;
            height: 80px !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%) !important;
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3) !important;
            border: none !important;
            padding: 0 !important;
        }
        
        .stAudioInput button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 6px 16px rgba(255, 107, 107, 0.4) !important;
        }
        
        .stAudioInput button:active {
            transform: scale(0.95) !important;
        }
        
        /* Better spacing for chat messages */
        .stChatMessage {
            padding: 0.8rem !important;
            margin-bottom: 0.5rem !important;
        }
    }
    
    /* General improvements */
    .stAudio {
        margin: 0.5rem 0;
    }
    
    /* Enhanced chat bubble styling with distinct colors */
    .stChatMessage {
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    /* User messages - Light Blue background */
    div[data-testid="stChatMessage"]:has(div[aria-label*="user"]) {
        background-color: #E3F2FD !important;
        border-left: 4px solid #2196F3 !important;
    }
    
    /* Assistant messages - Light Green background */
    div[data-testid="stChatMessage"]:has(div[aria-label*="assistant"]) {
        background-color: #E8F5E9 !important;
        border-left: 4px solid #4CAF50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏭 Trikhya SOP Saathi")
    st.caption("Voice-Activated AI Supervisor • Hindi, Marathi, English")

def render_sidebar():
    """Render sidebar with information."""
    with st.sidebar:
        st.header("📋 Status")
        
        if st.session_state.current_model:
            st.success(f"✅ {st.session_state.current_model}")
        
        if st.session_state.vector_store:
            st.success("✅ Manual Loaded")
        else:
            st.warning("⚠️ No manual loaded")
        
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
    """Render chat history with Q&A pairs grouped correctly (newest first)."""
    # Group messages into Q&A pairs
    qa_pairs = []
    i = 0
    while i < len(st.session_state.messages):
        if i + 1 < len(st.session_state.messages):
            # Check if we have a user-assistant pair
            if (st.session_state.messages[i]["role"] == "user" and 
                st.session_state.messages[i + 1]["role"] == "assistant"):
                qa_pairs.append((st.session_state.messages[i], st.session_state.messages[i + 1]))
                i += 2
            else:
                # Single message without pair
                qa_pairs.append((st.session_state.messages[i], None))
                i += 1
        else:
            # Last message without pair
            qa_pairs.append((st.session_state.messages[i], None))
            i += 1
    
    # Reverse to show latest first, but keep Q&A pairs together
    for user_msg, assistant_msg in reversed(qa_pairs):
        # Display user question
        with st.chat_message("user", avatar="👷"):
            st.markdown(f"**Question:** {user_msg['content']}")
        
        # Display assistant answer if exists
        if assistant_msg:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**Answer:** {assistant_msg['content']}")
                source = assistant_msg.get("source", "")
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
    
    # Render UI
    render_header()
    
    # ========================================================================
    # MODEL SELECTION (Top of page for mobile visibility)
    # ========================================================================
    st.markdown("### 🚗 Select Assembly Line")
    selected_model = st.radio(
        "Choose your production line:",
        options=list(MODEL_FILE_MAPPING.keys()),
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # DETECT MODEL CHANGE - Clear state if model switched
    if st.session_state.current_model != selected_model:
        st.session_state.current_model = selected_model
        st.session_state.vector_store = None
        st.session_state.vector_store_loaded = False
        st.session_state.messages = []  # Clear chat history
        st.session_state.last_processed_audio = None
        st.rerun()  # Force UI refresh
    
    # LOAD MODEL-SPECIFIC PDF
    if not st.session_state.vector_store_loaded:
        pdf_path = MODEL_FILE_MAPPING[selected_model]
        documents = load_single_pdf(pdf_path)
        if documents:
            st.session_state.vector_store = create_vector_store(documents)
            st.session_state.vector_store_loaded = True
        else:
            st.session_state.vector_store_loaded = True  # Mark as attempted
    
    render_sidebar()
    
    # Check if vector store is ready
    if not st.session_state.vector_store:
        st.error(f"⚠️ Failed to load manual for {selected_model}. Please check the file exists.")
        st.stop()
    
    st.divider()
    
    # Audio input section - microphone is the hero element
    st.markdown("### 🎤 Ask Your Question")
    audio_bytes = st.audio_input("Record your question", label_visibility="collapsed")
    
    if audio_bytes:
        # Get audio hash to prevent reprocessing the same audio
        audio_data = audio_bytes.getvalue()
        audio_hash = hash(audio_data)
        
        # Only process if this is new audio
        if audio_hash != st.session_state.last_processed_audio:
            st.session_state.last_processed_audio = audio_hash
            
            with st.spinner("🎧 Transcribing your question..."):
                # Transcribe audio
                question = transcribe_audio(client, audio_data)
            
            if question:
                # Display transcribed question
                st.info(f"**You asked:** {question}")
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                
                with st.spinner("🧠 Searching manuals and generating answer..."):
                    # CONTEXTUAL QUERY REWRITING: Rewrite query using chat history
                    # This prevents hallucinations on follow-up questions
                    rewritten_query = rewrite_query_with_context(
                        client, 
                        question, 
                        st.session_state.messages[:-1]  # Exclude the just-added question
                    )
                    
                    # Get answer from RAG using the rewritten query
                    answer, source = get_answer_from_rag(
                        rewritten_query,  # Use rewritten query for search
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
    
    # Divider before history
    st.divider()
    
    # Chat history section (compact)
    if st.session_state.messages:
        st.markdown("### 📜 Chat History")
        render_chat_history()

if __name__ == "__main__":
    main()
