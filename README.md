# 🏭 Trikhya SOP Saathi

A voice-activated AI supervisor for factory floor workers that answers questions based on PDF manuals using RAG (Retrieval-Augmented Generation).

## 🎯 Features

- **Voice Input**: Record questions using your microphone
- **Multilingual Support**: Automatically detects and responds in Hindi, Marathi, or English
- **RAG Pipeline**: Searches through PDF manuals using FAISS vector database
- **Audio Output**: Text-to-speech responses for hands-free operation
- **Source Citations**: Shows which manual the answer came from
- **Chat History**: Maintains conversation context

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Orchestration**: LangChain
- **AI Models**: 
  - OpenAI Whisper (Speech-to-Text)
  - GPT-4o (Question Answering)
  - TTS-1 (Text-to-Speech)
- **Vector Database**: FAISS
- **PDF Processing**: PyPDF

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API key

## 🚀 Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**
   
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

4. **Add PDF manuals**
   
   Create a `manuals` folder and add your PDF files:
   ```bash
   mkdir manuals
   # Copy your PDF manuals to the manuals/ folder
   ```

## ▶️ Running the Application

### Option 1: Quick Start (Recommended)

Use the provided setup script that handles everything:

```bash
./setup_and_run.sh
```

This script will:
- Create a virtual environment (if not exists)
- Install all dependencies
- Run the application with network access for mobile devices

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # On Mac/Linux
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

The application will open in your default browser at `http://localhost:8501`

### 📱 Mobile Access

To access the app on your mobile device:

1. Make sure your mobile and computer are on the **same WiFi network**
2. Run the app with network access:
   ```bash
   streamlit run app.py --server.address=0.0.0.0
   ```
3. Look for the **Network URL** in the terminal output (e.g., `http://192.168.x.x:8501`)
4. Open that URL on your mobile browser

**Note:** The `setup_and_run.sh` script automatically enables network access for mobile demos.

## 📖 Usage

1. **Record Question**: Click the microphone icon and ask your question in Hindi, Marathi, or English
2. **Wait for Processing**: The app will transcribe, search manuals, and generate an answer
3. **Listen to Response**: The answer will be played automatically via audio
4. **View Source**: Check which manual the answer came from

## 🏗️ Project Structure

```
spark_minda_agent/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .env                  # Your API keys (create this)
├── manuals/              # PDF manuals folder (create this)
│   ├── manual1.pdf
│   └── manual2.pdf
└── README.md             # This file
```

## ⚙️ Configuration

You can modify these constants in `app.py`:

- `CHUNK_SIZE`: Size of text chunks for RAG (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `TOP_K_RESULTS`: Number of relevant chunks to retrieve (default: 3)
- `SYSTEM_PROMPT`: Instructions for the AI supervisor

## 🔧 Troubleshooting

**No manuals loaded**
- Ensure the `manuals/` folder exists
- Add at least one PDF file to the folder
- Restart the application

**API Key errors**
- Verify your `.env` file contains a valid `OPENAI_API_KEY`
- Check that the key has sufficient credits

**Audio not working**
- Ensure your browser allows microphone access
- Check that your speakers/headphones are connected

## 📝 License

This project is for demonstration purposes for Spark Minda factory floor operations.

## 🤝 Support

For issues or questions, please contact the development team.
