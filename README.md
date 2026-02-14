# 🏭 Spark Minda SOP Saathi

A voice-activated AI supervisor for factory floor workers that answers questions based on PDF manuals using RAG (Retrieval-Augmented Generation).

## 🎯 Features

- **Voice Input**: Record questions using your microphone
- **Multilingual Support**: Automatically detects and responds in Hindi, Marathi, or English
- **RAG Pipeline**: Searches through PDF manuals using FAISS vector database
- **Audio Output**: Text-to-speech responses for hands-free operation
- **Source Citations**: Shows which manual the answer came from
- **Mobile Optimized**: Responsive design for factory floor use

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Orchestration**: LangChain
- **AI Models**: OpenAI Whisper (ASR), GPT-4o (Logic), TTS-1 (Audio Output)
- **Vector Database**: FAISS
- **PDF Processing**: PyPDF

---

## � Streamlit Cloud Deployment

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at https://share.streamlit.io/)
- OpenAI API key

### Deployment Steps

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit: Spark Minda SOP Saathi"
   git remote add origin https://github.com/YOUR_USERNAME/trikhya-sop-saathi.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app" → Select your repository
   - Main file path: `app.py`
   - Click "Advanced settings" → "Secrets"
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
     ```
   - Click "Deploy!"

3. **Access Your App**:
   - Your app will be live at: `https://trikhya-sop-saathi.streamlit.app`
   - Share this URL with factory supervisors
   - No installation or setup needed on their end!

### Updating the App

To make changes after deployment:
```bash
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically redeploy within 1-2 minutes.

---

## 📖 Usage

1. **Open the app** on any device with internet
2. **Tap the microphone** icon to record your question
3. **Wait** for AI to process (transcribe → search → generate answer)
4. **Listen** to the audio response
5. **Check** the source document citation

**Supported Languages**: Hindi (हिंदी), Marathi (मराठी), English

---

## 🏗️ Project Structure

```
spark_minda_agent/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml         # API keys (update this!)
├── manuals/                 # PDF manuals folder
│   ├── Mahindra_Thar_SOP.pdf
│   └── Maruti_Brezza_SOP.pdf
├── README.md                # This file
└── DEPLOYMENT.md            # Detailed deployment guide
```

---

## 🔧 Troubleshooting

**App won't start on Streamlit Cloud**
- Verify `OPENAI_API_KEY` is set in Streamlit Cloud secrets
- Check deployment logs in Streamlit Cloud dashboard
- Ensure `manuals/` folder exists in GitHub repo

**No manuals loaded**
- Ensure `manuals/` folder (lowercase) is committed to GitHub
- Verify PDF files are in the folder
- Check app logs for errors

**Audio not working**
- Ensure browser allows microphone access
- Try Chrome or Safari (best compatibility)
- Check mobile browser settings

---

## 📝 Next Steps

- **For Demo**: Use Streamlit Cloud deployment (current setup)
- **For Production**: Consider GCP deployment for better control and security

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)


