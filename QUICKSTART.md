# 🏭 Trikhya SOP Saathi - Quick Start Guide

## ✅ Setup Complete!

Your virtual environment is ready and all dependencies are installed.

## 🚀 Running the App

### For Mobile Demo (Recommended):

```bash
./setup_and_run.sh
```

This will start the app with network access. Look for the **Network URL** in the terminal (e.g., `http://192.168.x.x:8501`) and open it on your mobile device.

### For Desktop Only:

```bash
source venv/bin/activate
streamlit run app.py
```

## 📱 Mobile Access Steps

1. **Ensure same WiFi**: Your mobile and computer must be on the same network
2. **Run the app**: Use `./setup_and_run.sh` 
3. **Find Network URL**: Look in terminal for something like `Network URL: http://192.168.1.100:8501`
4. **Open on mobile**: Type that URL in your mobile browser

## 📋 Before Demo

Make sure you have:
- ✅ Added your OpenAI API key to `.env` file
- ✅ Added PDF manuals to the `manuals/` folder
- ✅ Both devices on same WiFi network

## 🎤 Using the App

1. Tap the microphone icon
2. Ask your question in Hindi, Marathi, or English
3. Wait for the AI to process
4. Listen to the audio response
5. Check the source document citation

## 🔧 Troubleshooting

**"No manuals loaded"**
- Add PDF files to the `manuals/` folder
- Restart the app

**Can't access on mobile**
- Verify same WiFi network
- Check firewall settings
- Try the Network URL shown in terminal

**Audio not working**
- Allow microphone permissions in browser
- Check speaker volume

## 📞 Need Help?

Check the full README.md for detailed documentation.
