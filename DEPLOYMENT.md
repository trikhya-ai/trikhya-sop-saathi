# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

- GitHub account
- Streamlit Cloud account (free at https://streamlit.io/cloud)
- Your OpenAI API key

---

## 🎯 Step-by-Step Deployment

### **Step 1: Push Code to GitHub**

1. **Initialize Git repository** (if not already done):
   ```bash
   cd /Users/abhishekgoel/my-agy-projects/spark_minda_agent
   git init
   git add .
   git commit -m "Initial commit: Spark Minda SOP Saathi"
   ```

2. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name: `trikhya-sop-saathi` (or any name you prefer)
   - Keep it **Private** (recommended for API keys)
   - Don't initialize with README (we already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/trikhya-sop-saathi.git
   git branch -M main
   git push -u origin main
   ```

### **Step 2: Upload PDF Manuals to GitHub**

**Important**: Your PDF manuals need to be in the repository for Streamlit Cloud to access them.

```bash
# Make sure manuals are in the correct folder
# Note: GitHub repo should use lowercase 'manuals' folder
mv Manuals manuals  # Rename if needed
git add manuals/
git commit -m "Add PDF manuals"
git push
```

### **Step 3: Deploy on Streamlit Cloud**

1. **Go to Streamlit Cloud**:
   - Visit https://share.streamlit.io/
   - Sign in with GitHub

2. **Create New App**:
   - Click "New app"
   - Select your repository: `trikhya-sop-saathi`
   - Main file path: `app.py`
   - App URL: Choose a custom name (e.g., `trikhya-sop-saathi`)

3. **Add Secrets** (CRITICAL):
   - Click "Advanced settings"
   - Go to "Secrets" section
   - Add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
   ```
   - Click "Save"

4. **Deploy**:
   - Click "Deploy!"
   - Wait 2-3 minutes for deployment

### **Step 4: Test Your App**

Your app will be live at:
```
https://trikhya-sop-saathi.streamlit.app
```

Test it:
- ✅ Open on your phone
- ✅ Record a voice question
- ✅ Verify AI response
- ✅ Check audio playback

---

## 📱 Demo Day Workflow

### **Before the Demo**:
1. Deploy app to Streamlit Cloud (done once)
2. Test the URL on your phone
3. Bookmark the URL for easy access

### **During the Demo**:
1. Hand supervisor your phone or have them open on theirs
2. Navigate to: `https://trikhya-sop-saathi.streamlit.app`
3. Let them ask questions directly
4. Show real-time responses

### **After You Leave**:
- They keep using the same URL
- No setup needed on their end
- Works on any device with internet

---

## 🔄 Updating the App

When you need to make changes:

```bash
# Make your code changes
git add .
git commit -m "Update: description of changes"
git push
```

Streamlit Cloud will **automatically redeploy** within 1-2 minutes!

---

## ⚠️ Important Notes

### **PDF Manuals**:
- Must be in the `manuals/` folder in your GitHub repo
- Streamlit Cloud will load them on startup
- To add new manuals: push to GitHub and app will restart

### **API Key Security**:
- Never commit `.env` file to GitHub (already in `.gitignore`)
- Always use Streamlit Cloud secrets for the API key
- Keep your repository private if it contains sensitive SOPs

### **Free Tier Limits**:
- Streamlit Cloud free tier is sufficient for demos
- App may sleep after inactivity (wakes up in ~30 seconds)
- For production, consider upgrading or on-premise deployment

---

## 🐛 Troubleshooting

**App won't start**:
- Check that `OPENAI_API_KEY` is set in Streamlit secrets
- Verify `manuals/` folder exists in repo
- Check deployment logs in Streamlit Cloud dashboard

**"No manuals loaded" error**:
- Ensure `manuals/` folder (lowercase) is in GitHub repo
- Verify PDF files are committed and pushed
- Check app logs for file loading errors

**Audio not working**:
- Ensure browser has microphone permissions
- Test on different browsers (Chrome/Safari work best)
- Check mobile browser compatibility

---

## 📞 Support

- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Community: https://discuss.streamlit.io/

---

**Ready to deploy?** Follow Step 1 above to get started! 🚀
