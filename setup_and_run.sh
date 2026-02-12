#!/bin/bash

# Trikhya SOP Saathi - Setup and Run Script
# This script sets up the virtual environment and runs the application

echo "🏭 Trikhya SOP Saathi - Setup Script"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting Streamlit application..."
echo "📱 The app will open in your browser. For mobile demo, use the Network URL shown below."
echo ""

# Run with network URL visible for mobile access
streamlit run app.py --server.address=0.0.0.0
