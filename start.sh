#!/bin/bash
# FinAnalyst NLP Platform - Start Script
# ----------------------------------------
# Starts both backend (FastAPI) and frontend (Python HTTP server)

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     FinAnalyst NLP Platform v1.0         ║"
echo "║     Finance Q&A + Document Analyzer      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "ERROR: Python 3 is required. Install it from https://python.org"
  exit 1
fi

# Install backend deps
echo "→ Installing backend dependencies..."
cd "$(dirname "$0")/backend"
pip install -r requirements.txt -q

# Download TextBlob corpora if needed
echo "→ Setting up TextBlob..."
python3 -c "import textblob; textblob.download_corpora()" 2>/dev/null || true

# Start backend in background
echo "→ Starting FastAPI backend on http://localhost:8000 ..."
python3 main.py &
BACKEND_PID=$!

# Wait for backend to come up
sleep 2

# Start frontend
echo "→ Starting frontend on http://localhost:3000 ..."
cd "$(dirname "$0")/frontend"
python3 -m http.server 3000 &
FRONTEND_PID=$!

echo ""
echo "✓ FinAnalyst is running!"
echo ""
echo "  Open in browser:  http://localhost:3000"
echo "  API docs:         http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo ""

# Cleanup on exit
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait
