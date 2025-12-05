#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js first."
    exit 1
fi
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi
echo "Installing npm dependencies..."
cd "$SCRIPT_DIR"
npm install
echo "Installing Python dependencies..."
cd "$SCRIPT_DIR"
if [ -d "venv" ]; then
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
fi
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements.txt
echo "Cleaning up any existing processes on ports 5001 and 3000..."
lsof -ti:5001 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 2
echo "Starting backend server..."
cd "$SCRIPT_DIR/behaviour_analysis"
python3 behavior_server.py > /dev/null 2>&1 &
BACKEND_PID=$!
echo "Waiting for backend to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:5001 > /dev/null 2>&1; then
        echo "Backend is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "Warning: Backend may not be ready, but continuing anyway..."
    fi
    sleep 1
done
echo "Starting frontend..."
cd "$SCRIPT_DIR"
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found in $SCRIPT_DIR"
    exit 1
fi
export REACT_APP_API_BASE_URL=http://localhost:5001
export PORT=3000
BROWSER=none npm start > "$SCRIPT_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "Waiting for frontend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "Frontend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Warning: Frontend may not be ready, but continuing anyway..."
    fi
    sleep 1
done
URL="http://localhost:3000"
echo "Running Computer Vision attacker..."
cd "$SCRIPT_DIR/attacker/computer_vision"
python3 cv_attacker_run.py "$URL" || true
sleep 2
if [ -n "$GEMINI_API_KEY" ]; then
    echo "Running LLM Sycophancy attacker..."
    cd "$SCRIPT_DIR/attacker/sycophancy"
    python3 llm_sycophancy_run.py "$URL" --gemini-api-key "$GEMINI_API_KEY" || true
    sleep 2
    echo "Running Universal LLM attacker..."
    cd "$SCRIPT_DIR/attacker/llm"
    python3 universal_attacker.py "$URL" --gemini-api-key "$GEMINI_API_KEY" || true
    sleep 2
else
    echo "Skipping LLM attackers (GEMINI_API_KEY not set)"
fi
echo "All attackers completed."
echo "Press Ctrl+C to stop frontend and backend servers"
trap "kill $FRONTEND_PID $BACKEND_PID 2>/dev/null; exit" INT TERM
wait
