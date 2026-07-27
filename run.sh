#!/bin/bash
# InclusiveAI — local dev server (creates/uses a venv automatically)
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install -r backend/requirements.txt --quiet
cd backend
echo "InclusiveAI running at http://127.0.0.1:8000"
python3 -m uvicorn main:app --reload --port 8000
