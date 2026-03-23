#!/usr/bin/env bash
# Run FastAPI backend from Research project root
cd "$(dirname "$0")/.."
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
