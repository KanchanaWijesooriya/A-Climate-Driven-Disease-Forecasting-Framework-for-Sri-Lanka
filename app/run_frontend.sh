#!/usr/bin/env bash
# Run Streamlit frontend from Research project root
cd "$(dirname "$0")/.."
streamlit run app/frontend/streamlit_app.py --server.port 8501
