# A Climate-Driven Disease Forecasting Framework for Sri Lanka

A multi-disease forecasting system that uses climate data (rainfall, temperature, humidity) to predict weekly case counts for Leptospirosis, Typhus, Hepatitis A, and Chickenpox across Sri Lanka's 25 districts. The system combines XGBoost and LightGBM ensembles with quantile regression for uncertainty-aware predictions and includes a web dashboard for visualisation and SHAP-based explainability.

---

## Screenshots

### Dashboard

<img width="1919" height="1079" alt="Image" src="https://github.com/user-attachments/assets/7fad05fe-3a04-4633-81f6-048408356d81" />

*Main dashboard showing disease selector, district, and case summaries.*

### Predictions

<img width="1919" height="1079" alt="Image" src="https://github.com/user-attachments/assets/f8105ce2-e6f8-4643-9f55-f8add450db46" />

*Prediction view with lower, median, and upper forecasts per district.*

### Explainability (SHAP)

<img width="1905" height="1063" alt="Image" src="https://github.com/user-attachments/assets/4642e2e7-c033-4530-bc2f-52109df357b6" />

*SHAP bar chart showing which climate features drive risk for a given prediction.*

---

## Features

- **4 diseases:** Leptospirosis, Typhus, Hepatitis A, Chickenpox
- **25 districts:** Nationwide coverage
- **Ensemble model:** Weighted blend of XGBoost and LightGBM with quantile regression (0.05, 0.5, 0.95)
- **Prediction intervals:** Lower, median, and upper bounds per district-week
- **SHAP explainability:** See which climate features drive each prediction
- **Web dashboard:** Streamlit frontend + FastAPI backend
- **Optional live weather:** OpenWeather API for future-week forecasts

---

## Prerequisites

- Python 3.10 or 3.12
- Git
- Install venv
- Install requirements.txt

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Research.git
cd Research
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root directory and add the following:

```env
# Optional: For live weather forecasts in the dashboard
# Get a free key at https://openweathermap.org/api
OPENWEATHER_API_KEY=your_api_key_here

# Optional: Override API URL if backend runs elsewhere (default: http://127.0.0.1:8000)
# API_BASE=http://127.0.0.1:8000

# Optional: Enable LSTM model selector in frontend (default: false)
# LSTM_ENABLED=false
```

**Minimal setup:** You can run the app without `OPENWEATHER_API_KEY`. It will use the last available historical week for future predictions and show a notice in the dashboard.

---

## Data

The pipeline expects a cleaned dataset: `Final_Data_Counts_CLEANED.csv` in the project root. This file should contain weekly district-level disease counts and climate variables. If your data file is elsewhere, update the path in `step_01_dataset_finalization.py` (variable `data_path`).

---

## Running the Pipeline

From the project root (with the virtual environment activated):

```bash
# Step 1: Split data into train/val/test per disease
python step_01_dataset_finalization.py

# Step 2: ARIMA baseline (optional)
python step_02_arima_baseline.py

# Step 3: Train XGBoost + LightGBM ensemble and blend
python step_03_ensemble_blending.py

# Step 4: SHAP explainability
python step_04_shap_explainability.py

# Step 5: LSTM validation (optional)
python step_05_lstm_validation.py

# Step 6: Model evaluation
python step_06_model_evaluation.py
```

**Minimum to run the app:** Steps 1, 3, and 4 must complete successfully. The artifacts are written to `model_data/artifacts/<disease_id>/`. Step 5 (LSTM) requires TensorFlow; uncomment it in `requirements.txt` if needed.

---

## Running the Web Application

### Start the backend (FastAPI)

```bash
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the frontend (Streamlit)

In a new terminal, with the same virtual environment activated:

```bash
streamlit run app/frontend/streamlit_app.py
```

Or use the script:

```bash
./app/run_frontend.sh
```

The dashboard will open at `http://localhost:8501`. Ensure the backend is running on port 8000 before using the frontend.

---

## Project Structure

```
Research/
├── app/
│   ├── backend/          # FastAPI server, inference, data service
│   └── frontend/         # Streamlit dashboard
├── model_data/           # Train/val/test splits, artifacts, SHAP outputs
├── scripts/              # Figure generation scripts
├── step_01_dataset_finalization.py
├── step_03_ensemble_blending.py
├── step_04_shap_explainability.py
├── step_05_lstm_validation.py
├── step_06_model_evaluation.py
├── .env                  # Create this with OPENWEATHER_API_KEY (optional)
├── requirements.txt
└── README.md
```

---

## License

All Rights Reserved.

Copyright (c) 2026. This project and its source code are proprietary. No part of this work may be reproduced, distributed, or transmitted in any form or by any means without prior written permission.

---

**Acknowledgements:** Disease data from the Epidemiology Unit, Ministry of Health, Sri Lanka. Climate data from NASA POWER and OpenWeather.
