# Sargodha AQI Prediction Model

## Introduction

This project predicts the next three days of Air Quality Index (AQI) for Sargodha, Pakistan. It combines historical weather and air-quality data from Open-Meteo with daily feature engineering and three Gradient Boosting models. The Streamlit dashboard shows forecasts, AQI categories, weather context, trends, and SHAP-based explanations.

Project detail, data flow, feature engineering, training, registry, and deployment notes are documented in [project_details.ipynb](project_details.ipynb).

## How to Run

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root when Hopsworks access is required:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
```

The Streamlit app also requires the Hopsworks credentials used by its model-loading code. Do not commit `.env` or API keys.

### 3. Start the dashboard

```bash
streamlit run app.py
```

### 4. Retrain the models

```bash
python retrain.py
```

The retraining script loads `sargodha_raw_data_3yrs (5).csv` when available, otherwise fetches historical data from Open-Meteo. It creates daily features, trains the three forecast horizons, saves local artifacts under `retrain_artifacts/`, and uploads improved versions to Hopsworks when credentials are available.
