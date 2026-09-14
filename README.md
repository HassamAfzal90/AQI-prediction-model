
# Sargodha AQI Prediction Model
<img width="596" height="335" alt="images (2)" src="https://github.com/user-attachments/assets/ec2b0a68-c7d4-476e-82bb-49f0fc43c9c6" />






A 3-day air quality (AQI) forecasting app for Sargodha, Punjab, Pakistan — built on Open-Meteo weather/AQI data, a Hopsworks-managed ML pipeline, and a Streamlit dashboard.

**Live App:** [AQI Dashboard](https://aqi-prediction-model-9t6xkwfuyapxj7jtkdv38x.streamlit.app/)

## What it does

Predicts AQI for the next 3 days in Sargodha using historical weather + pollution data, and explains *why* the forecast looks the way it does using SHAP. The dashboard shows:

- 3-day AQI forecast (Day 1, Day 2, Day 3 — separate models for each)
- risk classification & health advisory banners
- historical AQI trend + forecast overlay charts
- SHAP-based feature explanations

## Tech Stack

**Core:** Python 3.12+, Streamlit, Pandas, NumPy, scikit-learn, Joblib, Plotly, SHAP

**Data & Deployment:** Open-Meteo (Archive, Forecast, Air Quality APIs), Hopsworks Model Registry, GitHub Actions (weekly auto-retraining), Streamlit Community Cloud

**Model:** 3 separate Gradient Boosting Regressors (`sargodha_aqi_gbr_day1/2/3`)

| Horizon | RMSE | R² |
|---|---:|---:|
| Day 1 | 12.15 | 0.806 |
| Day 2 | 23.08 | 0.719 |
| Day 3 | 23.24 | 0.710 |

## Run Locally
=======
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
>>>>>>> Stashed changes

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Set up Hopsworks API key
# create a .env file in the project root with:
HOPSWORKS_API_KEY=your_key_here

# 4. Run the app
streamlit run app.py
```

<<<<<<< Updated upstream
App opens at `http://localhost:8501` — live weather + AQI data fetches automatically, model loads from Hopsworks, and forecast + SHAP charts render on the dashboard.

## Auto-Retraining

`retrain.py` runs every Sunday via GitHub Actions — retrains all 3 models and only pushes a new version to Hopsworks if it beats the current one. Falls back to a local save if Hopsworks login fails.

```bash
python -m pip install -r requirements-retrain.txt
```
=======
### 4. Retrain the models

```bash
python retrain.py
```

The retraining script loads `sargodha_raw_data_3yrs (5).csv` when available, otherwise fetches historical data from Open-Meteo. It creates daily features, trains the three forecast horizons, saves local artifacts under `retrain_artifacts/`, and uploads improved versions to Hopsworks when credentials are available.

