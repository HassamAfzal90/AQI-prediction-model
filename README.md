# Sargodha AQI Prediction Model

A production-style air quality forecasting project for Sargodha, Punjab, Pakistan. The system combines historical weather and AQI data from Open-Meteo with a Hopsworks-managed machine learning pipeline and presents a clean 3-day AQI forecast dashboard in Streamlit.

## Project Summary

This project is designed to answer one practical question:

> What will the AQI look like over the next 3 days in Sargodha, and which factors are pushing it up or down?

The solution combines:

- live weather data from Open-Meteo
- historical AQI and pollutant data from Open-Meteo
- engineered lag and rolling features for pollution patterns
- three trained Gradient Boosting models for Day 1, Day 2, and Day 3
- SHAP-based explainability to interpret the forecast
- a real-time Streamlit front end for a user-friendly dashboard

The app is meant for monitoring pollution risk, highlighting vulnerable periods, and making AQI forecasts more interpretable for end users.

## Live App

Live deployment link:

- Streamlit app: https://your-app-name.streamlit.app

> Replace the placeholder with the actual deployed URL from Streamlit Community Cloud once the app is published.

## Dashboard Preview

![Dashboard overview](docs/dashboard-preview.png)

![AQI trend forecast](docs/trend-forecast.png)

---

## Why this project matters

Air quality is highly sensitive to both weather and recent pollution patterns. In cities like Sargodha, AQI can change quickly due to:

- PM2.5 and PM10 fluctuations
- temperature and humidity trends
- wind conditions
- seasonal behavior
- recent pollution persistence

This project uses historical patterns and future weather signals to forecast AQI up to 3 days ahead so that users can act early when pollution is expected to worsen.

---

## Core functionality

- 3-day AQI forecast for each day horizon
- live weather metrics from Open-Meteo
- AQI risk classification and health advisory labels
- hazard banners for moderate and high pollution cases
- historical AQI trend charts plus model forecast overlay
- feature-based explanation using SHAP values
- local `.env` support plus Streamlit Cloud secret management

---

## Architecture overview

```text
Open-Meteo Archive API + Forecast API
            |
            v
Historical data ingestion and daily aggregation
            |
            v
Feature engineering (lags, rolling stats, seasonality, future weather)
            |
            v
Three separate Gradient Boosting models
            |
            +--> Hopsworks Model Registry
            |
            +--> Local retraining artifacts
            |
            v
Streamlit dashboard (app.py)
            |
            v
AQI forecast + alerts + interpretability + charts
```

---

## Repository structure

```text
.
├── app.py                          # Streamlit dashboard
├── retrain.py                     # Training pipeline and model registry workflow
├── new.ipynb                      # Notebook experimentation and model analysis
├── requirements.txt               # Runtime app dependencies
├── requirements-retrain.txt       # Retraining dependencies
├── README.md                      # Project documentation
├── .env                           # Local environment variables (ignored in Git)
├── .gitignore                     # Git ignore rules
├── docs/
│   ├── dashboard-preview.png      # Main dashboard screenshot
│   └── trend-forecast.png        # Forecast chart screenshot
├── sargodha_raw_data_3yrs (5).csv # Raw historical weather/AQI dataset
├── sargodha_features_daily_v2.csv # Daily feature-engineered dataset
├── retrain_artifacts/             # Generated trained model artifacts
└── .github/
    └── workflows/
        └── retrain.yml            # Automated retraining workflow
```

---

## Technology stack

### Core
- Python 3.10+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Joblib
- Plotly
- SHAP

### Data and deployment
- Open-Meteo Archive API
- Open-Meteo Forecast API
- Open-Meteo Air Quality API
- Hopsworks Model Registry
- GitHub Actions
- Streamlit Community Cloud

### ML libraries used during experimentation
- Gradient Boosting Regressor
- Random Forest Regressor
- Ridge / linear baselines
- SHAP explainability workflow

---

## Model details

The final production-style deployment uses three separate models, one for each forecast horizon:

- Day 1 model: `sargodha_aqi_gbr_day1`
- Day 2 model: `sargodha_aqi_gbr_day2`
- Day 3 model: `sargodha_aqi_gbr_day3`

### Model type
The trained forecasting models are Gradient Boosting Regressors (GBR), selected because they perform well on structured tabular environmental data and handle non-linear interactions between weather and AQI variables.

### Model scenarios
The project explicitly trains and supports these use cases:

1. Day 1 forecast
   - uses recent AQI persistence
   - includes short-term weather and pollutant patterns
   - focuses on immediate next-day pollution risk

2. Day 2 forecast
   - uses slightly longer lag patterns
   - incorporates forecast weather and previous AQI behavior
   - useful for short-range operational planning

3. Day 3 forecast
   - captures broader trend continuity and seasonality
   - uses future weather drivers for the next 3-day horizon
   - useful for early warnings and public health planning

---

## Accuracy and validation summary

The notebook experiments and model registry workflow report the following approximate validation metrics for the GBR models:

| Horizon | Model | RMSE | R² | Notes |
|---|---|---:|---:|---|
| Day 1 | Gradient Boosting Regressor | 25.21 | 0.706 | Good short-term predictive strength |
| Day 2 | Gradient Boosting Regressor | 24.49 | 0.715 | Best overall R² among the three horizons |
| Day 3 | Gradient Boosting Regressor | 24.97 | 0.701 | Stable 3-day trend forecasting |

These values indicate that the model is reasonably capable of carrying short-term AQI dynamics, with the Day 2 configuration being the strongest in the recorded benchmark run.

> Note: these metrics reflect the notebook/model-registry evaluation runs and may vary slightly with retraining data and package versions.

---

## Dataset and feature engineering pipeline

The project uses daily aggregated environmental data. The feature engineering process is driven by the notebook workflow and retraining script.

### Raw inputs
- temperature
- relative humidity
- wind speed
- pressure
- precipitation
- PM10
- PM2.5
- AQI
- carbon monoxide
- nitrogen dioxide
- sulphur dioxide
- ozone

### Engineered features
- lagged AQI values: 1, 2, 3, 5, 7, 14 days
- lagged PM2.5 values
- lagged temperature values
- rolling AQI statistics: mean, max, min, std over 3, 7, 14, 30-day windows
- calendar signals: month, day of week, day of year, weekend flag
- seasonal sine/cosine terms
- future weather variables for the target day horizon

This makes the feature space richer and allows the model to learn both persistence and periodic environmental patterns.

---

## Notebook pipeline (`new.ipynb`)

The notebook performs the full experimental workflow:

1. Load the raw hourly weather and AQI dataset
2. Convert hourly records to daily aggregated values
3. Engineer lag, rolling, seasonal, and forecast-weather features
4. Split the data into train/test sets
5. Train multiple model families for comparison
6. Compute RMSE, MAE, and R² scores
7. Save the strongest model artifacts
8. Export feature names and model objects for registry deployment
9. Generate SHAP plots for explanation and insight

### Main notebook contribution
The notebook was used to validate the modeling approach and to confirm which features matter most for AQI prediction. It also generates explainability plots that show which variables push AQI upward or downward.

### SHAP explainability flow
The notebook uses SHAP on the trained model and analyses the feature contributions. The app then mirrors this in the dashboard so end users can see which feature is most responsible for the predicted AQI behavior.

---

## Retraining pipeline (`retrain.py`)

The project includes a dedicated retraining pipeline that automates training and model registration.

### Retraining flow
`retrain.py` does the following:

1. Loads or fetches raw historical weather and AQI data
2. Builds the daily feature dataset
3. Creates engineered features matching the notebook pipeline
4. Trains separate Gradient Boosting models for Day 1, Day 2, and Day 3
5. Compares the new metrics against existing Hopsworks versions
6. Uploads new versions only if they improve the current model
7. Saves artifacts to `retrain_artifacts/`

### Important model-management behavior
The script is designed to be conservative:

- it does not upload a new model if the metrics do not improve
- it keeps registry history clean
- it exposes versioned model artifacts for easier rollback and comparison

### Artifact names
The retraining system writes model artifacts like:

- `model.pkl`
- `features.pkl`
- `retrain_artifacts/...`

These are then registered with Hopsworks by model name:

- `sargodha_aqi_gbr_day1`
- `sargodha_aqi_gbr_day2`
- `sargodha_aqi_gbr_day3`

---

## Deployment and Streamlit app

The Streamlit app in [app.py](app.py) is the public user-facing dashboard. It:

- loads the Hopsworks API key from `.env` or environment variables
- retrieves live weather and AQI data using Open-Meteo APIs
- assembles feature vectors for each forecast day
- loads the corresponding model from Hopsworks
- predicts AQI for the next 3 days
- renders charts, hazard banners, and explainability panels

### Local run command

```bash
streamlit run app.py
```

### Deployment note
This project is designed for deployment on Streamlit Community Cloud. The app is expected to run when the repo is connected to a Streamlit app and the required secrets are configured.

---

## Hopsworks dependency and trigger caveats

The project depends on both Hopsworks and the ML stack, and this creates real production risks:

- Hopsworks API versions can change across environments
- model registry serialization may differ across scikit-learn / NumPy versions
- older pickles may fail if a newer environment is used without compatibility handling
- scheduled retraining or automation can fail if dependency versions drift or the API key is invalid

This project already includes compatibility handling to reduce breakage from older scikit-learn pickle formats, including a compatibility shim for the legacy `_loss` module in [app.py](app.py).

### Why auto triggers sometimes fail
The automated trigger may fail because:

- package version mismatches in GitHub Actions or Cloud environments
- model registry access issues caused by stale or expired Hopsworks credentials
- incompatibility between the saved model object and the runtime environment
- different dependency locks between local development and deployment environments

This means the app may still work locally, but retraining automation can fail if environment parity is not maintained carefully.

---

## Environment setup

### App dependencies

```bash
python -m pip install -r requirements.txt
```

### Retraining dependencies

```bash
python -m pip install -r requirements-retrain.txt
```

### Hopsworks secret setup

Local `.env` file:

```dotenv
HOPSWORKS_API_KEY=your_key_here
```

Streamlit Cloud secret:

```toml
HOPSWORKS_API_KEY = "your_key_here"
```

---

## Project highlights

- Real AQI forecasting for Sargodha
- 3-day horizon modeling
- Hopsworks-based production model registry integration
- SHAP explainability in the dashboard
- operational retraining logic for model improvement
- Streamlit-ready deployment 

---

## Best interpretation of the final design

The complete system is best viewed as a practical ML operations pipeline:

- data acquisition from external APIs
- feature engineering from raw environmental signals
- model training and version tracking
- deployment in a front-end dashboard
- explanation layer for decision transparency
- scheduled/retraining automation with caution around version drift

This combination makes the project useful not only as a forecast model, but also as an explainable, monitorable environmental analytics application.

---

## Conclusion

This project demonstrates a full workflow for short-term air quality forecasting in a local context using real-world public data. It combines forecasting, explainability, operational deployment, and model versioning in a single end-to-end solution.

The core model set is built around Gradient Boosting Regressors, and the project is structured to support local experimentation, retraining, deployment, and model registry tracking.

---

## Useful next steps

- add a direct link to the final deployed Streamlit app in this README
- add a screenshot of the SHAP panel itself
- add a release/version badge
- add a changelog section for retraining updates
- pin exact dependency versions to reduce Hopsworks trigger failures

- the API key is valid
- the Hopsworks project is correct
- the model versions exist in the registry
- the model artifact includes `features.pkl` and a valid model object

### Open-Meteo requests fail

Verify internet access and confirm the API is responding properly. Some fallback values are used for the live cards, but forecast execution will fail if the required historical or model data is unavailable.

## Validation Commands

```bash
python -m py_compile app.py retrain.py
python -m pip check
python -c "import hopsworks, joblib, numpy, pandas, requests, sklearn; print('Dependencies loaded successfully')"
```

## Security Notes

- Do not commit `.env` files or secret credentials
- Do not upload API keys to GitHub in plain text
- Use GitHub secrets or Streamlit secrets for deployment
- Rotate exposed keys immediately if they are accidentally shared

## License

This project is intended for educational and research use. Please use it responsibly and ensure that any external API or cloud credentials remain secure.

## Project Status

This repository is actively used for AQI forecasting, Hopsworks model registry integration, and Streamlit-based visualization.

## Contact

For questions or collaboration inquiries, please reach out through the repository issues or the project maintainer profile on GitHub.
