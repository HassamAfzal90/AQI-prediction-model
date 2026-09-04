# Sargodha AQI Prediction Model

<img width="596" height="335" alt="Sargodha AQI Banner" src="https://github.com/user-attachments/assets/ec2b0a68-c7d4-476e-82bb-49f0fc43c9c6" />

A production-style air quality forecasting project for Sargodha, Punjab, Pakistan. The system combines historical weather and AQI data from Open-Meteo with a Hopsworks-managed machine learning pipeline and presents a clean 3-day AQI forecast dashboard in Streamlit.

---

## Table of Contents

1. [Project Summary](#project-summary)
2. [Live App](#live-app)
3. [Dashboard Preview](#dashboard-preview)
4. [Why This Project Matters](#why-this-project-matters)
5. [Core Functionality](#core-functionality)
6. [Architecture Overview](#architecture-overview)
7. [Repository Structure](#repository-structure)
8. [Full Technology Stack — In Detail](#full-technology-stack--in-detail)
9. [Data Sources — In Detail](#data-sources--in-detail)
10. [Notebook Pipeline (`new.ipynb`) — Full Detailed Walkthrough](#notebook-pipeline-newipynb--full-detailed-walkthrough)
11. [Feature Engineering — In Depth](#feature-engineering--in-depth)
12. [Model Details](#model-details)
13. [Accuracy and Validation Summary](#accuracy-and-validation-summary)
14. [Retraining Pipeline (`retrain.py`) — Full Detailed Walkthrough](#retraining-pipeline-retrainpy--full-detailed-walkthrough)
15. [Streamlit Dashboard (`app.py`) — Full Detailed Walkthrough](#streamlit-dashboard-apppy--full-detailed-walkthrough)
16. [Hopsworks Integration — In Detail](#hopsworks-integration--in-detail)
17. [GitHub Actions Automation — In Detail](#github-actions-automation--in-detail)
18. [Environment Setup](#environment-setup)
19. [Operational Notes and Version-Drift Handling](#operational-notes-and-version-drift-handling)
20. [Project Highlights](#project-highlights)
21. [Best Interpretation of the Final Design](#best-interpretation-of-the-final-design)
22. [Conclusion](#conclusion)

---

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

The app is meant for monitoring pollution risk, highlighting vulnerable periods, and making AQI forecasts more interpretable for end users. It is built as an end-to-end MLOps-style system rather than a one-off notebook experiment — meaning data ingestion, feature engineering, training, versioning, deployment, and retraining are all treated as first-class, connected stages of the same pipeline.

---

## Live App

Live deployment link:

* **Streamlit App:** [Live AQI Dashboard](https://aqi-prediction-model-9t6xkwfuyapxj7jtkdv38x.streamlit.app/)

---

## Dashboard Preview

<img width="1600" height="900" alt="Dashboard Preview" src="https://github.com/user-attachments/assets/1abfff0d-85ae-4635-b2b8-fa1dd39410d5" />

<img width="1600" height="900" alt="Trend Forecast" src="https://github.com/user-attachments/assets/0225cbbd-d362-4238-a1d3-18a81d81d780" />

---

## Why This Project Matters

Air quality is highly sensitive to both weather and recent pollution patterns. In cities like Sargodha, AQI can change quickly due to:

- PM2.5 and PM10 fluctuations
- temperature and humidity trends
- wind conditions
- seasonal behavior
- recent pollution persistence

Sargodha, like many mid-sized cities in Punjab, does not have dense, continuous, publicly accessible ground-level air quality monitoring in the way major capitals do. This makes a data-driven forecasting approach especially valuable — the model has to reconstruct pollution behavior from weather signals, pollutant history, and seasonal patterns rather than rely on a dense sensor network.

This project uses historical patterns and future weather signals to forecast AQI up to 3 days ahead so that users — especially vulnerable groups such as children, elderly individuals, and people with respiratory conditions — can act early when pollution is expected to worsen.

---

## Core Functionality

- 3-day AQI forecast for each day horizon (Day 1, Day 2, Day 3), each served by its own dedicated model
- live weather metrics from Open-Meteo (temperature, humidity, wind, pressure, precipitation)
- AQI risk classification and health advisory labels (e.g. Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous)
- hazard banners for moderate and high pollution cases, so risk is visually obvious at a glance
- historical AQI trend charts plus model forecast overlay, so users can see how the forecast continues from recent real values
- feature-based explanation using SHAP values, so the "why" behind a prediction is visible, not just the number
- local `.env` support plus Streamlit Cloud secret management, so the same codebase runs identically in local development and in production

---

## Architecture Overview

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

At a conceptual level, the system has three loosely-coupled layers:

1. **Data layer** — Open-Meteo APIs act as the single source of truth for both historical weather/AQI and forward-looking weather forecasts.
2. **Modeling layer** — feature engineering and three independently trained Gradient Boosting Regressors, one per forecast horizon, registered and versioned in Hopsworks.
3. **Presentation layer** — the Streamlit app, which is a thin, stateless client that pulls the latest registered models and the latest live data on every run, so the dashboard always reflects the most current model and the most current weather.

---

## Repository Structure

```text
.
├── app.py                          # Streamlit interactive dashboard
├── retrain.py                      # Retraining pipeline & Hopsworks model registry logic
├── new.ipynb                       # Notebook experimentation and EDA
├── requirements.txt                # Production app dependencies
├── requirements-retrain.txt        # Retraining pipeline dependencies
├── README.md                       # Project documentation
├── .env                            # Local environment variables (Ignored by Git)
├── .gitignore                      # Git ignore rules
├── docs/
│   ├── dashboard-preview.png       # Dashboard screenshot
│   └── trend-forecast.png          # Forecast chart screenshot
├── sargodha_raw_data_3yrs (5).csv  # Raw historical dataset
├── sargodha_features_daily_v2.csv  # Processed feature-engineered dataset
├── retrain_artifacts/              # Locally generated model artifacts
└── .github/
    └── workflows/
        └── retrain.yml             # GitHub Actions weekly retraining workflow
```

---

## Full Technology Stack — In Detail

### Core Language and Runtime
- **Python 3.12+** — the entire pipeline, from data ingestion to the dashboard, is written in Python for consistency between the notebook, retraining script, and app.

### Data Handling
- **Pandas** — used throughout for loading CSVs, resampling hourly data into daily aggregates, merging weather and AQI tables, and building the final feature matrix.
- **NumPy** — underlies numerical operations such as rolling-window statistics, seasonal sine/cosine transforms, and array manipulation before feeding data into scikit-learn.

### Machine Learning
- **scikit-learn** — provides the `GradientBoostingRegressor` implementation used for all three horizon models, plus train/test splitting, metric functions (RMSE, MAE, R²), and preprocessing utilities.
- **Joblib** — used to serialize (`.pkl`) and deserialize trained models and feature lists, both for local `retrain_artifacts/` storage and for uploading to the Hopsworks Model Registry.

### Explainability
- **SHAP (SHapley Additive exPlanations)** — computes per-feature contribution values for each prediction, letting the notebook and the dashboard both show *which* weather/pollution variables are pushing the forecasted AQI up or down, and by how much.

### Visualization
- **Plotly** — powers the interactive charts in the Streamlit dashboard: historical AQI trend lines, forecast overlays, and SHAP-based feature contribution charts. Plotly is chosen over static plotting libraries because it supports hover tooltips and zooming directly inside the Streamlit app.

### Web App / Dashboard
- **Streamlit** — the front-end framework for the entire user-facing dashboard. It handles layout, live re-computation on each page load, secrets management (via `st.secrets` in the cloud and `.env` locally), and rendering of Plotly charts and hazard banners.

### Model Registry / MLOps
- **Hopsworks** — acts as the model registry and version-control system for the three GBR models. Each model version is tagged with its validation metrics, which lets `retrain.py` compare a newly trained model against the currently deployed one before deciding whether to promote it.

### Automation
- **GitHub Actions** — schedules and runs the weekly retraining workflow (`retrain.yml`) in a clean CI environment, ensuring retraining happens automatically without manual intervention, and supports on-demand runs via `workflow_dispatch`.

### Deployment
- **Streamlit Community Cloud** — hosts the public-facing dashboard, pulling the latest code from GitHub and reading `HOPSWORKS_API_KEY` from its secrets manager.

### External Data APIs
- **Open-Meteo Archive API** — historical weather data (temperature, humidity, wind, pressure, precipitation) going back years, used to build the training dataset.
- **Open-Meteo Forecast API** — forward-looking weather data used both as live weather metrics in the dashboard and as "future weather" features for the Day 1–3 models.
- **Open-Meteo Air Quality API** — historical and current AQI and pollutant concentrations (PM2.5, PM10, CO, NO₂, SO₂, O₃), the backbone of both the target variable and several engineered features.

### ML Libraries Explored During Experimentation (in `new.ipynb`)
- **Gradient Boosting Regressor** — the final chosen model family, selected for its strength on structured/tabular data with non-linear feature interactions.
- **Random Forest Regressor** — evaluated as a bagging-based alternative; generally more robust to noise but slightly less precise than GBR in this dataset's benchmarks.
- **Ridge / Linear baselines** — used as a sanity-check baseline to confirm that the more complex tree-based models were actually adding predictive value over a simple linear relationship between weather/pollution features and AQI.
- **SHAP explainability workflow** — applied on top of the winning GBR models to validate feature importance and confirm the model's behavior was physically sensible (e.g. that PM2.5 lags and recent AQI persistence dominate short-horizon forecasts).

---

## Data Sources — In Detail

| Source | Data Provided | Role in Project |
|---|---|---|
| Open-Meteo Archive API | Historical hourly temperature, humidity, wind speed, pressure, precipitation (multi-year) | Builds the historical weather half of the training dataset |
| Open-Meteo Air Quality API | Historical + current PM2.5, PM10, AQI, CO, NO₂, SO₂, O₃ | Builds the target variable (AQI) and pollutant-based engineered features |
| Open-Meteo Forecast API | Forward-looking weather for the next several days | Supplies "future weather" features used by the Day 1, 2, and 3 models at prediction time, and drives the live weather metrics shown in the dashboard |

All three sources are queried live at forecast time in `app.py`, and in bulk historical form in `new.ipynb` / `retrain.py` for training.

---

## Notebook Pipeline (`new.ipynb`) — Full Detailed Walkthrough

The notebook is the experimental backbone of the project — every modeling decision used in production (feature set, model family, hyperparameters) was first validated here before being ported into `retrain.py`. The full pipeline inside the notebook proceeds as follows:

### 1. Data Loading
- Loads the raw hourly weather and AQI dataset (`sargodha_raw_data_3yrs (5).csv`), spanning roughly three years of hourly records pulled from Open-Meteo's Archive and Air Quality APIs.
- Performs initial sanity checks: null counts, data type checks, date range verification, and duplicate timestamp detection.

### 2. Hourly-to-Daily Aggregation
- Converts hourly records into daily aggregated values, since the forecasting target is a daily AQI figure rather than an hourly one.
- Aggregation choices are feature-specific: for example, AQI and pollutant concentrations are aggregated using daily mean/max, while precipitation is summed and wind speed is averaged.
- This step is essential because hourly noise (e.g. a single high-pollution hour during a traffic peak) would otherwise distort a daily forecasting target if not properly smoothed.

### 3. Feature Engineering
The notebook builds a rich feature set on top of the daily aggregated table (fully detailed in the [Feature Engineering](#feature-engineering--in-depth) section below), including:
- lagged AQI, PM2.5, and temperature values
- rolling statistics (mean, max, min, std) over multiple window sizes
- calendar and seasonal signals
- forecast-horizon-specific future weather variables

### 4. Train/Test Split
- Splits the engineered dataset into train and test sets, using a time-aware split (rather than a random shuffle) so that the model is always validated on data that comes *after* the training period — this avoids information leakage from future days into the training set, which is a common and serious mistake in time-series forecasting.

### 5. Model Training and Comparison
- Trains multiple model families side by side for each forecast horizon (Day 1, Day 2, Day 3): Gradient Boosting Regressor, Random Forest Regressor, and Ridge/linear baselines.
- Each model is trained independently per horizon because the relevant features and signal strength differ meaningfully between a 1-day-ahead and a 3-day-ahead forecast.

### 6. Metric Computation
- Computes RMSE, MAE, and R² for every trained model on the held-out test set.
- These metrics are compared across model families to select the best-performing option per horizon — Gradient Boosting Regressor came out ahead consistently, which is why it became the production model family.

### 7. Saving the Strongest Artifacts
- Once the best-performing model per horizon is identified, its trained object is serialized with Joblib and saved locally, alongside the exact list of feature names used at training time (critical, since the feature vector must be reconstructed identically at inference time in `app.py`).

### 8. Exporting for Registry Deployment
- Model objects and feature name lists are exported in a format compatible with the Hopsworks Model Registry upload logic later used in `retrain.py`, so that the notebook's validated models and the production retraining pipeline stay perfectly aligned.

### 9. SHAP Explainability Generation
- Runs SHAP's `TreeExplainer` (well-suited to Gradient Boosting models) on the trained models to compute per-feature contribution values.
- Generates summary plots and dependency plots showing which features push AQI up or down, and how strongly — for example, confirming that recent AQI lags and PM2.5 concentration dominate the Day 1 model, while seasonal and broader weather trend features become more influential for the Day 3 model.

### Main Notebook Contribution
The notebook was used to validate the modeling approach end-to-end and to confirm which features matter most for AQI prediction across different horizons. It also generates the explainability plots that show, visually, which variables push AQI upward or downward — this analysis directly informed which features were kept in the final production feature set used by `retrain.py` and `app.py`.

### SHAP Explainability Flow (Notebook → App)
The notebook first establishes SHAP on the trained model and analyses feature contributions offline, as part of the experimentation and validation process. The Streamlit app then mirrors this exact approach in production: at prediction time, it runs SHAP on the live feature vector against the currently loaded model, so end users see a real-time explanation of *their* specific forecast rather than a static, pre-computed example.

---

## Feature Engineering — In Depth

### Raw Inputs (from Open-Meteo)
- temperature
- relative humidity
- wind speed
- pressure
- precipitation
- PM10
- PM2.5
- AQI
- carbon monoxide (CO)
- nitrogen dioxide (NO₂)
- sulphur dioxide (SO₂)
- ozone (O₃)

### Engineered Features

**Lag features** — capture persistence and short-term momentum in pollution levels:
- lagged AQI values at 1, 2, 3, 5, 7, and 14 days
- lagged PM2.5 values
- lagged temperature values

**Rolling window statistics** — capture medium-term trend and volatility:
- rolling mean, max, min, and standard deviation of AQI over 3, 7, 14, and 30-day windows
- these features let the model distinguish between "AQI has been consistently high" versus "AQI just spiked once," which behave very differently going forward

**Calendar and seasonal signals** — capture recurring, periodic patterns:
- month, day of week, day of year, and a weekend flag
- seasonal sine/cosine terms (cyclical encoding of day-of-year), which allow the model to understand that day 365 and day 1 are adjacent in the seasonal cycle, something a raw day-of-year integer cannot represent on its own

**Future weather variables** — the forward-looking signal that makes multi-day forecasting possible:
- temperature, humidity, wind, pressure, and precipitation forecasts for the *target* day, pulled from the Open-Meteo Forecast API
- without these, the model would only be able to extrapolate from past pollution patterns; including forecasted weather lets it react to, for example, an incoming high-wind day that is expected to disperse pollutants and lower AQI

This combined feature space allows the models to learn both **persistence** (how pollution levels carry over from recent days) and **periodicity** (how pollution behaves seasonally), while also reacting to **forward-looking weather conditions** that a purely historical model would miss entirely.

---

## Model Details

The final production-style deployment uses three separate models, one for each forecast horizon:

<img width="1206" height="382" alt="Model Details" src="https://github.com/user-attachments/assets/dc708f61-3838-4789-aacb-9e27fcaef57e" />

- Day 1 model: `sargodha_aqi_gbr_day1`
- Day 2 model: `sargodha_aqi_gbr_day2`
- Day 3 model: `sargodha_aqi_gbr_day3`

### Model Type
The trained forecasting models are Gradient Boosting Regressors (GBR), selected because they perform well on structured tabular environmental data and handle non-linear interactions between weather and AQI variables — for example, the way wind speed's effect on AQI changes depending on current PM2.5 concentration, which a linear model cannot capture but a tree-based ensemble can.

### Model Scenarios

The project explicitly trains and supports these use cases:

1. **Day 1 forecast**
   - uses recent AQI persistence as the dominant signal
   - includes short-term weather and pollutant patterns
   - focuses on immediate next-day pollution risk, where recent history is the strongest predictor

2. **Day 2 forecast**
   - uses slightly longer lag patterns, since the most recent day's data is one step further removed from the target
   - incorporates forecast weather and previous AQI behavior in roughly equal measure
   - useful for short-range operational planning (e.g. deciding whether outdoor activity is advisable two days out)

3. **Day 3 forecast**
   - captures broader trend continuity and seasonality, since short-term persistence weakens at this distance
   - leans more heavily on future weather drivers and seasonal/calendar features for the 3-day horizon
   - useful for early warnings and public health planning, giving more lead time to act

Training a **separate model per horizon**, rather than one model with a "days ahead" input, was a deliberate design choice — it lets each model specialize its feature importance and hyperparameters to the signal characteristics of its specific horizon.

---

## Accuracy and Validation Summary

<img width="1239" height="360" alt="Accuracy Summary" src="https://github.com/user-attachments/assets/58435be2-c51c-4f82-bd5e-435b52afd5c2" />

The notebook experiments and model registry workflow report the following approximate validation metrics for the GBR models:

| Horizon | Model | RMSE | R² | Notes |
|---|---|---:|---:|---|
| Day 1 | Gradient Boosting Regressor | 12.15 | 0.806 | Good short-term predictive strength |
| Day 2 | Gradient Boosting Regressor | 23.08 | 0.719 | Best overall R² among the three horizons |
| Day 3 | Gradient Boosting Regressor | 23.24 | 0.710 | Stable 3-day trend forecasting |

These values indicate that the model is reasonably capable of carrying short-term AQI dynamics, with the Day 2 configuration being the strongest in the recorded benchmark run. It is expected and normal for RMSE to increase and R² to decrease slightly as the forecast horizon extends, since uncertainty compounds the further out a forecast reaches — the Day 1 model benefits the most from strong AQI persistence, while Day 2 and Day 3 rely progressively more on weather-driven and seasonal signal.

---

## Retraining Pipeline (`retrain.py`) — Full Detailed Walkthrough

The project includes a dedicated automated retraining pipeline that runs every Sunday and registers a new Hopsworks model version only when the retrained model is actually better than what's currently deployed.

### Automated Schedule
The GitHub Actions workflow in `.github/workflows/retrain.yml` is configured with a weekly cron trigger:

- `0 0 * * 0` = every Sunday at 00:00 UTC
- manual runs are also supported via `workflow_dispatch`, so retraining can be triggered on demand without waiting for the schedule

### Retraining Flow — Step by Step

1. **Load raw historical weather and AQI data** — pulls the latest available historical records so that the most recent week's data is included in every retraining cycle, keeping the model current.
2. **Build the daily feature dataset** — repeats the same hourly-to-daily aggregation logic validated in the notebook, ensuring the retrained model sees data in exactly the same shape and structure as the original training run.
3. **Create engineered features matching the notebook pipeline** — lags, rolling stats, calendar/seasonal signals, and future weather variables are recomputed identically to how they were built in `new.ipynb`, so there is no drift between how the notebook validated the approach and how production retrains it.
4. **Train separate Gradient Boosting models for Day 1, Day 2, and Day 3** — three independent training runs, each using the horizon-specific feature set.
5. **Compare the new metrics against the current Hopsworks model versions** — the script fetches the currently registered model's validation metric and compares it directly against the freshly retrained model's metric on the same evaluation basis.
6. **Register a new version only if the model improves on the existing metric baseline** — this is the key safeguard: a worse-performing model is never allowed to silently replace a better one in production.
7. **Save trained artifacts to `retrain_artifacts/`** — regardless of whether the model gets promoted to Hopsworks, the trained artifact is always saved locally for inspection, comparison, and rollback purposes.

### Important Model-Management Behavior

The script is intentionally conservative:

- it does not upload a new model if the metrics do not improve
- it keeps the registry history clean and versioned, avoiding registry bloat from marginal or regressive retrains
- it preserves rollback-friendly artifacts for comparison and inspection, so a human can always audit what a given week's retrain produced
- if Hopsworks login fails, it logs the issue and still saves the trained model locally instead of crashing — this means a transient credential or network issue during the scheduled GitHub Actions run does not result in a lost training cycle

### Artifact Names

The retraining system writes model artifacts like:

- `model.pkl` — the serialized Gradient Boosting Regressor for a given horizon
- `features.pkl` — the exact ordered list of feature names the model expects, used to rebuild the correct feature vector at inference time
- `retrain_artifacts/...` — the local directory where all of the above are stored per retraining run

These are then registered with Hopsworks by model name:

- `sargodha_aqi_gbr_day1`
- `sargodha_aqi_gbr_day2`
- `sargodha_aqi_gbr_day3`

This is the expected production behavior: the model is retrained automatically every Sunday, and a new improved version is saved to Hopsworks when the score is better than the current one — giving the system a continuously self-improving (but safely gated) model lifecycle.

---

## Streamlit Dashboard (`app.py`) — Full Detailed Walkthrough

The Streamlit app is the public user-facing dashboard, and ties every other part of the pipeline together at request time. On each run/load, it:

1. **Loads credentials** — reads the Hopsworks API key from `.env` (local development) or from Streamlit Cloud's secrets manager (production deployment), so the same code path works in both environments without modification.
2. **Retrieves live weather and AQI data** — calls the Open-Meteo Forecast and Air Quality APIs directly to get the most current conditions and the near-term weather forecast, rather than relying on any cached or stale data.
3. **Assembles feature vectors for each forecast day** — reconstructs the exact same lag, rolling, calendar, seasonal, and future-weather features used in training, in the exact same order defined by each horizon's saved `features.pkl`, for Day 1, Day 2, and Day 3 independently.
4. **Loads the corresponding model from Hopsworks** — fetches the currently registered (best-performing) version of `sargodha_aqi_gbr_day1`, `_day2`, and `_day3` from the Model Registry, so the dashboard is always serving the latest promoted model without needing a redeploy.
5. **Predicts AQI for the next 3 days** — runs each horizon's model against its respective feature vector.
6. **Renders charts, hazard banners, and explainability panels** — using Plotly for the historical trend + forecast overlay chart, conditional hazard banners based on AQI thresholds, and a SHAP-based breakdown of which features are driving each day's prediction.

### Local Run Command

```bash
streamlit run app.py
```

### Deployment Note
This project is designed for deployment on Streamlit Community Cloud. The app is expected to run when the repo is connected to a Streamlit app and the required secrets are configured.

---

## Hopsworks Integration — In Detail

Hopsworks serves as the **Model Registry** layer of this project — the single source of truth for "which model version is currently live." Its role includes:

- storing serialized model artifacts (via the Joblib-serialized `model.pkl` files) under three distinct registered model names, one per horizon
- storing each model version's associated validation metrics, so `retrain.py` can programmatically compare a new candidate model's score against the currently deployed version's score
- serving the currently-promoted model back to `app.py` at inference time, so the dashboard always reflects the latest approved model without requiring a manual redeploy of the Streamlit app itself
- maintaining a clean, auditable version history, since only genuinely improved models are ever registered as new versions

---

## Operational Notes and Version-Drift Handling

The project depends on both Hopsworks and the broader ML stack, so the operating environment must stay consistent across the notebook, the retraining workflow, and the deployed app:

- Hopsworks API versions can change across environments
- model registry serialization may differ across scikit-learn / NumPy versions
- older pickles can fail if a newer environment is used without compatibility handling
- scheduled retraining requires a valid `HOPSWORKS_API_KEY` and correct registry access

This project already includes compatibility handling to reduce breakage from older scikit-learn pickle formats, including a compatibility shim for the legacy `_loss` module in `app.py` — this specifically addresses a known breaking change where scikit-learn refactored its internal loss-function module path between versions, which would otherwise cause `joblib.load()` to fail when loading a model pickled under an older scikit-learn version into a newer runtime.

### Operational Requirements for the Scheduled Pipeline

The automated workflow is intended to run cleanly when:

- the GitHub Actions environment has the required Python and package versions
- the Hopsworks API key is valid and not expired
- the model registry is reachable from the workflow environment
- the retrained model actually improves the current baseline metric

When those conditions are met, the Sunday retrain runs as designed and saves improved models to Hopsworks. If no improvement is found, the script skips registration and keeps the new model only as a local artifact for review.

---

## GitHub Actions Automation — In Detail

The `.github/workflows/retrain.yml` workflow is the automation backbone that keeps the models current without manual effort:

- **Trigger:** a weekly cron schedule (`0 0 * * 0`, every Sunday at 00:00 UTC), plus a `workflow_dispatch` trigger for manual, on-demand runs
- **Environment setup:** installs the exact dependencies pinned in `requirements-retrain.txt`, keeping the retraining environment isolated and reproducible from the app's own dependency set in `requirements.txt`
- **Execution:** runs `retrain.py` end-to-end — data loading, feature engineering, training, metric comparison, and conditional Hopsworks registration
- **Secrets:** the `HOPSWORKS_API_KEY` is injected into the workflow via GitHub Actions' encrypted repository secrets, never hard-coded into the codebase
- **Failure handling:** because `retrain.py` is written to fail gracefully on registry connectivity issues (saving artifacts locally rather than crashing), a failed Hopsworks login does not fail the entire scheduled run silently without a trace — the model is still produced and stored as a workflow artifact for review

---

## Environment Setup

### App Dependencies

```bash
python -m pip install -r requirements.txt
```

### Retraining Dependencies

```bash
python -m pip install -r requirements-retrain.txt
```

### Hopsworks Secret Setup

Local `.env` file:

```dotenv
HOPSWORKS_API_KEY=your_key_here
```

Streamlit Cloud secret:

```toml
HOPSWORKS_API_KEY = "your_key_here"
```

---

## Project Highlights

- Real AQI forecasting for Sargodha, built on genuine multi-year historical data rather than synthetic examples
- 3-day horizon modeling with three independently specialized models rather than a single one-size-fits-all forecaster
- Hopsworks-based production model registry integration, with metric-gated promotion logic
- SHAP explainability in the dashboard, giving users a transparent "why" behind every forecast
- operational retraining logic for model improvement, scheduled automatically via GitHub Actions
- Streamlit-ready deployment, running identically from local `.env` config and from Streamlit Cloud secrets

---

## Best Interpretation of the Final Design

The complete system is best viewed as a practical ML operations pipeline, not just a model:

- **data acquisition** from external APIs (Open-Meteo Archive, Forecast, and Air Quality endpoints)
- **feature engineering** from raw environmental signals (lags, rolling stats, seasonality, forecast weather)
- **model training and version tracking** (three horizon-specific Gradient Boosting Regressors, registered in Hopsworks)
- **deployment in a front-end dashboard** (Streamlit, reading live data and the latest registered models on every load)
- **explanation layer for decision transparency** (SHAP-based feature attribution surfaced directly in the UI)
- **scheduled/retraining automation with caution around version drift** (weekly GitHub Actions run, metric-gated promotion, and compatibility shims for library version mismatches)

This combination makes the project useful not only as a forecast model, but also as an explainable, monitorable environmental analytics application that can keep improving itself safely over time.

---

## Conclusion

This project demonstrates a full workflow for short-term air quality forecasting in a local context using real-world public data. It combines forecasting, explainability, operational deployment, and model versioning in a single end-to-end solution.

The core model set is built around Gradient Boosting Regressors, and the project is structured to support local experimentation (via the notebook), automated retraining (via `retrain.py` and GitHub Actions), production model versioning (via Hopsworks), and a transparent, user-facing forecast experience (via the Streamlit dashboard).
