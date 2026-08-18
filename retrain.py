import os
import pathlib
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib3.util.retry import Retry

import hopsworks


def load_env_file(env_path: pathlib.Path):
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


load_env_file(pathlib.Path(__file__).parent / ".env")

RAW_CSV = pathlib.Path("sargodha_raw_data_3yrs (5).csv")
FEATURE_CSV = pathlib.Path("sargodha_features_daily_v2.csv")
MODEL_ARTIFACT_ROOT = pathlib.Path("retrain_artifacts")

MODEL_INFO = [
    ("sargodha_aqi_gbr_day1", 4, "Day 1"),
    ("sargodha_aqi_gbr_day2", 3, "Day 2"),
    ("sargodha_aqi_gbr_day3", 3, "Day 3"),
]

RAW_START_CUTOFF = datetime(2022, 7, 29).date()
FORECAST_WEATHER_COLS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
]


def create_requests_session(retries=4, backoff_factor=1.0):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_raw_data(lat, lon, years=3):
    end_date = datetime.utcnow().date() - timedelta(days=2)
    start_date = end_date - timedelta(days=years * 365)
    if start_date < RAW_START_CUTOFF:
        start_date = RAW_START_CUTOFF

    session = create_requests_session()

    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation",
        "timezone": "auto",
    }
    weather_res = session.get(weather_url, params=weather_params, timeout=60)
    weather_res.raise_for_status()
    weather_json = weather_res.json()
    if "hourly" not in weather_json:
        raise RuntimeError(f"Weather history not available: {weather_json}")

    weather_df = pd.DataFrame(weather_json["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": "auto",
    }
    aqi_res = session.get(aqi_url, params=aqi_params, timeout=60)
    aqi_res.raise_for_status()
    aqi_json = aqi_res.json()
    if "hourly" not in aqi_json:
        raise RuntimeError(f"AQI history not available: {aqi_json}")

    aqi_df = pd.DataFrame(aqi_json["hourly"])
    aqi_df["time"] = pd.to_datetime(aqi_df["time"])

    raw_df = pd.merge(weather_df, aqi_df, on="time", how="outer")
    raw_df = raw_df.sort_values("time").reset_index(drop=True)
    raw_df = raw_df.rename(columns={"us_aqi": "AQI"})

    if raw_df["AQI"].isna().all():
        raise RuntimeError("Fetched raw data does not contain AQI values.")

    return raw_df


def load_raw_data():
    if RAW_CSV.exists():
        print(f"Loading raw dataset from {RAW_CSV}")
        df = pd.read_csv(RAW_CSV, parse_dates=["time"])
    else:
        print("Raw CSV not found. Fetching raw data from Open-Meteo.")
        df = fetch_raw_data(32.0836, 72.6711, years=3)
        df.to_csv(RAW_CSV, index=False)
        print(f"Saved raw dataset to {RAW_CSV}")
    return df


def build_feature_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.set_index("time")

    agg_dict = {
        "temperature_2m": ["mean", "max", "min"],
        "relative_humidity_2m": "mean",
        "pressure_msl": "mean",
        "wind_speed_10m": "mean",
        "precipitation": "sum",
        "pm10": "mean",
        "pm2_5": "mean",
        "carbon_monoxide": "mean",
        "nitrogen_dioxide": "mean",
        "sulphur_dioxide": "mean",
        "ozone": "mean",
        "AQI": ["mean", "max"],
    }

    daily = df.resample("D").agg(agg_dict)
    daily.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in daily.columns]
    daily = daily.rename(columns={"AQI_mean": "AQI"})

    daily["month"] = daily.index.month
    daily["day_of_week"] = daily.index.dayofweek
    daily["day_of_year"] = daily.index.dayofyear
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily["doy_sin"] = np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * daily["day_of_year"] / 365.25)

    for lag in [1, 2, 3, 5, 7, 14]:
        daily[f"AQI_lag{lag}"] = daily["AQI"].shift(lag)
        daily[f"pm2_5_lag{lag}"] = daily["pm2_5_mean"].shift(lag)
        daily[f"temperature_2m_lag{lag}"] = daily["temperature_2m_mean"].shift(lag)

    for window in [3, 7, 14, 30]:
        daily[f"AQI_rolling_{window}d_mean"] = daily["AQI"].shift(1).rolling(window).mean()
        daily[f"AQI_rolling_{window}d_max"] = daily["AQI"].shift(1).rolling(window).max()
        daily[f"AQI_rolling_{window}d_min"] = daily["AQI"].shift(1).rolling(window).min()
        daily[f"AQI_rolling_{window}d_std"] = daily["AQI"].shift(1).rolling(window).std()

    for h in [1, 2, 3]:
        daily[f"temp_future_h{h}"] = daily["temperature_2m_mean"].shift(-h)
        daily[f"humidity_future_h{h}"] = daily["relative_humidity_2m_mean"].shift(-h)
        daily[f"wind_future_h{h}"] = daily["wind_speed_10m_mean"].shift(-h)

    daily["target_day1"] = daily["AQI"].shift(-1)
    daily["target_day2"] = daily["AQI"].shift(-2)
    daily["target_day3"] = daily["AQI"].shift(-3)

    daily_v2 = daily.dropna().reset_index()

    daily_v2.to_csv(FEATURE_CSV, index=False)
    print(f"Saved feature dataset to {FEATURE_CSV} ({daily_v2.shape[0]} rows)")

    return daily_v2


def extract_metrics(model_meta) -> dict:
    if model_meta is None:
        return {}
    return getattr(model_meta, "training_metrics", {}) or getattr(model_meta, "_training_metrics", {}) or {}


def choose_best_metrics(model_versions: list) -> tuple[dict, int | None]:
    best_metrics = {}
    best_version = None
    for version_meta in model_versions:
        metrics = extract_metrics(version_meta)
        if not metrics:
            continue

        if not best_metrics:
            best_metrics = metrics
            best_version = getattr(version_meta, "version", None)
            continue

        current_better = False
        if "r2" in metrics and "r2" in best_metrics:
            if metrics["r2"] > best_metrics["r2"]:
                current_better = True
            elif metrics["r2"] == best_metrics["r2"] and "rmse" in metrics and "rmse" in best_metrics:
                current_better = metrics["rmse"] < best_metrics["rmse"]
        elif "rmse" in metrics and "rmse" in best_metrics:
            current_better = metrics["rmse"] < best_metrics["rmse"]

        if current_better:
            best_metrics = metrics
            best_version = getattr(version_meta, "version", None)

    return best_metrics, best_version


def improved(new_metrics: dict, curr_metrics: dict) -> bool:
    if not curr_metrics:
        return True

    has_rmse = "rmse" in new_metrics and "rmse" in curr_metrics
    has_r2 = "r2" in new_metrics and "r2" in curr_metrics

    if has_rmse and has_r2:
        return new_metrics["rmse"] < curr_metrics["rmse"] and new_metrics["r2"] >= curr_metrics["r2"]
    if has_rmse:
        return new_metrics["rmse"] < curr_metrics["rmse"]
    if has_r2:
        return new_metrics["r2"] > curr_metrics["r2"]
    return False


def create_model_metadata(mr, model_name: str, metrics: dict, example_row: pd.DataFrame):
    if hasattr(mr, "python"):
        return mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description="Notebook-aligned retrained model for Sargodha AQI forecast.",
            input_example=example_row,
        )
    return mr._sklearn.create_model(
        name=model_name,
        metrics=metrics,
        description="Notebook-aligned retrained model for Sargodha AQI forecast.",
        input_example=example_row,
    )


def save_model_version(model_meta, artifacts_dir: pathlib.Path):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return model_meta.save(str(artifacts_dir))


def train_and_upload_models(daily_v2: pd.DataFrame, mr):
    split_idx = int(len(daily_v2) * 0.80)
    train = daily_v2.iloc[:split_idx]
    test = daily_v2.iloc[split_idx:]

    success_count = 0
    for h, target_col in zip([1, 2, 3], ["target_day1", "target_day2", "target_day3"]):
        model_name = f"sargodha_aqi_gbr_day{h}"
        feat_cols = [
            c for c in daily_v2.columns
            if c not in ["time", "target_day1", "target_day2", "target_day3"]
            and not c.startswith("temp_future_")
            and not c.startswith("humidity_future_")
            and not c.startswith("wind_future_")
        ] + [
            f"temp_future_h{h}",
            f"humidity_future_h{h}",
            f"wind_future_h{h}",
        ]

        X_train = train[feat_cols]
        y_train = train[target_col]
        X_test = test[feat_cols]
        y_test = test[target_col]

        # Optimized hyperparameters per horizon
        if h == 1:
            model = GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=4,
                random_state=42,
            )
        elif h == 2:
            model = GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=4,
                random_state=42,
            )
        else:  # h == 3
            model = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.015,
                max_depth=7,
                subsample=0.9,
                random_state=42,
            )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }

        print(f"Horizon day{h} metrics: {metrics}")

        versions = mr.get_models(model_name)
        current_metrics, current_version = choose_best_metrics(versions)
        print(f"Best existing metrics for {model_name} v{current_version}: {current_metrics}")

        if improved(metrics, current_metrics):
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            artifacts_dir = MODEL_ARTIFACT_ROOT / f"{model_name}_{timestamp}"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, artifacts_dir / "model.pkl")
            joblib.dump(feat_cols, artifacts_dir / "features.pkl")

            example_input = X_test.iloc[[0]]
            model_obj = create_model_metadata(mr, model_name, metrics, example_input)
            saved_meta = save_model_version(model_obj, artifacts_dir)
            print(f"Saved improved model for {model_name} as registry version {saved_meta.version}.")
            success_count += 1
        else:
            print(f"Skipping upload for {model_name}; no improvement over current metrics.")

    return success_count


def run_retrain():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is required in the environment.")

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=api_key,
        project="colab",
    )
    mr = project.get_model_registry()

    raw_df = load_raw_data()
    daily_v2 = build_feature_dataframe(raw_df)

    print("Training and uploading models using notebook pipeline...")
    success_count = train_and_upload_models(daily_v2, mr)

    if success_count == 0:
        print("No model versions were registered because no improvement was found.")
    else:
        print(f"{success_count} new model version(s) registered.")


if __name__ == "__main__":
    run_retrain()
