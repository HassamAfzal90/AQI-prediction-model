
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hopsworks
import os
import re
import sys
import types
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta

st.set_page_config(page_title="Sargodha AQI Live Forecast", layout="wide")


def create_requests_session(retries=3, backoff_factor=1.0):
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

# Compatibility shim for old scikit-learn pickles that reference the legacy `_loss` module.
def ensure_sklearn_loss_compat():
    try:
        import sklearn._loss._loss as sklearn_loss
        if "_loss" not in sys.modules:
            _loss_module = types.ModuleType("_loss")
            for attr in dir(sklearn_loss):
                if not attr.startswith("_"):
                    setattr(_loss_module, attr, getattr(sklearn_loss, attr))
            sys.modules["_loss"] = _loss_module
    except Exception:
        pass

ensure_sklearn_loss_compat()

st.title("🌬️ Sargodha AQI 3-Day Live Forecast System")
st.markdown(f"**Current Date:** {datetime.now().strftime('%A, %d %B %Y')} | **Location:** Sargodha, Punjab")

# Sidebar Configuration
st.sidebar.header("🔑 Hopsworks Connection")
api_key = st.sidebar.text_input("Hopsworks API Key", type="password")

# Sargodha Coordinates
LATITUDE = 32.0836
LONGITUDE = 72.6711

# Helper function to extract predict-capable model object
def extract_model(obj):
    if hasattr(obj, "predict"):
        return obj
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        for item in obj:
            if hasattr(item, "predict"):
                return item
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if hasattr(val, "predict"):
                return val
    return None

# --- 0. BUILD FEATURE VECTOR FOR EACH MODEL HORIZON ---
@st.cache_data(ttl=86400)
def fetch_historical_daily_data(lat, lon, lookback_days=45):
    end_date = datetime.utcnow().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days)

    session = create_requests_session(retries=4, backoff_factor=1.0)
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto",
    }
    w_res = session.get(weather_url, params=weather_params, timeout=60)
    w_res.raise_for_status()
    weather_json = w_res.json()
    if "hourly" not in weather_json:
        raise ValueError(f"Weather history not available: {weather_json}")

    weather_df = pd.DataFrame(weather_json["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.set_index("time").resample("D").mean()

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
        raise ValueError(f"AQI history not available: {aqi_json}")

    aqi_df = pd.DataFrame(aqi_json["hourly"])
    aqi_df["time"] = pd.to_datetime(aqi_df["time"])
    aqi_df = aqi_df.set_index("time").resample("D").mean()
    aqi_df = aqi_df.rename(columns={"us_aqi": "AQI"})

    combined = pd.concat([weather_df, aqi_df[["AQI", "pm2_5"]]], axis=1)
    combined = combined[["AQI", "pm2_5", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]]
    combined = combined.dropna()
    if len(combined) < 35:
        raise ValueError("Not enough historical daily data available for lag/rolling feature construction.")
    return combined

@st.cache_data(ttl=600)
def fetch_forecast_weather(lat, lon, horizon_days=5):
    session = create_requests_session(retries=4, backoff_factor=1.0)
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": horizon_days,
        "timezone": "auto",
    }
    forecast_res = session.get(forecast_url, params=forecast_params, timeout=60)
    forecast_res.raise_for_status()
    forecast_json = forecast_res.json()
    if "hourly" not in forecast_json:
        raise ValueError(f"Weather forecast not available: {forecast_json}")

    fc_df = pd.DataFrame(forecast_json["hourly"])
    fc_df["time"] = pd.to_datetime(fc_df["time"])
    fc_df = fc_df.set_index("time").resample("D").mean()
    return fc_df


def build_feature_vector(historical, forecast_daily, feature_names):
    latest_date = historical.index.max().date()
    latest_row = historical.loc[historical.index.date == latest_date]
    if latest_row.empty:
        raise ValueError("Latest historical date missing from assembled daily data.")
    latest_row = latest_row.iloc[0]

    def get_lag(name, lag):
        date = latest_date - timedelta(days=lag)
        if date not in historical.index.date:
            raise ValueError(f"Missing historical data for lag {lag} days ago ({date}).")
        return historical.loc[historical.index.date == date].iloc[0][name]

    def get_roll(stat, window):
        series = historical["AQI"].shift(1).rolling(window=window)
        value = getattr(series, stat)().iloc[-1]
        if pd.isna(value):
            raise ValueError(f"Missing rolling feature {stat} over {window} days.")
        return value

    feature_vector = []
    for feature in feature_names:
        if match := re.match(r"aqi_lag_(\d+)$", feature):
            feature_vector.append(get_lag("AQI", int(match.group(1))))
        elif match := re.match(r"pm25_lag_(\d+)$", feature):
            feature_vector.append(get_lag("pm2_5", int(match.group(1))))
        elif match := re.match(r"temp_lag_(\d+)$", feature):
            feature_vector.append(get_lag("temperature_2m", int(match.group(1))))
        elif match := re.match(r"aqi_roll_(mean|max|min|std)_(\d+)$", feature):
            stat = match.group(1)
            window = int(match.group(2))
            feature_vector.append(get_roll(stat, window))
        elif feature == "sin_day":
            day_of_year = latest_date.timetuple().tm_yday
            feature_vector.append(np.sin(2 * np.pi * day_of_year / 365.25))
        elif feature == "cos_day":
            day_of_year = latest_date.timetuple().tm_yday
            feature_vector.append(np.cos(2 * np.pi * day_of_year / 365.25))
        elif feature == "month":
            feature_vector.append(latest_date.month)
        elif feature == "dayofweek":
            feature_vector.append(latest_date.weekday())
        elif match := re.match(r"(temp|humidity|wind)_future_h(\d+)$", feature):
            kind = match.group(1)
            horiz = int(match.group(2))
            target_date = latest_date + timedelta(days=horiz)
            if target_date not in forecast_daily.index.date:
                raise ValueError(f"Forecast weather not available for {target_date}.")
            weather_row = forecast_daily.loc[forecast_daily.index.date == target_date].iloc[0]
            if kind == "temp":
                feature_vector.append(weather_row["temperature_2m"])
            elif kind == "humidity":
                feature_vector.append(weather_row["relative_humidity_2m"])
            else:
                feature_vector.append(weather_row["wind_speed_10m"])
        else:
            raise ValueError(f"Unknown feature name: {feature}")

    return np.array(feature_vector).reshape(1, -1)

# --- 1. OPEN-METEO API FETCHING WITH BETTER TIMEOUT & RETRY ---
@st.cache_data(ttl=600)
def fetch_live_weather_and_aqi():
    data = {"pm25": 45.0, "pm10": 90.0, "temp": 30.0, "humidity": 55.0, "wind": 10.0}
    
    # Weather Fetch
    session = create_requests_session(retries=3, backoff_factor=0.5)
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        }
        w_res = session.get(weather_url, params=weather_params, timeout=30)
        w_res.raise_for_status()
        w_json = w_res.json()
        curr_w = w_json.get("current", {})
        data["temp"] = curr_w.get("temperature_2m", 30.0)
        data["humidity"] = curr_w.get("relative_humidity_2m", 55.0)
        data["wind"] = curr_w.get("wind_speed_10m", 10.0)
    except Exception as e:
        st.warning(f"Weather API Warning: {e}. Fallback values used.")

    # Air Quality Fetch
    try:
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality"
        aqi_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "pm10,pm2_5",
        }
        a_res = session.get(aqi_url, params=aqi_params, timeout=30)
        a_res.raise_for_status()
        a_json = a_res.json()
        curr_a = a_json.get("current", {})
        data["pm25"] = curr_a.get("pm2_5", 45.0)
        data["pm10"] = curr_a.get("pm10", 90.0)
    except Exception as e:
        st.warning(f"Air Quality API Warning: {e}. Fallback values used.")

    return data

# --- 2. HOPSWORKS MODEL LOADING WITH SAFELY UNPACKED MODEL OBJECT ---
@st.cache_resource(show_spinner="Hopsworks Registry se Models download aur extract ho rahe hain...")
def load_all_models(key):
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=key,
        project="colab"
    )
    mr = project.get_model_registry()
    
    models_dict = {}
    model_info = [
        ("sargodha_aqi_gbr_day1", 4, "Day 1"),
        ("sargodha_aqi_gbr_day2", 3, "Day 2"),
        ("sargodha_aqi_gbr_day3", 3, "Day 3")
    ]
    
    for m_name, m_ver, label in model_info:
        model_meta = mr.get_model(m_name, version=m_ver)
        model_dir = model_meta.download()

        model_obj = None
        feature_names = None
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file == "features.pkl":
                    feature_names = joblib.load(file_path)
                elif file.endswith(".pkl") or file.endswith(".joblib"):
                    if file == "features.pkl":
                        continue
                    raw_obj = joblib.load(file_path)
                    extracted = extract_model(raw_obj)
                    if extracted is not None:
                        model_obj = extracted
            if model_obj is not None and feature_names is not None:
                break

        if model_obj is None:
            raise ValueError(f"No model object found in downloaded artifact for {m_name}.")
        if feature_names is None:
            raise ValueError(f"No features.pkl found for {m_name}; cannot build the model input vector.")

        models_dict[label] = {
            "model": model_obj,
            "features": feature_names,
        }
                
    return models_dict

models = None
if api_key:
    try:
        models = load_all_models(api_key)
        st.sidebar.success("✅ Day 1, 2, 3 Models Ready!")
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
else:
    st.info("👈 Left sidebar mein Hopsworks API Key enter karein.")

# --- 3. LIVE DATA UI ---
st.subheader("📡 Live Weather & Air Quality Data")
if st.button("🔄 Refresh Live API Data"):
    st.cache_data.clear()

live_data = fetch_live_weather_and_aqi()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current PM2.5", f"{live_data['pm25']} µg/m³")
m2.metric("Current PM10", f"{live_data['pm10']} µg/m³")
m3.metric("Temperature", f"{live_data['temp']} °C")
m4.metric("Humidity", f"{live_data['humidity']} %")
m5.metric("Wind Speed", f"{live_data['wind']} km/h")

st.divider()

# --- 4. PREDICTION & ALARMING ALERT SYSTEM ---
if st.button("🚀 Run 3-Day AQI Forecast", type="primary"):
    if not models or len(models) < 3:
        st.error("Models load nahi hue! Sidebar me API Key verify karein.")
    else:
        try:
            historical = fetch_historical_daily_data(LATITUDE, LONGITUDE)
            forecast_daily = fetch_forecast_weather(LATITUDE, LONGITUDE, horizon_days=5)

            preds = []
            for horizon_label, label in [("Day 1", "Day 1"), ("Day 2", "Day 2"), ("Day 3", "Day 3")]:
                model_info = models[label]
                model = model_info["model"]
                feature_names = model_info["features"]
                x_vec = build_feature_vector(historical, forecast_daily, feature_names)
                preds.append(float(model.predict(x_vec)[0]))

            today = datetime.now()
            dates = [(today + timedelta(days=i+1)).strftime("%Y-%m-%d (%A)") for i in range(3)]
            preds = [round(v, 2) for v in preds]

            df_res = pd.DataFrame({"Forecast Date": dates, "Predicted AQI": preds})

            st.subheader("📅 3-Day AQI Predictions")
            st.dataframe(df_res, use_container_width=True)
            st.line_chart(df_res.set_index("Forecast Date"))

            max_aqi = max(preds)
            if max_aqi > 200:
                st.error(f"🚨 **SEVERE HAZARDOUS ALARM!** Projected AQI will reach **{max_aqi}**. Air quality is very unhealthy/hazardous!")
            elif max_aqi > 150:
                st.warning(f"⚠️ **UNHEALTHY AIR QUALITY ALERT!** Projected AQI will reach **{max_aqi}**. Mask recommended.")
            else:
                st.success(f"✅ **ACCEPTABLE AIR QUALITY.** Maximum predicted AQI is **{max_aqi}**.")

        except Exception as err:
            st.error(f"Prediction Error: {err}")
