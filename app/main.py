import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
import io
from sklearn.preprocessing import StandardScaler
from random import randint
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_autorefresh import st_autorefresh
import altair as alt
import pydeck as pdk
import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload



# ---------- CONFIG ----------
working_dir = os.path.dirname(os.path.abspath(__file__))
CLASS_MODEL_PATH = working_dir + "/trained_models/classification_model.pkl"   # classification model (pickle)
REG_MODEL_PATH = working_dir + "/trained_models/regression_model.pkl"         # regression model (pickle)
CSV_PATH = working_dir + "/testing_data.csv"          # CSV with test rows (features only)
AUTO_REFRESH_MS = 300_000   # 5 minutes in milliseconds
HISTORY_CHART_LIMIT = 144   # rows shown on the trend charts
MAX_STORED_ROWS = 5000      # cap on the Drive CSV so download/upload stays fast as it grows


st.set_page_config(page_title="Kendira Forest Fire Prevention Dashboard", layout="wide")
st.title("Kendira Forest Fire Prevention Dashboard")

# ---------- Google Drive connection (keeps database.csv persistent across restarts/redeploys) ----------
DRIVE_FILE_ID = st.secrets["drive_file_id"]

@st.cache_resource
def get_drive_service():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

drive_service = get_drive_service()

def load_history_from_drive():
    """Download the current database.csv from Drive as a DataFrame."""
    try:
        request = drive_service.files().get_media(fileId=DRIVE_FILE_ID)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buffer.seek(0)
        return pd.read_csv(buffer, parse_dates=["timestamp"])
    except Exception as e:
        st.warning(f"Could not load history from Drive ({e}). Starting with empty history.")
        return pd.DataFrame()

def save_history_to_drive(df):
    """Overwrite database.csv on Drive with the given DataFrame."""
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    buffer.seek(0)
    media = MediaIoBaseUpload(buffer, mimetype="text/csv", resumable=False)
    drive_service.files().update(fileId=DRIVE_FILE_ID, media_body=media).execute()

# ---------- Load models using pickle ----------
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        st.stop()

classification_model = load_pickle(CLASS_MODEL_PATH)
regression_model = load_pickle(REG_MODEL_PATH)

# ---------- Load test dataset ----------
try:
    df_test = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"CSV file not found: {CSV_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Error reading CSV {CSV_PATH}: {e}")
    st.stop()

if df_test.shape[0] == 0:
    st.error("CSV file appears empty.")
    st.stop()

# ---------- Fit scaler once on the full dataset  ----------
scaler = StandardScaler()
scaler.fit(df_test.values)

# ---------- Sensors emplacements ----------
def get_kendira_sensors():
    sensors = [
        {"id": "S1", "lat": 36.540556, "lon": 5.027500},  # Central
        {"id": "S2", "lat": 36.567583, "lon": 5.027500},  # North
        {"id": "S3", "lat": 36.518033, "lon": 5.027500},  # South
        {"id": "S4", "lat": 36.540556, "lon": 5.061160},  # East
        {"id": "S5", "lat": 36.540556, "lon": 4.993840},  # West
        {"id": "S6", "lat": 36.558574, "lon": 5.049940},  # North-East
        {"id": "S7", "lat": 36.527042, "lon": 5.044330},  # South-East
        {"id": "S8", "lat": 36.524340, "lon": 5.007300},  # South-West
    ]
    return sensors

sensors = get_kendira_sensors()

# ---------- Last update display ----------
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

# ---------- Manual refresh button ----------
if st.button("Run next simulation (force update)"):
    st.session_state.last_update = time.time()

# ---------- Auto-refresh each 5 min ----------
refresh_count = st_autorefresh(interval=AUTO_REFRESH_MS, key="autorefresh")
if time.time() - st.session_state.last_update > (AUTO_REFRESH_MS / 1000.0):
    st.session_state.last_update = time.time()

# ------------ Logic ------------
results = []
for sensor in sensors:
    # Random test line
    idx = randint(0, len(df_test) - 1)
    sample = df_test.iloc[idx]
    X_raw = sample.values.reshape(1, -1)

    # Classification
    try:
        class_pred = classification_model.predict(X_raw)
        if hasattr(classification_model, "predict_proba"):
            class_prob = classification_model.predict_proba(X_raw)[:, 1][0]
            class_label = "Risk" if class_prob > 0.5 else "No risk"
        else:
            class_label = "Risk" if int(class_pred[0]) == 1 else "No risk"
            class_prob = None
    except Exception:
        class_label, class_prob = "Error", None

    # Regression
    try:
        X_scaled = scaler.transform(X_raw)
        reg_pred = regression_model.predict(X_scaled)
        reg_value = float(reg_pred[0])
    except Exception:
        reg_value = None

    results.append({
        "id": sensor["id"],
        "lat": sensor["lat"],
        "lon": sensor["lon"],
        "class_label": class_label,
        "reg_value": reg_value,
        "sample_idx": idx
    })

# ----------- Dashboard -----------
col_map, col_info = st.columns([1, 1.25])

with col_map:
    st.subheader("Forest map with sensor locations")

    map_df = pd.DataFrame({
        "lat": [s["lat"] for s in results],
        "lon": [s["lon"] for s in results],
        "id": [s["id"] for s in results]
    })

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius=150,
        pickable=True
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_text="id",
        get_size=16,
        get_color=[245, 245, 245, 255],
        get_alignment_baseline="'bottom'"
    )

    # View
    view_state = pdk.ViewState(
        latitude=map_df["lat"].mean(),
        longitude=map_df["lon"].mean(),
        zoom=12
    )

    st.pydeck_chart(pdk.Deck(layers=[layer, text_layer], initial_view_state=view_state))

    st.write("**Auto-refresh:** every 5 minutes (or force it with the button above).")

with col_info:
    st.subheader("Metrics sent by each sensor and corresponding prediction")

    df_results = pd.DataFrame(results)

    features = []
    for r in results:
        idx = r["sample_idx"]
        sample_features = df_test.iloc[idx].to_dict()
        sample_features["id"] = r["id"]
        features.append(sample_features)

    df_features = pd.DataFrame(features)

    df_full = pd.merge(df_results, df_features, on="id")

    ordered_cols = (
        ["id", "lat", "lon", "class_label", "reg_value"]
        + [c for c in df_test.columns]
    )
    df_full = df_full[ordered_cols]

    # Add timestamp
    df_full["timestamp"] = pd.to_datetime(time.ctime(st.session_state.last_update))

    # ---- Persist to Google Drive (download current file, append, cap size, re-upload) ----
    try:
        existing_history = load_history_from_drive()
        full_history = pd.concat([existing_history, df_full], ignore_index=True)
        full_history = full_history.sort_values("timestamp").tail(MAX_STORED_ROWS)
        save_history_to_drive(full_history)
    except Exception as e:
        st.error(f"Error writing to Drive: {e}")
        full_history = df_full  # fall back to this cycle's data so the app doesn't crash


    def highlight_class(val):
        if val == "Risk":
            return "background-color: #ff4d4d; color: white;"  # red
        elif val == "No risk":
            return "background-color: #4CAF50; color: white;"  # green
        return ""

    def color_reg(val):
        if pd.isna(val):
            return ""
        color = f"rgba({int(255*val)}, {int(255*(1-val))}, 100, 0.7)"
        return f"background-color: {color};"

    styled_df = df_full.style.applymap(highlight_class, subset=["class_label"]) \
                             .applymap(color_reg, subset=["reg_value"])

    st.dataframe(styled_df, use_container_width=True)

import altair as alt

# ---- Global History (already downloaded above when we saved to Drive) ----

history = full_history.copy()
if not history.empty:
    history = history.sort_values("timestamp").tail(HISTORY_CHART_LIMIT)

# Rename columns for the charts
history = history.rename(columns={
    "id": "sensor_id",
    "temperature_air_C": "temperature",
    "humidity_percent": "humidity"
})

# Format date + time for display so rows from different days are never confused
# just because the clock time happens to match
if not history.empty:
    history["time_str"] = pd.to_datetime(history["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
else:
    history["time_str"] = []

st.subheader(f"Temperature and humidity — last {len(history)} readings")

col_temp, col_hum = st.columns(2)

with col_temp:
    st.markdown("**Temperature (°C)**")
    chart_temp = (
        alt.Chart(history)
        .mark_line(point=True)
        .encode(
            x=alt.X("time_str:N", title="Date & time", sort=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("temperature:Q", title="Temperature (°C)"),
            color="sensor_id:N"
        )
        .properties(width="container", height=300)
    )
    st.altair_chart(chart_temp, use_container_width=True)

with col_hum:
    st.markdown("**Humidity (%)**")
    chart_hum = (
        alt.Chart(history)
        .mark_line(point=True)
        .encode(
            x=alt.X("time_str:N", title="Date & time", sort=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("humidity:Q", title="Humidity (%)"),
            color="sensor_id:N"
        )
        .properties(width="container", height=300)
    )
    st.altair_chart(chart_hum, use_container_width=True)


# --- Footer ---
st.markdown("---")
st.write(f"Last sensor update: **{time.ctime(st.session_state.last_update)}**")
st.caption("This dashboard simulates data sent by the sensors by randomly picking a row from testing_data.csv for each sensor every 5 minutes.")