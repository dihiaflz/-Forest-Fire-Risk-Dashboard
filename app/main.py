import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from random import randint
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_autorefresh import st_autorefresh

# ---------- CONFIG ----------
# Paths (adjust if needed)
CLASS_MODEL_PATH = "C:/Users/ASUS/Documents/AI_projects/Prevent forest fire CERIST/app/trained_models/classification_model.pkl"   # your classification model (pickle)
REG_MODEL_PATH = "C:/Users/ASUS/Documents/AI_projects/Prevent forest fire CERIST/app/trained_models/regression_model.pkl"         # your regression model (pickle)
CSV_PATH = "C:/Users/ASUS/Documents/AI_projects/Prevent forest fire CERIST/app/testing_data.csv"          # CSV with test rows (features only)

# Map defaults (you can change coordinates)
DEFAULT_LAT = 36.75
DEFAULT_LON = 3.05
MAP_ZOOM = 12

# Simulation settings
AUTO_REFRESH_MS = 300_000   # 5 minutes in milliseconds
# ----------------------------

st.set_page_config(page_title="Forest Fire Sensor Dashboard", layout="wide")
st.title("Kendira Forest — Sensor Dashboard (Simulation)")

# Utility: load pickle with friendly error
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

# Load models
classification_model = load_pickle(CLASS_MODEL_PATH)
regression_model = load_pickle(REG_MODEL_PATH)

# Load test dataset
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

# Fit scaler once on the full dataset
scaler = StandardScaler()
scaler.fit(df_test.values)

# Session state: index for current sample
if "current_idx" not in st.session_state:
    st.session_state.current_idx = randint(0, len(df_test) - 1)
    st.session_state.last_update = time.time()

# Layout: left column for map + controls, right for metrics and model outputs
col_map, col_info = st.columns([1, 1.25])

with col_map:
    st.subheader("Sensor Location")
    lat = st.number_input("Sensor Latitude", value=float(DEFAULT_LAT), format="%.6f")
    lon = st.number_input("Sensor Longitude", value=float(DEFAULT_LON), format="%.6f")
    # Map (Streamlit's simple map)
    map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(map_df, zoom=MAP_ZOOM)
    st.write("**Auto-refresh:** every 5 minutes (or click Next sample to force).")

    # Manual next sample button
    if st.button("Next sample (simulate new sensor reading)"):
        st.session_state.current_idx = randint(0, len(df_test) - 1)
        st.session_state.last_update = time.time()

    # Auto-refresh widget to trigger rerun every AUTO_REFRESH_MS
    # This triggers a rerun silently; we use it to update the sample index by comparing time
    from streamlit_autorefresh import st_autorefresh  # add streamlit-autorefresh to requirements

    # Register autorefresh; it will return an integer that increments every refresh
    refresh_count = st_autorefresh(interval=AUTO_REFRESH_MS, key="autorefresh")
    # If enough time passed (>= 5 minutes) update sample
    if time.time() - st.session_state.get("last_update", 0) > (AUTO_REFRESH_MS / 1000.0):
        st.session_state.current_idx = randint(0, len(df_test) - 1)
        st.session_state.last_update = time.time()

with col_info:
    st.subheader("Sensor Readings (current sample)")
    idx = st.session_state.current_idx
    sample = df_test.iloc[idx]
    # Show raw values
    st.write(f"**Sample index:** {idx}")
    st.table(sample.to_frame(name="value"))

    # Prepare features for models
    X_raw = sample.values.reshape(1, -1)  # shape (1, n_features)
    # Classification uses raw X_raw (you mentioned no scaling for classification)
    try:
        class_pred = classification_model.predict(X_raw)
        # If classification model returns probabilities, handle that
        if hasattr(classification_model, "predict_proba"):
            class_prob = classification_model.predict_proba(X_raw)[:, 1][0]
            class_label = "Risk" if class_prob > 0.5 else "No risk"
        else:
            # assume direct label output (0/1)
            class_label = "Risk" if int(class_pred[0]) == 1 else "No risk"
            class_prob = None
    except Exception as e:
        st.error(f"Error running classification model: {e}")
        class_label = "Error"
        class_prob = None

    # Regression: scale then predict
    try:
        X_scaled = scaler.transform(X_raw)  # (1, n_features)
        reg_pred = regression_model.predict(X_scaled)
        # if regression output is probability between 0..1 already, use it; otherwise you may want to clip or convert scale
        reg_value = float(reg_pred[0])
    except Exception as e:
        st.error(f"Error running regression model: {e}")
        reg_value = None

    # Show outputs
    st.markdown("### Model outputs")
    st.metric(label="Classification (risk)", value=class_label)
    st.metric(label="Regression output (probability of risk)", value=reg_value)
    st.progress(min(max(reg_value, 0.0), 1.0) if 0.0 <= reg_value <= 1.0 else 0.0)


# Footer: diagnostics
st.markdown("---")
st.write(f"Last simulation update: {time.ctime(st.session_state.last_update)}")
st.caption("This dashboard simulates sensor input by picking a random row from X_test.csv every 5 minutes.")

# End of app
