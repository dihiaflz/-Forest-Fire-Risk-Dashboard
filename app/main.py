import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from random import randint
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_autorefresh import st_autorefresh
import altair as alt
import pydeck as pdk
import os



# ---------- CONFIG ----------
working_dir = os.path.dirname(os.path.abspath(__file__))
CLASS_MODEL_PATH = working_dir + "/trained_models/classification_model.pkl"   # classification model (pickle)
REG_MODEL_PATH = working_dir + "/trained_models/regression_model.pkl"         # regression model (pickle)
CSV_PATH = working_dir + "/testing_data.csv"          # CSV with test rows (features only)
AUTO_REFRESH_MS = 300_000   # 5 minutes in milliseconds


st.set_page_config(page_title="Tableau de bord prévention de feu de forêt de Kendira", layout="wide")
st.title("Tableau de bord prévention de feu de forêt de Kendira")

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
        {"id": "S2", "lat": 36.567583, "lon": 5.027500},  # Nord
        {"id": "S3", "lat": 36.518033, "lon": 5.027500},  # Sud
        {"id": "S4", "lat": 36.540556, "lon": 5.061160},  # Est
        {"id": "S5", "lat": 36.540556, "lon": 4.993840},  # Ouest
        {"id": "S6", "lat": 36.558574, "lon": 5.049940},  # Nord-Est
        {"id": "S7", "lat": 36.527042, "lon": 5.044330},  # Sud-Est
        {"id": "S8", "lat": 36.524340, "lon": 5.007300},  # Sud-Ouest
    ]
    return sensors

sensors = get_kendira_sensors()

# ---------- Last update display ----------
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

# ---------- Manuel refresh button ----------
if st.button("Prochaine simulation (forcer la mise à jour)"):
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
    st.subheader("Sensor Locations")

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

    st.write("**Auto-refresh:** chaque 5 minutes (ou forcer avec le bouton en haut).")

with col_info:
    st.subheader("Métriques envoyées par chaque capteur et prédiction correspondante")

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

    # Save in a csv file
    DATABASE_PATH = working_dir + "/database.csv"
    try:
        df_full.to_csv(DATABASE_PATH, mode="a", header=not pd.io.common.file_exists(DATABASE_PATH), index=False, encoding="utf-8")
    except Exception as e:
        st.error(f"Erreur lors de l'écriture dans la base de données: {e}")


    def highlight_class(val):
        if val == "Risk":
            return "background-color: #ff4d4d; color: white;"  # rouge
        elif val == "No risk":
            return "background-color: #4CAF50; color: white;"  # vert
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

# ---- Global History (from database.csv) ----

try:
    history = pd.read_csv(DATABASE_PATH, parse_dates=["timestamp"])
except FileNotFoundError:
    st.warning("Aucun historique trouvé (database.csv manquant ou vide).")
    history = pd.DataFrame(columns=["timestamp", "id", "temperature_air_C", "humidity_percent"])

# Renommer les colonnes pour les graphes
history = history.rename(columns={
    "id": "sensor_id",
    "temperature_air_C": "temperature",
    "humidity_percent": "humidity"
})

# Limiter à 200 dernières entrées (optionnel, pour éviter surcharge graphique)
history = history.tail(200)

# Formater le temps pour affichage
history["time_str"] = pd.to_datetime(history["timestamp"]).dt.strftime("%H:%M:%S")

st.subheader("Tendances des métriques avec le temps")

col_temp, col_hum = st.columns(2)

with col_temp:
    st.markdown("**Température (°C)**")
    chart_temp = (
        alt.Chart(history)
        .mark_line(point=True)
        .encode(
            x=alt.X("time_str:N", title="Temps", axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("temperature:Q", title="Température (°C)"),
            color="sensor_id:N"
        )
        .properties(width="container", height=300)
    )
    st.altair_chart(chart_temp, use_container_width=True)

with col_hum:
    st.markdown("**Humidité (%)**")
    chart_hum = (
        alt.Chart(history)
        .mark_line(point=True)
        .encode(
            x=alt.X("time_str:N", title="Temps", axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("humidity:Q", title="Humidité (%)"),
            color="sensor_id:N"
        )
        .properties(width="container", height=300)
    )
    st.altair_chart(chart_hum, use_container_width=True)


# --- Footer ---
st.markdown("---")
st.write(f"Dernière mise à jour des capteurs: **{time.ctime(st.session_state.last_update)}**")
st.caption("Ce tableau de bord simule les données enovoyées par les capteurs par choisir aléatoirement une ligne du testing_data.csv pour chaque capteur chaque 5 minutes.")

