# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config(
    page_title="Subscription Prediction App",
    layout="wide"
)

# =============================================================================
# ❄️ SNOW EFFECT (CSS + PARTICLES)
# =============================================================================
def add_snow_effect():
    st.markdown(
        """
        <div id="particles-js"></div>
        <style>
        #particles-js {
            position: fixed;
            width: 100vw;
            height: 100vh;
            top: 0;
            left: 0;
            z-index: -1;
        }
        </style>

        <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
        <script>
        particlesJS("particles-js", {
          "particles": {
            "number": { "value": 120 },
            "color": { "value": "#ffffff" },
            "shape": { "type": "circle" },
            "opacity": { "value": 0.8 },
            "size": { "value": 3 },
            "move": {
              "enable": true,
              "speed": 1,
              "direction": "bottom"
            }
          },
          "interactivity": { "events": { "onhover": { "enable": false } } },
          "retina_detect": true
        });
        </script>
        """,
        unsafe_allow_html=True
    )

add_snow_effect()

# =============================================================================
# 🎵 BACKGROUND MUSIC
# =============================================================================
st.markdown(
    """
    <audio autoplay loop>
        <source src="https://cdn.pixabay.com/audio/2022/10/30/audio_2dbbeef8b1.mp3" type="audio/mp3">
    </audio>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# DATA LOAD
# =============================================================================
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")   # <-- kendi csv dosyan

df_raw = load_data()

# =============================================================================
# DEFAULTS (SIMULATOR SAFE)
# =============================================================================
DEFAULTS = {col: df_raw[col].mode()[0] for col in df_raw.columns}

# =============================================================================
# FEATURE ENGINEERING (ROW-BASED)
# =============================================================================
def process_data_pipeline(df):
    df = df.copy()

    climate_map = {
        "Winter": "Cold",
        "Fall": "Cold",
        "Spring": "Mild",
        "Summer": "Hot"
    }
    df["CLIMATE_GROUP_NEW"] = df["SEASON"].map(climate_map).fillna("Mild")

    df["SPEND_PER_PREV_PURCHASE"] = (
        df["PURCHASE_AMOUNT_(USD)"] / (df["PREVIOUS_PURCHASES"] + 1)
    )

    df["REVIEW_WEIGHTED_SPEND"] = (
        df["PURCHASE_AMOUNT_(USD)"] * df["REVIEW_RATING"]
    )

    return df

# =============================================================================
# CONDITIONAL PROBABILITIES
# =============================================================================
def compute_conditional_probs(df, group_col, target_col):
    return (
        df.groupby([group_col, target_col])
        .size()
        .div(df.groupby(group_col).size(), level=0)
        .to_dict()
    )

def map_conditional_probs(df, prob_dict, group_col, target_col):
    return df.apply(
        lambda x: prob_dict.get((x[group_col], x[target_col]), 0),
        axis=1
    )

# =============================================================================
# TARGET
# =============================================================================
df_raw["TARGET"] = (df_raw["SUBSCRIPTION_STATUS"] == "Yes").astype(int)

# =============================================================================
# FEATURE PIPELINE
# =============================================================================
df_feat = process_data_pipeline(df_raw)

probs_cat = compute_conditional_probs(df_feat, "CLIMATE_GROUP_NEW", "CATEGORY")
probs_size = compute_conditional_probs(df_feat, "CLIMATE_GROUP_NEW", "SIZE")
probs_season = compute_conditional_probs(df_feat, "CLIMATE_GROUP_NEW", "SEASON")

df_feat["P_CATEGORY_given_CLIMATE_NEW"] = map_conditional_probs(
    df_feat, probs_cat, "CLIMATE_GROUP_NEW", "CATEGORY"
)
df_feat["P_SIZE_given_CLIMATE_NEW"] = map_conditional_probs(
    df_feat, probs_size, "CLIMATE_GROUP_NEW", "SIZE"
)
df_feat["P_SEASON_given_CLIMATE_NEW"] = map_conditional_probs(
    df_feat, probs_season, "CLIMATE_GROUP_NEW", "SEASON"
)

df_feat["CLIMATE_ITEM_FIT_SCORE_NEW"] = (
    df_feat["P_CATEGORY_given_CLIMATE_NEW"] *
    df_feat["P_SIZE_given_CLIMATE_NEW"] *
    df_feat["P_SEASON_given_CLIMATE_NEW"]
)

# =============================================================================
# MODEL DATASET
# =============================================================================
DROP_COLS = [
    "CUSTOMER_ID",
    "SUBSCRIPTION_STATUS",
    "TARGET",
    "ITEM_PURCHASED",
    "COLOR"
]

X = pd.get_dummies(df_feat.drop(columns=DROP_COLS, errors="ignore"))
y = df_feat["TARGET"]

# =============================================================================
# TRAIN TEST
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

roc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])

# =============================================================================
# SAVE STATE
# =============================================================================
st.session_state.update({
    "final_model": model,
    "scaler_model": scaler,
    "X_columns": X.columns,
    "probs_cat": probs_cat,
    "probs_size": probs_size,
    "probs_season": probs_season,
    "best_threshold": 0.5
})

# =============================================================================
# UI
# =============================================================================
st.title("📊 Subscription Prediction App")
st.metric("ROC-AUC", f"{roc:.3f}")

# =============================================================================
# SIMULATOR
# =============================================================================
st.header("🧪 Live Prediction Simulator")

with st.form("simulator"):
    age = st.slider("Age", 18, 70, 30)
    spend = st.number_input("Spend ($)", 1, 1000, 120)
    prev = st.number_input("Previous Purchases", 0, 100, 4)
    rating = st.slider("Rating", 1.0, 5.0, 4.2)
    season = st.selectbox("Season", df_raw["SEASON"].unique())
    category = st.selectbox("Category", df_raw["CATEGORY"].unique())
    size = st.selectbox("Size", df_raw["SIZE"].unique())
    submit = st.form_submit_button("🔮 Predict")

if submit:
    row = DEFAULTS.copy()
    row.update({
        "AGE": age,
        "PURCHASE_AMOUNT_(USD)": spend,
        "PREVIOUS_PURCHASES": prev,
        "REVIEW_RATING": rating,
        "SEASON": season,
        "CATEGORY": category,
        "SIZE": size,
        "SUBSCRIPTION_STATUS": "No"
    })

    sim_df = process_data_pipeline(pd.DataFrame([row]))

    sim_df["P_CATEGORY_given_CLIMATE_NEW"] = map_conditional_probs(
        sim_df, st.session_state["probs_cat"], "CLIMATE_GROUP_NEW", "CATEGORY"
    )
    sim_df["P_SIZE_given_CLIMATE_NEW"] = map_conditional_probs(
        sim_df, st.session_state["probs_size"], "CLIMATE_GROUP_NEW", "SIZE"
    )
    sim_df["P_SEASON_given_CLIMATE_NEW"] = map_conditional_probs(
        sim_df, st.session_state["probs_season"], "CLIMATE_GROUP_NEW", "SEASON"
    )

    sim_df["CLIMATE_ITEM_FIT_SCORE_NEW"] = (
        sim_df["P_CATEGORY_given_CLIMATE_NEW"] *
        sim_df["P_SIZE_given_CLIMATE_NEW"] *
        sim_df["P_SEASON_given_CLIMATE_NEW"]
    )

    sim_enc = pd.get_dummies(sim_df.drop(columns=DROP_COLS, errors="ignore"))
    sim_enc = sim_enc.reindex(columns=st.session_state["X_columns"], fill_value=0)
    sim_scaled = st.session_state["scaler_model"].transform(sim_enc)

    prob = st.session_state["final_model"].predict_proba(sim_scaled)[0, 1]

    st.success(f"🎯 Subscription Probability: **{prob:.2%}**")
