############################################################
# STREAMLIT CUSTOMER ANALYTICS + SEGMENTATION + CRM
############################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid")

st.set_page_config(
    page_title="Customer Analytics Platform",
    layout="wide"
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from scipy.stats import chi2_contingency
from datetime import datetime

# Modelleme
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Veri İşleme
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, precision_score, recall_score, 
                             f1_score, roc_curve, auc, silhouette_score)

# Ayarlar
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Miuul Alışveriş Analizi V2", page_icon="🛍️", layout="wide")

# CSS ve Kar Taneleri
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

local_css("style.css")

# Kar taneleri
animation_symbol = "❄️"
st.markdown(f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    """, unsafe_allow_html=True)

# Müzik
st.sidebar.markdown("---")
def fallback_audio():
    url = "https://www.mfiles.co.uk/mp3-downloads/jingle-bells-keyboard.mp3"
    st.sidebar.audio(url)
    st.sidebar.info("🎵 Müzik için Play'e basın")

fallback_audio()

# Tema
def apply_modern_christmas_theme():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(180deg, #050a14 0%, #001219 100%);
            color: #ffffff;
        }
        [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.05);
            border: 2px solid #f4a261;
            border-radius: 15px;
            padding: 15px 10px;
            box-shadow: 0px 4px 15px rgba(244, 162, 97, 0.2);
            text-align: center;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: bold;
        }
        [data-testid="stMetricLabel"] {
            color: #d62828 !important;
            font-size: 1.1rem !important;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] {
            background-color: #000814 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        div.stButton > button {
            background-color: #d62828 !important;
            color: white !important;
            border-radius: 25px !important;
            border: none !important;
            transition: 0.3s;
            width: 100%;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #f4a261 !important;
            transform: scale(1.02);
        }
        button[data-baseweb="tab"] {
            font-size: 18px;
            color: #f8f9fa !important;
        }
        button[aria-selected="true"] {
            border-bottom: 3px solid #d62828 !important;
            font-weight: bold;
        }
        .snowflake {
            color: #fff; font-size: 1.2em; position: fixed; top: -10%; z-index: 9999;
            animation-name: snowflakes-fall, snowflakes-shake;
            animation-duration: 10s, 3s; animation-iteration-count: infinite;
            pointer-events: none;
        }
        @keyframes snowflakes-fall { 0% {top:-10%} 100% {top:100%} }
        @keyframes snowflakes-shake { 0% {transform:translateX(0px)} 50% {transform:translateX(80px)} 100% {transform:translateX(0px)} }
        </style>
    """, unsafe_allow_html=True)

apply_modern_christmas_theme()
############################################################
# SIDEBAR
############################################################

st.sidebar.title("📊 Analiz Menüsü")

menu = st.sidebar.radio(
    "Bölüm Seç",
    [
        "📂 Veri Yükleme",
        "🔍 EDA",
        "👥 Abone Analizi",
        "🧩 Segmentasyon",
        "🏷 Segment Profilleri",
        "🤖 Model Karşılaştırma",
        "📣 CRM Aksiyonları"
    ]
)

############################################################
# DATA LOAD
############################################################

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.upper().str.replace(" ", "_")
    return df

############################################################
# 1. DATA UPLOAD
############################################################

if menu == "📂 Veri Yükleme":

    st.title("📂 Veri Yükleme")

    file = st.file_uploader("CSV dosyasını yükle", type=["csv"])

    if file:
        df = load_data(file)
        st.session_state["df"] = df
        st.success("Veri başarıyla yüklendi")
        st.dataframe(df.head())
        st.write("Veri Boyutu:", df.shape)

############################################################
# 2. EDA
############################################################

elif menu == "🔍 EDA":

    st.title("🔍 Exploratory Data Analysis")

    df = st.session_state.get("df")

    if df is None:
        st.warning("Önce veri yüklemelisin")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Numerik Özet")
            st.dataframe(df.describe().T)

        with col2:
            st.subheader("Eksik Değerler")
            na = df.isnull().sum()
            st.dataframe(na[na > 0])

        num_col = st.selectbox(
            "Dağılımını görmek istediğin değişken",
            df.select_dtypes(include=["int64", "float64"]).columns
        )

        fig, ax = plt.subplots()
        sns.histplot(df[num_col], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

############################################################
# 3. SUBSCRIPTION ANALYSIS
############################################################

elif menu == "👥 Abone Analizi":

    st.title("👥 Abonelik Analizi")

    df = st.session_state.get("df")

    if df is None:
        st.warning("Önce veri yüklemelisin")
    else:
        rate = df["SUBSCRIPTION_STATUS"].value_counts(normalize=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Abone Oranı", f"%{rate.get('Yes',0)*100:.1f}")
        with col2:
            st.metric("Abone Olmayan", f"%{rate.get('No',0)*100:.1f}")

        fig, ax = plt.subplots()
        sns.countplot(x="SUBSCRIPTION_STATUS", data=df, ax=ax)
        st.pyplot(fig)

############################################################
# 4. SEGMENTATION
############################################################

elif menu == "🧩 Segmentasyon":

    st.title("🧩 Müşteri Segmentasyonu")

    df = st.session_state.get("df")

    if df is None:
        st.warning("Önce veri yüklemelisin")
    else:
        features = [
            "PURCHASE_AMOUNT_(USD)",
            "PREVIOUS_PURCHASES",
            "REVIEW_RATING"
        ]

        X = df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.slider("Cluster Sayısı", 2, 10, 4)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df["CLUSTER"] = kmeans.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, df["CLUSTER"])
        st.metric("Silhouette Score", f"{sil:.3f}")

        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
        pca_df["Cluster"] = df["CLUSTER"]

        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(
            x="PC1", y="PC2",
            hue="Cluster",
            data=pca_df,
            palette="tab10",
            ax=ax
        )
        st.pyplot(fig)

        st.session_state["segmented_df"] = df

############################################################
# 5. SEGMENT PROFILING
############################################################

elif menu == "🏷 Segment Profilleri":

    st.title("🏷 Segment Profilleri")

    df = st.session_state.get("segmented_df")

    if df is None:
        st.warning("Önce segmentasyon yapmalısın")
    else:
        profile = (
            df
            .groupby("CLUSTER")
            .agg(
                AvgSpend=("PURCHASE_AMOUNT_(USD)", "mean"),
                AvgFreq=("PREVIOUS_PURCHASES", "mean"),
                SubRate=("SUBSCRIPTION_STATUS", lambda x: (x=="Yes").mean())
            )
        )

        def segment_name(row):
            if row["AvgSpend"] > profile["AvgSpend"].median() and row["SubRate"] > 0.5:
                return "High Value Loyal"
            if row["AvgSpend"] > profile["AvgSpend"].median():
                return "High Potential"
            if row["SubRate"] > 0.5:
                return "Loyal Low Spend"
            return "Low Value"

        profile["Segment_Name"] = profile.apply(segment_name, axis=1)
        st.dataframe(profile.round(2))

############################################################
# 6. MODEL COMPARISON
############################################################

elif menu == "🤖 Model Karşılaştırma":

    st.title("🤖 Model Karşılaştırma")

    df = st.session_state.get("df")

    if df is None:
        st.warning("Önce veri yüklemelisin")
    else:
        y = (df["SUBSCRIPTION_STATUS"] == "Yes").astype(int)
        X = df.select_dtypes(include=["int64","float64"]).fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric="logloss"),
            "LightGBM": LGBMClassifier()
        }

        results = []

        for name, model in models.items():
            auc = cross_val_score(
                model, Xs, y,
                cv=5, scoring="roc_auc"
            ).mean()
            results.append([name, auc])

        res_df = pd.DataFrame(results, columns=["Model", "CV AUC"])
        st.dataframe(res_df.sort_values("CV AUC", ascending=False))

############################################################
# 7. CRM ACTIONS
############################################################

elif menu == "📣 CRM Aksiyonları":

    st.title("📣 CRM Aksiyonları")

    df = st.session_state.get("segmented_df")

    if df is None:
        st.warning("Önce segmentasyon yapmalısın")
    else:
        # Model proba yoksa demo amaçlı üret
        if "SUB_PROBA" not in df.columns:
            np.random.seed(42)
            df["SUB_PROBA"] = np.random.uniform(0.1, 0.9, len(df))

        spend_median = df["PURCHASE_AMOUNT_(USD)"].median()

        def crm_action(row):
            if row["SUB_PROBA"] >= 0.7 and row["PURCHASE_AMOUNT_(USD)"] >= spend_median:
                return "Upsell / Premium"
            if row["SUB_PROBA"] >= 0.7:
                return "Cross-sell"
            if row["SUB_PROBA"] >= 0.4:
                return "Nurture"
            if row["PURCHASE_AMOUNT_(USD)"] >= spend_median:
                return "Winback"
            return "Aggressive Promo"

        df["CRM_ACTION"] = df.apply(crm_action, axis=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ortalama Abonelik Olasılığı", f"{df['SUB_PROBA'].mean():.2f}")
        with col2:
            st.metric("Upsell Adayı", (df["CRM_ACTION"]=="Upsell / Premium").sum())
        with col3:
            st.metric("Winback Adayı", (df["CRM_ACTION"]=="Winback").sum())

        fig, ax = plt.subplots()
        sns.countplot(
            y="CRM_ACTION",
            data=df,
            order=df["CRM_ACTION"].value_counts().index,
            ax=ax
        )
        st.pyplot(fig)

        st.subheader("Öncelikli Müşteriler")
        st.dataframe(
            df[
                [
                    "CLUSTER",
                    "PURCHASE_AMOUNT_(USD)",
                    "PREVIOUS_PURCHASES",
                    "SUB_PROBA",
                    "CRM_ACTION"
                ]
            ]
            .sort_values("SUB_PROBA", ascending=False)
            .head(50)
        )
