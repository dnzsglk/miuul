# =============================================================================
# 🔒 MODEL EĞİTİMİ + KARŞILAŞTIRMA (CACHE'Lİ)
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

@st.cache_resource(show_spinner="🔄 Model cache'ten yükleniyor veya ilk kez eğitiliyor...")
def train_and_compare_models_cached(df_eng_train, df_eng_test, random_state=42):

    y_train = (df_eng_train["SUBSCRIPTION_STATUS"] == "Yes").astype(int)
    y_test = (df_eng_test["SUBSCRIPTION_STATUS"] == "Yes").astype(int)

    drop_cols = [
        'CUSTOMER_ID','SUBSCRIPTION_STATUS', 'ITEM_PURCHASED', 'LOCATION', 'COLOR', 'SIZE',
        'FREQUENCY_OF_PURCHASES', 'PAYMENT_METHOD', 'SHIPPING_TYPE',
        'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 'REVIEW_RATING',
        'AGE', 'DISCOUNT_APPLIED', 'SEASON', 'PROMO_CODE_USED'
    ]

    X_train_df, X_test_df = encode_train_test(df_eng_train, df_eng_test, drop_cols)

    leak_prefixes = ("SUB_FREQ_NEW", "PROMO_NO_SUB_NEW", "SHIP_SUB_NEW")
    leakage_cols = [c for c in X_train_df.columns if c.startswith(leak_prefixes)]

    X_train_base = X_train_df.drop(columns=leakage_cols, errors="ignore")
    X_test_base = X_test_df.drop(columns=leakage_cols, errors="ignore")

    rf_selector = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf_selector.fit(X_train_base, y_train)

    importances = pd.Series(
        rf_selector.feature_importances_,
        index=X_train_base.columns
    )

    keep_cols = importances[importances >= 0.01].index.tolist()

    X_train = X_train_base[keep_cols]
    X_test = X_test_base[keep_cols]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("RandomForest", RandomForestClassifier(random_state=random_state, class_weight='balanced')),
        ("XGBoost", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=random_state)),
        ("LightGBM", LGBMClassifier(random_state=random_state, verbose=-1))
    ]

    cv_results = []
    best_model = None
    best_score = -1
    best_name = None

    for name, model in models:
        scores = cross_val_score(
            model,
            X_train_s,
            y_train,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        mean_score = scores.mean()

        cv_results.append({
            "Model": name,
            "CV AUC Mean": mean_score,
            "Std": scores.std()
        })

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name

    best_model.fit(X_train_s, y_train)

    y_proba = (
        best_model.predict_proba(X_test_s)[:, 1]
        if hasattr(best_model, "predict_proba")
        else best_model.decision_function(X_test_s)
    )

    return {
        "final_model": best_model,
        "best_model_name": best_name,
        "cv_results": pd.DataFrame(cv_results),
        "scaler": scaler,
        "X_columns": X_train.columns.tolist(),
        "y_test": y_test,
        "y_proba": y_proba
    }

# =============================================================================
# 🔒 CACHE BLOĞU SONU
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from scipy.stats import chi2_contingency
from datetime import datetime

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
st.set_page_config(page_title="Deneme", page_icon="🛍️", layout="wide")

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

# Müzik Bölümü
import streamlit.components.v1 as components

# 2. SIDEBAR BÖLÜMÜNE EKLE
st.sidebar.markdown("---")

# Noel Ağacı
st.sidebar.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 70px; margin-bottom: 0px; filter: drop-shadow(0 0 10px #f4a261);">🎄</h1>
        <h3 style="color: #f4a261; margin-top: 0px;">Mutlu Yıllar!</h3>
    </div>
    """, unsafe_allow_html=True)

# Otomatik Müzik Çalar (JavaScript Tetikleyici)
# ===================== 🎵 SIDEBAR SES KONTROLÜ =====================
st.sidebar.markdown("### 🎵 Arka Plan Müziği")

music_on = st.sidebar.toggle("Müzik Aç / Kapat", value=False)
volume = st.sidebar.slider("Ses Seviyesi", 0.0, 1.0, 0.4, 0.05)

audio_url = "https://www.mfiles.co.uk/mp3-downloads/jingle-bells-keyboard.mp3"

# ===================== 🎶 KONTROLLÜ MÜZİK OYNATICI =====================
components.html(
    f"""
    <audio id="christmasAudio" loop>
        <source src="{audio_url}" type="audio/mp3">
    </audio>

    <script>
        const audio = document.getElementById("christmasAudio");
        audio.volume = {volume};

        if ({str(music_on).lower()}) {{
            audio.play().catch(() => {{}});
        }} else {{
            audio.pause();
        }}
    </script>
    """,
    height=0,
)
# ===============================================================


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

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# =============================================================================
# DATA PROCESSING PIPELINE
# =============================================================================
def process_data_pipeline(df):
    df_eng = df.copy()
    
    # Numeric force
    num_force_cols = ["PURCHASE_AMOUNT_(USD)", "PREVIOUS_PURCHASES", "AGE", "REVIEW_RATING"]
    for c in num_force_cols:
        if c in df_eng.columns:
            df_eng[c] = pd.to_numeric(df_eng[c], errors="coerce")

    if 'SUBSCRIPTION_STATUS' in df_eng.columns:
        df_eng['TEMP_TARGET'] = df_eng['SUBSCRIPTION_STATUS'].map({"Yes": 1, "No": 0})
    else:
        df_eng['TEMP_TARGET'] = 0 

    # Temel değişkenler
    df_eng['TOTAL_SPEND_WEIGHTED_NEW'] = df_eng['PREVIOUS_PURCHASES'] * df_eng['PURCHASE_AMOUNT_(USD)']
    df_eng['SPEND_PER_PURCHASE_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / (df_eng['PREVIOUS_PURCHASES'] + 1)
    
    freq_map = {'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26, 'Quarterly': 4, 'Annually': 1, 'Monthly': 12, 'Every 3 Months': 4}
    df_eng['FREQUENCY_VALUE_NEW'] = df_eng['FREQUENCY_OF_PURCHASES'].map(freq_map)

    pay_map = {'Cash': 'Cash', 'Credit Card': 'Card', 'Debit Card': 'Card', 'PayPal': 'Online', 'Venmo': 'Online', 'Bank Transfer': 'Online'}
    df_eng['PAYMENT_TYPE_NEW'] = df_eng['PAYMENT_METHOD'].map(pay_map)

    # Kategorik binning
    df_eng["AGE_NEW"] = pd.cut(df_eng["AGE"], bins=[0, 30, 45, 56, 200], labels=["18-30", "31-45", "46-56", "57-70"])
    df_eng["PURCHASE_AMOUNT_(USD)_NEW"] = pd.qcut(df_eng["PURCHASE_AMOUNT_(USD)"], q=4, labels=["Low", "Mid", "High", "Very High"])
    df_eng["LOYALTY_LEVEL_NEW"] = pd.cut(df_eng["PREVIOUS_PURCHASES"], bins=[0, 13, 25, 38, 200], labels=["Low", "Mid", "High", "Very High"], include_lowest=True)

    # Leakage features
    df_eng["SUB_FREQ_NEW"] = (df_eng["TEMP_TARGET"].astype(str) + "_" + df_eng["FREQUENCY_OF_PURCHASES"].astype(str))
    df_eng["PROMO_NO_SUB_NEW"] = ((df_eng["PROMO_CODE_USED"] == "Yes") & (df_eng["TEMP_TARGET"] == 0)).astype(int)
    df_eng["SHIP_SUB_NEW"] = (df_eng["SHIPPING_TYPE"].astype(str) + "_" + df_eng["TEMP_TARGET"].astype(str))

    # Sezon features
    df_eng["SEASON_CATEGORY_NEW"] = df_eng["SEASON"].astype(str) + "_" + df_eng["CATEGORY"].astype(str)
    df_eng["SEASON_COLOR_NEW"] = df_eng["SEASON"].astype(str) + "_" + df_eng["COLOR"].astype(str)
    df_eng["ITEM_CATEGORY_NEW"] = df_eng["CATEGORY"].astype(str) + "_" + df_eng["ITEM_PURCHASED"].astype(str)
    df_eng["HIGH_REVIEW_RATING_NEW"] = (df_eng["REVIEW_RATING"] >= 4).astype(int)
    df_eng["SPEND_RATING_NEW"] = df_eng["PURCHASE_AMOUNT_(USD)"] * df_eng["REVIEW_RATING"]

    top_locations = df_eng["LOCATION"].value_counts().nlargest(10).index
    df_eng["LOCATION_GROUPED_NEW"] = df_eng["LOCATION"].where(df_eng["LOCATION"].isin(top_locations), "Other")

    # İklim gruplaması
    cold_states = ["Alaska", "North Dakota", "South Dakota", "Minnesota", "Wisconsin", "Michigan", "Montana", "Wyoming", "Maine", "Vermont", "New Hampshire"]
    cool_states = ["Massachusetts", "Connecticut", "Rhode Island", "New York", "Pennsylvania", "New Jersey", "Ohio", "Indiana", "Illinois", "Iowa", "Nebraska", "Kansas", "Colorado", "Utah", "Idaho", "Washington", "Oregon"]
    warm_states = ["Virginia", "Maryland", "Delaware", "Kentucky", "Missouri", "West Virginia", "North Carolina", "Tennessee", "Arkansas", "Oklahoma"]
    hot_states = ["Florida", "Texas", "Louisiana", "Mississippi", "Alabama", "Georgia", "South Carolina", "Arizona", "Nevada", "New Mexico", "California"]
    tropical_states = ["Hawaii"]

    def climate_group(state):
        if state in cold_states: return "Cold"
        elif state in cool_states: return "Cool"
        elif state in warm_states: return "Warm"
        elif state in hot_states: return "Hot"
        elif state in tropical_states: return "Tropical"
        else: return "Unknown"

    df_eng["CLIMATE_GROUP_NEW"] = df_eng["LOCATION"].apply(climate_group)
    
    df_eng["LOYALTY_SCORE_NEW"] = pd.qcut(df_eng["PREVIOUS_PURCHASES"], q=4, labels=[1, 2, 3, 4]).astype(int)
    df_eng["PROMO_X_LOYALTY"] = ((df_eng["PROMO_CODE_USED"] == "Yes").astype(int) * df_eng["LOYALTY_SCORE_NEW"])
    df_eng["PROMO_X_FREQ"] = ((df_eng["PROMO_CODE_USED"] == "Yes").astype(int) * df_eng["FREQUENCY_VALUE_NEW"])

    if 'TEMP_TARGET' in df_eng.columns: 
        df_eng.drop(columns=['TEMP_TARGET'], inplace=True)

    return df_eng

# =============================================================================
# CONDITIONAL PROBABILITY VE GROUP MEAN RATIO FONKSİYONLARI
# =============================================================================
def fit_conditional_probs(train_df, group_col, cat_col, smoothing=1.0):
    ct = pd.crosstab(train_df[group_col], train_df[cat_col])
    probs = (ct + smoothing).div((ct + smoothing).sum(axis=1), axis=0)
    return probs

def map_conditional_probs(df, probs, group_col, cat_col):
    s = probs.stack()
    keys = list(zip(df[group_col], df[cat_col]))
    return pd.Series(keys, index=df.index).map(s)

def add_group_mean_ratio(train_df, test_df, group_col, value_col, new_col, fallback="global_mean"):
    train_df[value_col] = pd.to_numeric(train_df[value_col], errors="coerce")
    test_df[value_col] = pd.to_numeric(test_df[value_col], errors="coerce")
    
    means = train_df.groupby(group_col)[value_col].mean()
    
    denom_train = train_df[group_col].map(means).astype(float)
    denom_test = test_df[group_col].map(means).astype(float)
    
    train_df[new_col] = train_df[value_col] / denom_train
    test_df[new_col] = test_df[value_col] / denom_test
    
    if fallback == "global_mean":
        gm = train_df[value_col].mean()
        test_df[new_col] = test_df[new_col].fillna(test_df[value_col] / gm)
    else:
        test_df[new_col] = test_df[new_col].fillna(train_df[new_col].mean())
    
    return train_df, test_df

def encode_train_test(train_df, test_df, drop_cols):
    train_m = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).copy()
    test_m = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).copy()
    
    cat_cols = [c for c in train_m.columns if train_m[c].dtype == "O" or str(train_m[c].dtype) == "category"]
    train_enc = pd.get_dummies(train_m, columns=cat_cols, drop_first=False)
    test_enc = pd.get_dummies(test_m, columns=cat_cols, drop_first=False)
    
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, test_enc

# =============================================================================
# UYGULAMA ARAYÜZÜ
# =============================================================================

st.title("🛍️ Alışveriş Davranışları: Gelişmiş Analitik Panel V2")
st.markdown("""
Bu panel; **Leakage-Free Pipeline**, **Train/Test Split**, **Conditional Probabilities**, 
**Silhouette Score** ve **Threshold Optimizasyonu** ile donatılmıştır.
""")

# --- SIDEBAR: VERİ YÜKLEME ---
st.sidebar.header("📂 Veri Yönetimi")
uploaded_file = st.sidebar.file_uploader("CSV Dosyanızı Buraya Sürükleyin", type=["csv"])

if uploaded_file is None:
    st.info("Analize başlamak için lütfen 'shopping_behavior_updated.csv' dosyasını yükleyin.")
    st.stop()

# --- VERİ YÜKLEME ---
@st.cache_data
def get_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.upper().str.replace(" ", "_").str.strip()
    return df

df_raw = get_data(uploaded_file)

# SESSION STATE
if 'best_threshold' not in st.session_state:
    st.session_state['best_threshold'] = 0.50
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = {}
if 'final_model' not in st.session_state:
    st.session_state['final_model'] = None
if 'scaler_model' not in st.session_state:
    st.session_state['scaler_model'] = None
if 'X_columns' not in st.session_state:
    st.session_state['X_columns'] = None

# Sekmeler
tab_eda, tab_seg, tab_model, tab_comp, tab_crm, tab_sim = st.tabs([
    "📊 EDA", 
    "🧩 Segmentasyon", 
    "🎯 Model Eğitimi",
    "📄 Model Karşılaştırma",
    "💼 CRM Analizi",
    "🧪 Simülatör"
])

# =============================================================================
# VERİ İŞLEME
# =============================================================================
with st.spinner('Veri işleniyor...'):
    # Rare encoding
    df_rare = rare_encoder(df_raw, 0.01)
    
    # Correlation check
    if 'DISCOUNT_APPLIED' in df_rare.columns and 'PROMO_CODE_USED' in df_rare.columns:
        cv_score = cramers_v(df_rare['DISCOUNT_APPLIED'], df_rare['PROMO_CODE_USED'])
        if cv_score > 0.8:
            df_rare.drop(columns=['DISCOUNT_APPLIED'], inplace=True)
    
    # Feature engineering
    df_eng = process_data_pipeline(df_rare)
    
    # Train/Test split
    df_eng_train, df_eng_test = train_test_split(
        df_eng,
        test_size=0.20,
        random_state=42,
        stratify=df_eng["SUBSCRIPTION_STATUS"]
    )

# =============================================================================
# TAB 1: EDA
# =============================================================================
with tab_eda:
    st.header("📊 Keşifsel Veri Analizi")
    
    # Genel Metrikler
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Müşteri Sayısı", df_raw.shape[0])
    col2.metric("Ortalama Yaş", f"{df_raw['AGE'].mean():.1f}")
    col3.metric("Abonelik Oranı", f"%{(df_raw['SUBSCRIPTION_STATUS']=='Yes').mean()*100:.1f}")
    col4.metric("Ortalama Harcama", f"${df_eng['TOTAL_SPEND_WEIGHTED_NEW'].mean():.1f}")

    st.divider()

    # Görselleştirmeler
    st.subheader("📊 Abonelik Odaklı Görselleştirmeler")

    # === 1. SATIR ===
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("*Abonelik Durumuna Göre Harcama Dağılımı*")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for status in df_raw['SUBSCRIPTION_STATUS'].unique():
            data = df_raw[df_raw['SUBSCRIPTION_STATUS'] == status]['PURCHASE_AMOUNT_(USD)']
            sns.kdeplot(data, ax=ax1, label=status, fill=True, alpha=0.5)
        ax1.set_xlabel('Harcama Tutarı ($)')
        ax1.set_ylabel('Yoğunluk')
        ax1.set_title('Abonelik Durumuna Göre Harcama Dağılımı')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.markdown("*Kategori Bazlı Abonelik Oranları*")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        category_sub = (
            df_raw.groupby('CATEGORY')['SUBSCRIPTION_STATUS']
            .apply(lambda x: (x == 'Yes').mean() * 100)
            .sort_values()
        )
        sns.barplot(
            x=category_sub.values, 
            y=category_sub.index, 
            ax=ax2, 
            hue=category_sub.index, 
            palette='viridis', 
            legend=False
        )
        ax2.set_xlabel('Abonelik Oranı (%)')
        ax2.set_ylabel('Kategori')
        ax2.set_title('Kategori Bazında Abonelik Oranları')
        ax2.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig2)
        plt.close(fig2)
    
    # === 2. SATIR ===
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("*Promosyon Kullanımı vs Abonelik*")
        promo_sub = pd.crosstab(df_raw['PROMO_CODE_USED'], df_raw['SUBSCRIPTION_STATUS'], normalize='index') * 100
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        promo_sub.plot(kind='bar', ax=ax3, rot=0)
        ax3.set_xlabel('Promosyon Kullanımı')
        ax3.set_ylabel('Yüzde (%)')
        ax3.set_title('Promosyon Kullanımı ve Abonelik İlişkisi')
        ax3.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        st.markdown("*Cinsiyet Bazlı Abonelik Dağılımı*")
        gender_sub = pd.crosstab(df_raw['GENDER'], df_raw['SUBSCRIPTION_STATUS'], normalize='index') * 100
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        gender_sub.plot(kind='bar', ax=ax4, rot=0)
        ax4.set_xlabel('Cinsiyet')
        ax4.set_ylabel('Yüzde (%)')
        ax4.set_title('Cinsiyet Bazında Abonelik Dağılımı')
        ax4.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig4)
        plt.close(fig4)
        
# =============================================================================
# TAB 2: SEGMENTASYON
# =============================================================================
with tab_seg:
    st.header("🧩 K-Means Müşteri Segmentasyonu (Leakage-Free)")

    segmentation_features = [
        "PURCHASE_AMOUNT_(USD)",
        "PREVIOUS_PURCHASES",
        "FREQUENCY_VALUE_NEW",
        "SPEND_PER_PURCHASE_NEW",
        "TOTAL_SPEND_WEIGHTED_NEW"
    ]

    X_seg = df_eng[[c for c in segmentation_features if c in df_eng.columns]].copy()
    X_seg.fillna(0, inplace=True)

    scaler_seg = StandardScaler()
    X_scaled = scaler_seg.fit_transform(X_seg)

    optimal_k = 5
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)

    # Session state'e kaydet
    st.session_state["kmeans"] = kmeans
    st.session_state["scaler_seg"] = scaler_seg
    st.session_state["clusters"] = clusters
    st.session_state["optimal_k"] = optimal_k

    # Üst metrikler
    m1, m2 = st.columns(2)
    m1.metric("K (Cluster Sayısı)", optimal_k)
    m2.metric("Silhouette Score", f"{sil_score:.3f}")

    st.divider()

    # Grafikleri yan yana koyma
    st.subheader("🎨 Segment Görselleştirmeleri (2D vs 3D)")
    col_graph1, col_graph2 = st.columns(2)

    with col_graph1:
        # PCA 2D
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = clusters
        
        fig_pca, ax_pca = plt.subplots(figsize=(8, 7))
        scatter = ax_pca.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], 
                                cmap='viridis', s=50, alpha=0.6, edgecolors='w')
        plt.colorbar(scatter, ax=ax_pca, label='Cluster')
        
        ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varyans)')
        ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varyans)')
        ax_pca.set_title("2D Segment Dağılımı")
        st.pyplot(fig_pca)
        plt.close(fig_pca)

    with col_graph2:
        # PCA 3D
        from mpl_toolkits.mplot3d import Axes3D
        pca3d = PCA(n_components=3)
        comps3d = pca3d.fit_transform(X_scaled)
        df_pca3d = pd.DataFrame(comps3d, columns=["PC1", "PC2", "PC3"])
        df_pca3d["Cluster"] = clusters
        
        fig_3d = plt.figure(figsize=(8, 7))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        scatter_3d = ax_3d.scatter(
            df_pca3d["PC1"], df_pca3d["PC2"], df_pca3d["PC3"],
            c=df_pca3d["Cluster"], cmap="viridis", s=50, alpha=0.7, edgecolors='w'
        )
        
        ax_3d.set_xlabel(f'PC1 ({pca3d.explained_variance_ratio_[0]*100:.1f}%)')
        ax_3d.set_ylabel(f'PC2 ({pca3d.explained_variance_ratio_[1]*100:.1f}%)')
        ax_3d.set_zlabel(f'PC3 ({pca3d.explained_variance_ratio_[2]*100:.1f}%)')
        
        total_var_3d = sum(pca3d.explained_variance_ratio_) * 100
        ax_3d.set_title(f"3D Segment Dağılımı (Top: %{total_var_3d:.1f})")
        
        ax_3d.tick_params(axis='both', which='major', labelsize=8)
        st.pyplot(fig_3d)
        plt.close(fig_3d)

    st.divider()
    
    # Segment profilleri
    df_report = df_eng.copy()
    df_report["Cluster"] = clusters
    df_report["SUBSCRIPTION"] = (df_report["SUBSCRIPTION_STATUS"] == "Yes").astype(int)
    df_report["PROMO_USED_VAL"] = (df_report["PROMO_CODE_USED"] == "Yes").astype(int)

    st.session_state["df_report"] = df_report

    # mode için güvenli fonksiyon
    def safe_mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else "Unknown"

    st.subheader("📊 Segment Profilleri (Detaylı)")

    segment_profiles = df_report.groupby("Cluster").agg(
        N=("CUSTOMER_ID", "count"),
        Yas=("AGE", "mean"),
        Harcama_USD=("PURCHASE_AMOUNT_(USD)", "mean"),
        Sub_Pct=("SUBSCRIPTION", "mean"),
        PrevPur=("PREVIOUS_PURCHASES", "mean"),
        Kategori=("CATEGORY", safe_mode),
        Odeme=("PAYMENT_METHOD", safe_mode),
        Kargo=("SHIPPING_TYPE", safe_mode),
        Iklim=("CLIMATE_GROUP_NEW", safe_mode) if "CLIMATE_GROUP_NEW" in df_report.columns else ("LOCATION", safe_mode),
        Rating=("REVIEW_RATING", "mean"),
        Freq=("FREQUENCY_VALUE_NEW", "mean"),
        Promo_Pct=("PROMO_USED_VAL", "mean"),
        FitScore=("CLIMATE_ITEM_FIT_SCORE_NEW", "mean") if "CLIMATE_ITEM_FIT_SCORE_NEW" in df_report.columns else ("SUBSCRIPTION", "mean"),
        RelSpend=("REL_SPEND_CAT_NEW", "mean") if "REL_SPEND_CAT_NEW" in df_report.columns else ("SPEND_PER_PURCHASE_NEW", "mean"),
        TotWght=("TOTAL_SPEND_WEIGHTED_NEW", "mean"),
    ).reset_index()

    segment_profiles["Sub_Pct"] = segment_profiles["Sub_Pct"] * 100
    segment_profiles["Promo_Pct"] = segment_profiles["Promo_Pct"] * 100

    cluster_name_map = {
        3: "VIP'e En Yakınlar",
        0: "Avantaj Avcıları",
        2: "Büyük Sepetliler",
        4: "Kararsızlar",
        1: "Sessiz Kitle"
    }
    segment_profiles["Segment İsmi"] = (
        segment_profiles["Cluster"].map(cluster_name_map).fillna("Genel Segment")
    )

    cluster_action_map = {
        3: "Upsell / Premium",
        0: "Upsell / Premium",
        2: "Nurture / Education",
        4: "Winback / Aggressive Promo",
        1: "Winback / Aggressive Promo",
    }
    segment_profiles["Önerilen Aksiyon"] = segment_profiles["Cluster"].map(cluster_action_map).fillna("Genel")

    display_df = segment_profiles[[
        "Cluster", "Segment İsmi", "N", "Yas", "Harcama_USD", "Sub_Pct",
        "PrevPur", "Odeme", "Kargo", "FitScore", "RelSpend",
        "Promo_Pct", "TotWght", "Freq", "Önerilen Aksiyon"
    ]].sort_values("Cluster")

    display_df = display_df.reset_index(drop=True)

    st.dataframe(
        display_df.reset_index(drop=True).style
        .background_gradient(cmap="Blues", subset=["TotWght", "Sub_Pct", "Promo_Pct"])
        .format({
            "Yas": "{:.1f}",
            "Harcama_USD": "{:.1f}",
            "Sub_Pct": "{:.1f}%",
            "PrevPur": "{:.1f}",
            "FitScore": "{:.4f}",
            "RelSpend": "{:.2f}",
            "Promo_Pct": "{:.1f}%",
            "TotWght": "{:.1f}",
            "Freq": "{:.1f}",
        })
    )

    # ✅ Profili session state'e kaydet
    st.session_state["profile"] = segment_profiles

    st.divider()

    # Playbook / expander
    st.subheader("💡 Segment Bazlı Aksiyon Playbook")

    for _, r in display_df.iterrows():
        cl = int(r["Cluster"])
        with st.expander(f"📌 Cluster {cl} – {r['Segment İsmi']} ({r['Önerilen Aksiyon']})"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Müşteri", f"{r['N']:.0f}")
                st.metric("Abonelik", f"{r['Sub_Pct']:.1f}%")

            with c2:
                st.metric("TotWght", f"{r['TotWght']:.1f}")
                st.metric("Ort. Harcama", f"{r['Harcama_USD']:.1f}")

            with c3:
                st.metric("Promo", f"{r['Promo_Pct']:.1f}%")
                st.metric("Frekans", f"{r['Freq']:.1f}")

            if cl in [3, 0]:
                st.success("✅ Upsell / Premium")
                st.write("• Premium/Plus abonelik: ücretsiz kargo + özel kampanya erişimi")
                st.write("• Checkout ve satın alma sonrası 1 tık abonelik önerisi")
                st.write("• 30 gün deneme veya ilk 3 ay indirim (A/B test)")
            elif cl == 2:
                st.info("ℹ️ Nurture / Education")
                st.write("• Tasarruf simülasyonu: 'Abone olsaydınız X₺ daha az öderdiniz'")
                st.write("• Fayda anlatımı: fiyat değil, değer ve avantaj")
                st.write("• Email drip: 3 adım (fayda → örnek hesap → CTA)")
            else:
                st.error("🔴 Winback / Aggressive Promo")
                st.write("• 48 saatlik teklif + FOMO mesaj")
                st.write("• SMS/Push ağırlıklı yeniden aktivasyon")
                st.write("• Kısa anket + kişiselleştirme")

# =============================================================================
# TAB 3: MODEL EĞİTİMİ
# =============================================================================
with tab_model:
    st.header("🎯 Model Eğitimi (Leakage-Free Pipeline)")
    
    if st.button("🚀 Modeli Eğit"):
        with st.spinner("Model eğitiliyor..."):
            
            # Conditional probabilities
            probs_cat = fit_conditional_probs(df_eng_train, "CLIMATE_GROUP_NEW", "CATEGORY", smoothing=1.0)
            df_eng_train["P_CATEGORY_given_CLIMATE_NEW"] = map_conditional_probs(df_eng_train, probs_cat, "CLIMATE_GROUP_NEW", "CATEGORY")
            df_eng_test["P_CATEGORY_given_CLIMATE_NEW"] = map_conditional_probs(df_eng_test, probs_cat, "CLIMATE_GROUP_NEW", "CATEGORY")
            df_eng_test["P_CATEGORY_given_CLIMATE_NEW"].fillna(df_eng_train["P_CATEGORY_given_CLIMATE_NEW"].mean(), inplace=True)
            
            probs_size = fit_conditional_probs(df_eng_train, "CLIMATE_GROUP_NEW", "SIZE", smoothing=1.0)
            df_eng_train["P_SIZE_given_CLIMATE_NEW"] = map_conditional_probs(df_eng_train, probs_size, "CLIMATE_GROUP_NEW", "SIZE")
            df_eng_test["P_SIZE_given_CLIMATE_NEW"] = map_conditional_probs(df_eng_test, probs_size, "CLIMATE_GROUP_NEW", "SIZE")
            df_eng_test["P_SIZE_given_CLIMATE_NEW"].fillna(df_eng_train["P_SIZE_given_CLIMATE_NEW"].mean(), inplace=True)
            
            probs_season = fit_conditional_probs(df_eng_train, "CLIMATE_GROUP_NEW", "SEASON", smoothing=1.0)
            df_eng_train["P_SEASON_given_CLIMATE_NEW"] = map_conditional_probs(df_eng_train, probs_season, "CLIMATE_GROUP_NEW", "SEASON")
            df_eng_test["P_SEASON_given_CLIMATE_NEW"] = map_conditional_probs(df_eng_test, probs_season, "CLIMATE_GROUP_NEW", "SEASON")
            df_eng_test["P_SEASON_given_CLIMATE_NEW"].fillna(df_eng_train["P_SEASON_given_CLIMATE_NEW"].mean(), inplace=True)
            
            # Fit scores
            df_eng_train["CLIMATE_ITEM_FIT_SCORE_NEW"] = (
                df_eng_train["P_CATEGORY_given_CLIMATE_NEW"] *
                df_eng_train["P_SIZE_given_CLIMATE_NEW"] *
                df_eng_train["P_SEASON_given_CLIMATE_NEW"]
            )
            df_eng_test["CLIMATE_ITEM_FIT_SCORE_NEW"] = (
                df_eng_test["P_CATEGORY_given_CLIMATE_NEW"] *
                df_eng_test["P_SIZE_given_CLIMATE_NEW"] *
                df_eng_test["P_SEASON_given_CLIMATE_NEW"]
            )
            
            # Group mean ratios
            df_eng_train, df_eng_test = add_group_mean_ratio(df_eng_train, df_eng_test, "CATEGORY", "PURCHASE_AMOUNT_(USD)", "REL_SPEND_CAT_NEW", "global_mean")
            df_eng_train, df_eng_test = add_group_mean_ratio(df_eng_train, df_eng_test, "CLIMATE_GROUP_NEW", "PURCHASE_AMOUNT_(USD)", "PURCHASE_AMT_REL_CLIMATE_NEW", "global_mean")
            df_eng_train, df_eng_test = add_group_mean_ratio(df_eng_train, df_eng_test, "AGE_NEW", "PURCHASE_AMOUNT_(USD)", "REL_SPEND_AGE_NEW", "global_mean")
            df_eng_train, df_eng_test = add_group_mean_ratio(df_eng_train, df_eng_test, "CLIMATE_GROUP_NEW", "FREQUENCY_VALUE_NEW", "REL_FREQ_CLIMATE_NEW", "global_mean")
            
            # Encoding
            drop_cols = [
                'CUSTOMER_ID','SUBSCRIPTION_STATUS', 'ITEM_PURCHASED', 'LOCATION', 'COLOR', 'SIZE',
                'FREQUENCY_OF_PURCHASES', 'PAYMENT_METHOD', 'SHIPPING_TYPE',
                'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 'REVIEW_RATING',
                'AGE', 'DISCOUNT_APPLIED', 'SEASON', 'PROMO_CODE_USED'
            ]
            
            X_train_df, X_test_df = encode_train_test(df_eng_train, df_eng_test, drop_cols)
            
            y_train = (df_eng_train["SUBSCRIPTION_STATUS"] == "Yes").astype(int)
            y_test = (df_eng_test["SUBSCRIPTION_STATUS"] == "Yes").astype(int)
            
            # Leakage temizliği
            leak_prefixes = ("SUB_FREQ_NEW", "PROMO_NO_SUB_NEW", "SHIP_SUB_NEW")
            leakage_cols = [c for c in X_train_df.columns if c.startswith(leak_prefixes)]
            
            X_train_base = X_train_df.drop(columns=leakage_cols, errors="ignore")
            X_test_base = X_test_df.drop(columns=leakage_cols, errors="ignore")
            
            # Feature selection
            rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
            rf_selector.fit(X_train_base, y_train)
            
            importances = pd.Series(rf_selector.feature_importances_, index=X_train_base.columns).sort_values(ascending=False)
            keep_cols = importances[importances >= 0.01].index.tolist()
            
            X_train = X_train_base[keep_cols]
            X_test = X_test_base[keep_cols]
            
            scaler_model = StandardScaler()
            X_train_s = scaler_model.fit_transform(X_train)
            X_test_s = scaler_model.transform(X_test)
            
            # Model karşılaştırma (5-fold CV)
            models = [
                ("LogisticRegression", LogisticRegression(max_iter=1000)),
                ("RandomForest", RandomForestClassifier(random_state=42, class_weight='balanced')),
                ("XGBoost", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)),
                ("LightGBM", LGBMClassifier(random_state=42, verbose=-1))
            ]
            
            best_model_name = None
            best_model_score = -1
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cv_results = []
            
            for idx, (name, model) in enumerate(models):
                status_text.text(f"Cross-validation yapılıyor: {name}...")
                cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                cv_results.append({
                    'Model': name,
                    'CV AUC Mean': mean_score,
                    'Std Dev': std_score
                })
                
                if mean_score > best_model_score:
                    best_model_score = mean_score
                    best_model_name = name
                
                progress_bar.progress((idx + 1) / len(models))
            
            status_text.text(f"✅ Kazanan model: {best_model_name}")
            
            # CV sonuçları
            st.subheader("📊 Cross-Validation Sonuçları")
            cv_df = pd.DataFrame(cv_results)
            st.dataframe(cv_df.style.background_gradient(cmap='Greens', subset=['CV AUC Mean']).format({
                'CV AUC Mean': '{:.4f}',
                'Std Dev': '{:.4f}'
            }))
            
            # Final model eğitimi
            st.subheader(f"🎯 Final Model Optimizasyonu: {best_model_name}")
            
            if best_model_name == "RandomForest":
                params = {
                    'n_estimators': [200, 400],
                    'max_depth': [8, 12, 16],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                estimator = RandomForestClassifier(random_state=42, class_weight='balanced')
            elif best_model_name == "XGBoost":
                params = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [200, 400],
                    'max_depth': [3, 5, 7]
                }
                estimator = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
            elif best_model_name == "LightGBM":
                params = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [200, 400],
                    'num_leaves': [20, 31, 50]
                }
                estimator = LGBMClassifier(random_state=42, verbose=-1)
            else:
                params = {'C': [0.1, 1, 10]}
                estimator = LogisticRegression(max_iter=1000)
            
            with st.spinner(f"{best_model_name} GridSearch ile optimize ediliyor..."):
                grid = GridSearchCV(estimator, params, cv=5, scoring='roc_auc', n_jobs=-1)
                grid.fit(X_train_s, y_train)
                final_model = grid.best_estimator_
            
            st.success("✅ Model eğitimi tamamlandı!")
            st.info(f"**En İyi Parametreler:** {grid.best_params_}")
            
            # Test tahmini
            if hasattr(final_model, 'predict_proba'):
                y_proba = final_model.predict_proba(X_test_s)[:, 1]
            else:
                y_proba = final_model.decision_function(X_test_s)
            
            # Threshold optimizasyonu
            st.subheader("🎯 Threshold Optimizasyonu")
            
            def find_best_threshold(y_true, y_prob, target_recall=0.85):
                thresholds = np.linspace(0.05, 0.95, 19)
                best = None
                for thr in thresholds:
                    y_pred_thr = (y_prob >= thr).astype(int)
                    rec = recall_score(y_true, y_pred_thr, zero_division=0)
                    prec = precision_score(y_true, y_pred_thr, zero_division=0)
                    f1 = f1_score(y_true, y_pred_thr, zero_division=0)
                    
                    if rec >= target_recall:
                        if (best is None) or (prec > best["precision"]):
                            best = {"thr": thr, "precision": prec, "recall": rec, "f1": f1}
                return best
            
            target_recall = 0.85
            best_thr_result = find_best_threshold(y_test, y_proba, target_recall=target_recall)
            
            if best_thr_result:
                best_thr = best_thr_result["thr"]
                st.session_state['best_threshold'] = best_thr
                
                col_thr1, col_thr2, col_thr3 = st.columns(3)
                col_thr1.metric("Optimal Threshold", f"{best_thr:.2f}")
                col_thr2.metric("Precision", f"{best_thr_result['precision']:.3f}")
                col_thr3.metric("Recall", f"{best_thr_result['recall']:.3f}")
                
                y_pred = (y_proba >= best_thr).astype(int)
            else:
                st.warning(f"Recall >= {target_recall} sağlayan threshold bulunamadı. Default 0.50 kullanılıyor.")
                best_thr = 0.50
                y_pred = (y_proba >= best_thr).astype(int)
            
            st.divider()
            
            # Performans metrikleri
            st.subheader("📊 Model Performansı")
            
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                           xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                ax_cm.set_xlabel('Tahmin')
                ax_cm.set_ylabel('Gerçek')
                ax_cm.set_title('Confusion Matrix')
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            
            with col_perf2:
                st.markdown("**ROC Curve**")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve')
                ax_roc.legend()
                ax_roc.grid(True, alpha=0.3)
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            
            # Classification report
            st.markdown("**Classification Report**")
            report = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn').format('{:.3f}'))
            
            # Feature importance
            st.subheader("🔥 Feature Importance")
            
            if hasattr(final_model, 'feature_importances_'):
                importances = final_model.feature_importances_
                feature_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(20)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
                sns.barplot(x=feature_imp.values, y=feature_imp.index, palette='viridis', ax=ax_imp)
                ax_imp.set_xlabel('Importance')
                ax_imp.set_ylabel('Feature')
                ax_imp.set_title('Top 20 Feature Importance')
                st.pyplot(fig_imp)
                plt.close(fig_imp)
            elif hasattr(final_model, 'coef_'):
                importances = final_model.coef_[0]
                feature_imp = pd.DataFrame({'Feature': X_train.columns, 'Coef': importances})
                feature_imp["Odds_Ratio"] = np.exp(feature_imp["Coef"])
                feature_imp['Abs_Coef'] = feature_imp['Coef'].abs()
                feature_imp = feature_imp.sort_values(by='Abs_Coef', ascending=False).head(20)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Coef', y='Feature', data=feature_imp, palette='coolwarm', ax=ax_imp)
                ax_imp.axvline(0, color='black', linestyle='--')
                ax_imp.set_title('Top 20 Feature Coefficients')
                st.pyplot(fig_imp)
                plt.close(fig_imp)
            
            # Session state'e kaydet
            st.session_state['final_model'] = final_model
            st.session_state['scaler_model'] = scaler_model
            st.session_state['X_columns'] = X_train.columns.tolist()
            st.session_state['y_proba_test'] = y_proba
            st.session_state['y_test'] = y_test
            st.session_state['best_model_name'] = best_model_name
    
    else:
        st.info("👆 Modeli eğitmek için yukarıdaki butona tıklayın.")

# =============================================================================
# TAB 4: MODEL KARŞILAŞTIRMA (Sadece özet - detaylar kaldırıldı)
# =============================================================================
with tab_comp:
    st.header("📄 Model Karşılaştırması")
    st.info("Bu bölümde farklı modellerin performanslarını karşılaştırabilirsiniz. Model Eğitimi sekmesinde CV sonuçlarını görüntüleyebilirsiniz.")

# =============================================================================
# TAB 5: CRM ANALİZİ
# =============================================================================
with tab_crm:
    st.header("💼 CRM ve Segment Bazlı Aksiyon Planı")
    
    if 'df_report' in st.session_state and st.session_state['df_report'] is not None:
        
        st.subheader("📊 Segment Bazlı Abonelik Tahmini")
        
        df_report = st.session_state['df_report']
        
        # Her segment için detaylı analiz
        crm_summary = df_report.groupby('Cluster').agg({
            'CUSTOMER_ID': 'count',
            'SUBSCRIPTION_STATUS': lambda x: (x == 'Yes').mean(),
            'TOTAL_SPEND_WEIGHTED_NEW': 'mean',
            'PREVIOUS_PURCHASES': 'mean',
            'FREQUENCY_VALUE_NEW': 'mean',
            'PROMO_USED_VAL': 'mean'
        }).round(3)
        
        crm_summary.columns = ['n_customers', 'crm_target_rate', 'avg_spend', 'avg_prev_purchases', 'avg_freq', 'promo_rate']
        
        # CRM Aksiyon Belirleme
        spend_median = crm_summary["avg_spend"].median()
        target_mean = crm_summary["crm_target_rate"].mean()
        
        def crm_action(row):
            if row["crm_target_rate"] >= target_mean and row["avg_spend"] >= spend_median:
                return "Upsell / Premium teklif"
            elif row["crm_target_rate"] >= target_mean:
                return "Quick win / light incentive"
            elif row["crm_target_rate"] < target_mean and row["avg_spend"] >= spend_median:
                return "Retention / özel ilgi"
            else:
                return "Winback / agresif promosyon"
        
        crm_summary['action'] = crm_summary.apply(crm_action, axis=1)
        
        # Segment isimlerini ekle
        if 'profile' in st.session_state and st.session_state['profile'] is not None:
            profile = st.session_state['profile']
            segment_names = dict(zip(profile['Cluster'], profile['Segment İsmi']))
            crm_summary['Segment İsmi'] = crm_summary.index.map(segment_names)
            crm_summary = crm_summary[['Segment İsmi', 'n_customers', 'crm_target_rate', 'avg_spend', 
                                       'avg_prev_purchases', 'avg_freq', 'promo_rate', 'action']]
        
        # Türkçe kolon isimleri
        crm_summary_display = crm_summary.rename(columns={
            'Segment İsmi': 'Segment',
            'n_customers': 'Müşteri Sayısı',
            'crm_target_rate': 'Abonelik Oranı',
            'avg_spend': 'Ort. Harcama',
            'avg_prev_purchases': 'Ort. Alışveriş',
            'avg_freq': 'Ort. Frekans',
            'promo_rate': 'Promo Kullanım',
            'action': 'Önerilen Aksiyon'
        })
        
        crm_summary_display['Abonelik Oranı'] = (crm_summary_display['Abonelik Oranı'] * 100).round(1)
        crm_summary_display['Promo Kullanım'] = (crm_summary_display['Promo Kullanım'] * 100).round(1)
        
        crm_summary_display = crm_summary_display.sort_values('Abonelik Oranı', ascending=False)
        
        st.dataframe(crm_summary_display.style.background_gradient(
            cmap='RdYlGn', 
            subset=['Abonelik Oranı', 'Ort. Harcama']
        ).format({
            'Abonelik Oranı': '{:.1f}%',
            'Ort. Harcama': '${:.2f}',
            'Ort. Alışveriş': '{:.1f}',
            'Ort. Frekans': '{:.1f}',
            'Promo Kullanım': '{:.1f}%'
        }))
        
        st.info(f"""
        📊 **CRM Eşik Değerleri:**
        - Abonelik Ortalaması: %{target_mean*100:.1f}
        - Harcama Medyanı: ${spend_median:.2f}
        """)
        
        st.divider()
        
        # Aksiyon Öncelik Grafiği
        st.subheader("🎯 Aksiyon Öncelik Matrisi")
        
        fig_matrix, ax_matrix = plt.subplots(figsize=(12, 8))
        
        colors_map = {
            'Upsell / Premium teklif': '#28a745',
            'Quick win / light incentive': '#17a2b8',
            'Retention / özel ilgi': '#ff8c00',
            'Winback / agresif promosyon': '#dc3545'
        }
        
        for action in crm_summary['action'].unique():
            mask = crm_summary['action'] == action
            ax_matrix.scatter(
                crm_summary[mask]['crm_target_rate'] * 100,
                crm_summary[mask]['avg_spend'],
                s=crm_summary[mask]['n_customers'] * 2,
                c=colors_map.get(action, '#999999'),
                label=action,
                alpha=0.6,
                edgecolors='white',
                linewidth=2
            )
        
        ax_matrix.axvline(target_mean * 100, color='purple', linestyle='--', linewidth=1.5, alpha=0.6, 
                         label=f'Abonelik Ortalaması: {target_mean*100:.1f}%')
        ax_matrix.axhline(spend_median, color='blue', linestyle='--', linewidth=1.5, alpha=0.6, 
                         label=f'Harcama Medyanı: ${spend_median:.2f}')
        
        ax_matrix.set_xlabel('Abonelik Oranı (%)', fontsize=12)
        ax_matrix.set_ylabel('Ortalama Harcama ($)', fontsize=12)
        ax_matrix.set_title('CRM Aksiyon Öncelik Matrisi (Balon Boyutu = Müşteri Sayısı)', fontsize=14)
        ax_matrix.legend(loc='best', fontsize=9)
        ax_matrix.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_matrix)
        plt.close(fig_matrix)
    
    else:
        st.warning("⚠️ CRM analizi için önce Segmentasyon sekmesine gidin.")

# =============================================================================
# TAB 6: SİMÜLATÖR
# =============================================================================
with tab_sim:
    st.header("🧪 Canlı Tahmin Simülatörü")
    
    if 'final_model' not in st.session_state or st.session_state['final_model'] is None:
        st.warning("⚠️ Simülatörü kullanmak için önce 'Model Eğitimi' sekmesinden modeli eğitmelisiniz.")
    else:
        with st.form("sim_form"):
            c1, c2, c3 = st.columns(3)
            age = c1.slider("Yaş", 18, 70, 30)
            gender = c2.selectbox("Cinsiyet", df_raw['GENDER'].unique())
            spend = c3.number_input("Harcama Tutarı ($)", 1, 1000, 100)
            prev = c1.number_input("Geçmiş Alışveriş", 0, 100, 5)
            freq = c2.selectbox("Sıklık", df_raw['FREQUENCY_OF_PURCHASES'].unique())
            rating = c3.slider("Rating", 1.0, 5.0, 4.0)
            cat = c1.selectbox("Kategori", df_raw['CATEGORY'].unique())
            loc = c2.selectbox("Lokasyon", df_raw['LOCATION'].unique())
            promo = c3.selectbox("Promosyon Kullandı mı?", ["Yes", "No"])
            
            btn = st.form_submit_button("🔮 Tahmin Et")
        
        if btn:
            try:
                # Basitleştirilmiş feature'lar
                freq_map = {'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26, 'Quarterly': 4, 'Annually': 1, 'Monthly': 12, 'Every 3 Months': 4}
                freq_val = freq_map.get(freq, 12)
                
                total_spend = prev * spend
                spend_per_purchase = spend / (prev + 1)
                
                # Loyalty score
                if prev < 13:
                    loyalty_score = 1
                elif prev < 25:
                    loyalty_score = 2
                elif prev < 38:
                    loyalty_score = 3
                else:
                    loyalty_score = 4
                
                # Feature dictionary
                simple_features = {
                    'TOTAL_SPEND_WEIGHTED_NEW': total_spend,
                    'SPEND_PER_PURCHASE_NEW': spend_per_purchase,
                    'FREQUENCY_VALUE_NEW': freq_val,
                    'LOYALTY_SCORE_NEW': loyalty_score,
                    'HIGH_REVIEW_RATING_NEW': 1 if rating >= 4 else 0,
                    'SPEND_RATING_NEW': spend * rating,
                    'PROMO_X_LOYALTY': (1 if promo == 'Yes' else 0) * loyalty_score,
                    'PROMO_X_FREQ': (1 if promo == 'Yes' else 0) * freq_val
                }
                
                # Gender encoding
                if gender == 'Male':
                    simple_features['GENDER_Male'] = 1
                    simple_features['GENDER_Female'] = 0
                else:
                    simple_features['GENDER_Male'] = 0
                    simple_features['GENDER_Female'] = 1
                
                # Category one-hot
                categories = df_raw['CATEGORY'].unique()
                for category in categories:
                    simple_features[f'CATEGORY_{category}'] = 1 if cat == category else 0
                
                # DataFrame oluştur
                feature_df = pd.DataFrame([simple_features])
                
                # Model'in beklediği kolonları ekle
                X_columns = st.session_state['X_columns']
                for col in X_columns:
                    if col not in feature_df.columns:
                        feature_df[col] = 0
                
                feature_df = feature_df[X_columns]
                
                # Scale
                scaler_model = st.session_state['scaler_model']
                user_scaled = scaler_model.transform(feature_df)
                
                # Tahmin
                final_model = st.session_state['final_model']
                prob = final_model.predict_proba(user_scaled)[0][1]
                
                # Cluster tahmini
                predicted_cluster = None
                segment_name = "Bilinmiyor"
                
                if 'kmeans' in st.session_state and 'scaler_seg' in st.session_state:
                    try:
                        segmentation_features = np.array([[
                            spend,
                            prev,
                            freq_val,
                            spend_per_purchase,
                            total_spend
                        ]])
                        
                        user_seg_scaled = st.session_state['scaler_seg'].transform(segmentation_features)
                        predicted_cluster = st.session_state['kmeans'].predict(user_seg_scaled)[0]
                        
                        # Segment ismini al
                        if 'profile' in st.session_state and st.session_state['profile'] is not None:
                            profile = st.session_state['profile']
                            matching_row = profile[profile['Cluster'] == predicted_cluster]
                            if not matching_row.empty:
                                segment_name = matching_row.iloc[0]['Segment İsmi']
                    except Exception as e:
                        st.warning(f"Cluster tahmini yapılamadı: {str(e)}")
                
                thr = st.session_state['best_threshold']
                
                st.divider()
                
                # 3 kolonlu layout
                col_r1, col_r2, col_r3 = st.columns([1, 1, 1.5])
                
                with col_r1:
                    st.subheader("🎯 Abonelik Tahmini")
                    if prob >= thr:
                        st.success(f"### ✅ ABONE OLUR")
                        st.metric("İhtimal", f"%{prob*100:.1f}")
                    else:
                        st.error(f"### ❌ ABONE OLMAZ")
                        st.metric("İhtimal", f"%{prob*100:.1f}")
                    
                    st.caption(f"Threshold: %{thr*100:.0f}")
                    st.progress(prob)
                
                with col_r2:
                    st.subheader("🧩 Segment Tahmini")
                    if predicted_cluster is not None:
                        st.info(f"### Cluster {predicted_cluster}")
                        st.success(f"**{segment_name}**")
                    else:
                        st.warning("Segment tahmini yapılamadı")
                
                with col_r3:
                    st.subheader("📋 Müşteri Profili")
                    profile_col1, profile_col2 = st.columns(2)
                    
                    with profile_col1:
                        st.write(f"👤 **Yaş:** {age}")
                        st.write(f"🚹🚺 **Cinsiyet:** {gender}")
                        st.write(f"📍 **Lokasyon:** {loc}")
                        st.write(f"🛒 **Kategori:** {cat}")
                    
                    with profile_col2:
                        st.write(f"💰 **Harcama:** ${spend}")
                        st.write(f"📦 **Geçmiş Alışveriş:** {prev}")
                        st.write(f"🔄 **Sıklık:** {freq}")
                        st.write(f"⭐ **Rating:** {rating}")
                    
                    st.write(f"🎁 **Promosyon:** {promo}")
            
            except Exception as e:
                st.error(f"❌ Tahmin yapılırken hata oluştu: {str(e)}")
                st.info("💡 Lütfen önce 'Model Eğitimi' sekmesinden modeli eğittiğinizden emin olun.")