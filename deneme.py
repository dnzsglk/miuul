import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
st.set_page_config(page_title="Miuul Alışveriş Analizi", page_icon="🛍️", layout="wide")

# CSS, Kar ve Müzik
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

local_css("style.css")

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

# =============================================================================
# PIPELINE FONKSİYONLARI (AYNEN PYTHON PIPELINE'DAN)
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
    confusion_matrix_table = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix_table)[0]
    n = confusion_matrix_table.sum().sum()
    r, k = confusion_matrix_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def process_data(df):
    df_eng = df.copy()
    
    num_force_cols = ["PURCHASE_AMOUNT_(USD)", "PREVIOUS_PURCHASES", "AGE", "REVIEW_RATING"]
    for c in num_force_cols:
        if c in df_eng.columns:
            df_eng[c] = pd.to_numeric(df_eng[c], errors="coerce")
    
    df_eng['TEMP_TARGET'] = df_eng['SUBSCRIPTION_STATUS'].map({"Yes": 1, "No": 0})
    
    df_eng['TOTAL_SPEND_WEIGHTED_NEW'] = df_eng['PREVIOUS_PURCHASES'] * df_eng['PURCHASE_AMOUNT_(USD)']
    df_eng['SPEND_PER_PURCHASE_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / (df_eng['PREVIOUS_PURCHASES'] + 1)
    
    freq_map = {'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26, 'Quarterly': 4, 'Annually': 1, 'Monthly': 12, 'Every 3 Months': 4}
    df_eng['FREQUENCY_VALUE_NEW'] = df_eng['FREQUENCY_OF_PURCHASES'].map(freq_map)
    
    pay_map = {'Cash': 'Cash', 'Credit Card': 'Card', 'Debit Card': 'Card', 'PayPal': 'Online', 'Venmo': 'Online', 'Bank Transfer': 'Online'}
    df_eng['PAYMENT_TYPE_NEW'] = df_eng['PAYMENT_METHOD'].map(pay_map)
    
    df_eng["AGE_NEW"] = pd.cut(df_eng["AGE"], bins=[0, 30, 45, 56, df_eng["AGE"].max()], labels=["18-30", "31-45", "46-56", "57-70"], include_lowest=True)
    df_eng["PURCHASE_AMOUNT_(USD)_NEW"] = pd.qcut(df_eng["PURCHASE_AMOUNT_(USD)"], q=4, labels=["Low", "Mid", "High", "Very High"])
    df_eng["LOYALTY_LEVEL_NEW"] = pd.cut(df_eng["PREVIOUS_PURCHASES"], bins=[0, 13, 25, 38, 50], labels=["Low", "Mid", "High", "Very High"], include_lowest=True)
    
    df_eng["SUB_FREQ_NEW"] = (df_eng["TEMP_TARGET"].astype(str) + "_" + df_eng["FREQUENCY_OF_PURCHASES"].astype(str))
    df_eng["PROMO_NO_SUB_NEW"] = ((df_eng["PROMO_CODE_USED"] == "Yes") & (df_eng["TEMP_TARGET"] == 0)).astype(int)
    df_eng["SHIP_SUB_NEW"] = (df_eng["SHIPPING_TYPE"].astype(str) + "_" + df_eng["TEMP_TARGET"].astype(str))
    
    df_eng["SEASON_CATEGORY_NEW"] = df_eng["SEASON"].astype(str) + "_" + df_eng["CATEGORY"].astype(str)
    df_eng["SEASON_COLOR_NEW"] = df_eng["SEASON"].astype(str) + "_" + df_eng["COLOR"].astype(str)
    df_eng["ITEM_CATEGORY_NEW"] = df_eng["CATEGORY"].astype(str) + "_" + df_eng["ITEM_PURCHASED"].astype(str)
    df_eng["HIGH_REVIEW_RATING_NEW"] = (df_eng["REVIEW_RATING"] >= 4).astype(int)
    df_eng["SPEND_RATING_NEW"] = df_eng["PURCHASE_AMOUNT_(USD)"] * df_eng["REVIEW_RATING"]
    
    top_locations = df_eng["LOCATION"].value_counts().nlargest(10).index
    df_eng["LOCATION_GROUPED_NEW"] = df_eng["LOCATION"].where(df_eng["LOCATION"].isin(top_locations), "Other")
    
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
    
    df_eng.drop(columns=['TEMP_TARGET'], inplace=True)
    
    drop_cols = ['CUSTOMER_ID', 'ITEM_PURCHASED', 'LOCATION', 'COLOR', 'SIZE',
                 'FREQUENCY_OF_PURCHASES', 'PAYMENT_METHOD', 'SHIPPING_TYPE',
                 'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 'REVIEW_RATING',
                 'AGE', 'DISCOUNT_APPLIED', 'SEASON', 'PROMO_CODE_USED']
    
    cols_to_drop = [c for c in drop_cols if c in df_eng.columns]
    df_model = df_eng.drop(columns=cols_to_drop)
    
    cat_cols = [col for col in df_model.columns if df_model[col].dtype == 'O' or df_model[col].dtype.name == 'category']
    binary_cols = [col for col in cat_cols if df_model[col].nunique() <= 2]
    multi_cols = [col for col in cat_cols if df_model[col].nunique() > 2]
    
    le = LabelEncoder()
    for col in binary_cols:
        df_model[col] = le.fit_transform(df_model[col])
    
    df_encoded = pd.get_dummies(df_model, columns=multi_cols, drop_first=False)
    
    return df_eng, df_encoded

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
# UYGULAMA
# =============================================================================

st.title("🛍️ Alışveriş Davranışları: Pipeline Analitik Panel")
st.markdown("Bu panel, Python pipeline'ınızdaki tüm adımları takip eder ve aynı sonuçları üretir.")

st.sidebar.header("📂 Veri Yönetimi")
uploaded_file = st.sidebar.file_uploader("CSV Dosyanızı Yükleyin", type=["csv"])

if uploaded_file is None:
    st.info("Analize başlamak için 'shopping_behavior_updated.csv' dosyasını yükleyin.")
    st.stop()

@st.cache_data
def get_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.upper().str.replace(" ", "_").str.strip()
    return df

df_raw = get_data(uploaded_file)

# SESSION STATE
if 'pipeline_completed' not in st.session_state:
    st.session_state['pipeline_completed'] = False

# Sekmeler
tab_info, tab_run = st.tabs(["ℹ️ Bilgi", "🚀 Pipeline Çalıştır"])

with tab_info:
    st.header("📋 Pipeline Bilgileri")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Müşteri Sayısı", df_raw.shape[0])
    col2.metric("Ortalama Yaş", f"{df_raw['AGE'].mean():.1f}")
    col3.metric("Abonelik Oranı", f"%{(df_raw['SUBSCRIPTION_STATUS']=='Yes').mean()*100:.1f}")
    col4.metric("Ortalama Harcama", f"${df_raw['PURCHASE_AMOUNT_(USD)'].mean():.1f}")
    
    st.divider()
    
    st.subheader("🔄 Pipeline Adımları")
    st.markdown("""
    1. **Veri Yükleme ve EDA**
    2. **Rare Encoding** (< %1)
    3. **Feature Engineering** (60+ yeni feature)
    4. **Train/Test Split** (%80/%20)
    5. **Conditional Probabilities** (Leakage-free)
    6. **Group Mean Ratios**
    7. **Encoding** (Label + One-Hot)
    8. **Segmentasyon** (K-Means + Elbow + Silhouette)
    9. **Feature Selection** (Threshold: 0.01)
    10. **Model Karşılaştırma** (5-Fold CV)
    11. **GridSearch Optimization**
    12. **Threshold Optimization** (Recall ≥ 85%)
    13. **CRM Analizi**
    """)

with tab_run:
    st.header("🚀 Pipeline Çalıştır")
    
    if st.button("▶️ Tüm Pipeline'ı Çalıştır"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # ADIM 1: Rare Encoding
            status_text.text("⏳ Rare encoding...")
            progress_bar.progress(10)
            df_rare = rare_encoder(df_raw, 0.01)
            
            # Correlation check
            if 'DISCOUNT_APPLIED' in df_rare.columns and 'PROMO_CODE_USED' in df_rare.columns:
                cv_score = cramers_v(df_rare['DISCOUNT_APPLIED'], df_rare['PROMO_CODE_USED'])
                if cv_score > 0.8:
                    df_rare.drop(columns=['DISCOUNT_APPLIED'], inplace=True)
            
            # ADIM 2: Feature Engineering
            status_text.text("⏳ Feature engineering...")
            progress_bar.progress(20)
            df_eng, df_encoded = process_data(df_rare)
            
            # ADIM 3: Train/Test Split
            status_text.text("⏳ Train/Test split...")
            progress_bar.progress(25)
            df_eng_train, df_eng_test = train_test_split(df_eng, test_size=0.20, random_state=42, stratify=df_eng["SUBSCRIPTION_STATUS"])
            
            # ADIM 4: Conditional Probabilities
            status_text.text("⏳ Conditional probabilities...")
            progress_bar.progress(30)
            
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
            
            # ADIM 5: Group Mean Ratios
            status_text.text("⏳ Group mean ratios...")
            progress_bar.progress(35)
            
            df_eng_train, df_eng_test = add_group_mean_ratio(df_eng_train, df_eng_test, "CATEGORY", "PURCHASE_AMOUNT_(USD)", "REL_SPEND_CAT_NEW", "global_mean")
            df_eng_train, df_eng_test = add_group_mean_ratio(df_eng_train, df_eng_test, "CLIMATE_GROUP_NEW", "PURCHASE_AMOUNT_(USD)", "PURCHASE_AMT_REL_CLIMATE_NEW", "global_mean")
            df_eng_train, df_eng_test = add_group_mean_ratio(df_eng_train, df_eng_test, "AGE_NEW", "PURCHASE_AMOUNT_(USD)", "REL_SPEND_AGE_NEW", "global_mean")
            df_eng_train, df_eng_test = add_group_mean_ratio(df_eng_train, df_eng_test, "CLIMATE_GROUP_NEW", "FREQUENCY_VALUE_NEW", "REL_FREQ_CLIMATE_NEW", "global_mean")
            
            # ADIM 6: Encoding
            status_text.text("⏳ Encoding...")
            progress_bar.progress(40)
            
            drop_cols = [
                'CUSTOMER_ID','SUBSCRIPTION_STATUS', 'ITEM_PURCHASED', 'LOCATION', 'COLOR', 'SIZE',
                'FREQUENCY_OF_PURCHASES', 'PAYMENT_METHOD', 'SHIPPING_TYPE',
                'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 'REVIEW_RATING',
                'AGE', 'DISCOUNT_APPLIED', 'SEASON', 'PROMO_CODE_USED'
            ]
            
            X_train_df, X_test_df = encode_train_test(df_eng_train, df_eng_test, drop_cols)
            y_train = (df_eng_train["SUBSCRIPTION_STATUS"] == "Yes").astype(int)
            y_test = (df_eng_test["SUBSCRIPTION_STATUS"] == "Yes").astype(int)
            
            # ADIM 7: Segmentasyon
            status_text.text("⏳ K-Means segmentasyon...")
            progress_bar.progress(45)
            
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
            
            # Elbow Method
            wcss = []
            k_range = range(2, 11)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
                wcss.append(km.inertia_)
            
            # Optimal K
            p1 = np.array([k_range[0], wcss[0]])
            p2 = np.array([k_range[-1], wcss[-1]])
            dists = [np.abs(np.cross(p2-p1, p1-np.array([k_range[i], wcss[i]]))) / np.linalg.norm(p2-p1) for i in range(len(wcss))]
            optimal_k = k_range[np.argmax(dists)]
            
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            sil_score = silhouette_score(X_scaled, clusters)
            
            # ADIM 8: Leakage Removal + Feature Selection
            status_text.text("⏳ Feature selection...")
            progress_bar.progress(50)
            
            leak_prefixes = ("SUB_FREQ_NEW", "PROMO_NO_SUB_NEW", "SHIP_SUB_NEW")
            leakage_cols = [c for c in X_train_df.columns if c.startswith(leak_prefixes)]
            
            X_train_base = X_train_df.drop(columns=leakage_cols, errors="ignore")
            X_test_base = X_test_df.drop(columns=leakage_cols, errors="ignore")
            
            rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
            rf_selector.fit(X_train_base, y_train)
            
            importances = pd.Series(rf_selector.feature_importances_, index=X_train_base.columns).sort_values(ascending=False)
            keep_cols = importances[importances >= 0.01].index.tolist()
            
            X_train = X_train_base[keep_cols]
            X_test = X_test_base[keep_cols]
            
            scaler_model = StandardScaler()
            X_train_s = scaler_model.fit_transform(X_train)
            X_test_s = scaler_model.transform(X_test)
            
            # ADIM 9: Model Karşılaştırma
            status_text.text("⏳ Model karşılaştırma (5-Fold CV)...")
            progress_bar.progress(60)
            
            models = [
                ("LogisticRegression", LogisticRegression(max_iter=1000)),
                ("RandomForest", RandomForestClassifier(random_state=42, class_weight='balanced')),
                ("XGBoost", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)),
                ("LightGBM", LGBMClassifier(random_state=42, verbose=-1))
            ]
            
            best_model_name = None
            best_model_score = -1
            cv_results = []
            
            for name, model in models:
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
            
            # ADIM 10: GridSearch
            status_text.text(f"⏳ GridSearch optimization ({best_model_name})...")
            progress_bar.progress(75)
