import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from scipy.stats import chi2_contingency
import io
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
                             f1_score, roc_curve, auc)

# Ayarlar
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Miuul Alışveriş Analizi", page_icon="🛍️", layout="wide")

# CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

local_css("style.css")

# Kar taneleri ve müzik
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

def add_conditional_freq_feature(df, group_col, cat_col, prefix=None, smoothing=1.0):
    if prefix is None: prefix = f"P_{cat_col}_given_{group_col}"
    ct = pd.crosstab(df[group_col], df[cat_col])
    ct_smoothed = ct + smoothing
    probs = ct_smoothed.div(ct_smoothed.sum(axis=1), axis=0)
    feat = df[[group_col, cat_col]].apply(lambda r: probs.loc[r[group_col], r[cat_col]], axis=1)
    df[f"{prefix}"] = feat
    return df

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

    if 'SUBSCRIPTION_STATUS' in df_eng.columns:
        df_eng['TEMP_TARGET'] = df_eng['SUBSCRIPTION_STATUS'].map({"Yes": 1, "No": 0})
    else:
        df_eng['TEMP_TARGET'] = 0 

    df_eng['TOTAL_SPEND_WEIGHTED_NEW'] = df_eng['PREVIOUS_PURCHASES'] * df_eng['PURCHASE_AMOUNT_(USD)']
    df_eng['SPEND_PER_PURCHASE_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / (df_eng['PREVIOUS_PURCHASES'] + 1)
    
    freq_map = {'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26, 'Quarterly': 4, 'Annually': 1, 'Monthly': 12, 'Every 3 Months': 4}
    df_eng['FREQUENCY_VALUE_NEW'] = df_eng['FREQUENCY_OF_PURCHASES'].map(freq_map)

    pay_map = {'Cash': 'Cash', 'Credit Card': 'Card', 'Debit Card': 'Card', 'PayPal': 'Online', 'Venmo': 'Online', 'Bank Transfer': 'Online'}
    df_eng['PAYMENT_TYPE_NEW'] = df_eng['PAYMENT_METHOD'].map(pay_map)

    df_eng["AGE_NEW"] = pd.cut(df_eng["AGE"], bins=[0, 30, 45, 56, 200], labels=["18-30", "31-45", "46-56", "57-70"])
    df_eng["PURCHASE_AMOUNT_(USD)_NEW"] = pd.qcut(df_eng["PURCHASE_AMOUNT_(USD)"], q=4, labels=["Low", "Mid", "High", "Very High"])
    df_eng["LOYALTY_LEVEL_NEW"] = pd.cut(df_eng["PREVIOUS_PURCHASES"], bins=[0, 13, 25, 38, 200], labels=["Low", "Mid", "High", "Very High"], include_lowest=True)

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
    
    top_locations = df_eng["LOCATION"].value_counts().nlargest(10).index
    df_eng["TOP_LOCATION_NEW"] = df_eng["LOCATION"].where(df_eng["LOCATION"].isin(top_locations), "Other")

    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "CATEGORY", prefix="P_CATEGORY_given_CLIMATE_NEW", smoothing=1.0)
    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "SIZE", prefix="P_SIZE_given_CLIMATE_NEW", smoothing=1.0)
    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "SEASON", prefix="P_SEASON_given_CLIMATE_NEW", smoothing=1.0)

    df_eng["CLIMATE_ITEM_FIT_SCORE_NEW"] = (
        df_eng["P_CATEGORY_given_CLIMATE_NEW"] *
        df_eng["P_SIZE_given_CLIMATE_NEW"] *
        df_eng["P_SEASON_given_CLIMATE_NEW"]
    )

    climate_spend_mean = df_eng.groupby("CLIMATE_GROUP_NEW")["PURCHASE_AMOUNT_(USD)"].transform("mean")
    df_eng["PURCHASE_AMT_REL_CLIMATE_NEW"] = df_eng["PURCHASE_AMOUNT_(USD)"] / climate_spend_mean

    df_eng["CLIMATE_LOYALTY_NEW"] = (df_eng["CLIMATE_GROUP_NEW"].astype(str) + "_" + df_eng["LOYALTY_LEVEL_NEW"].astype(str))
    df_eng["LOYALTY_SCORE_NEW"] = pd.qcut(df_eng["PREVIOUS_PURCHASES"], q=4, labels=[1, 2, 3, 4]).astype(int)

    cat_spend_mean = df_eng.groupby('CATEGORY')['PURCHASE_AMOUNT_(USD)'].transform('mean')
    df_eng['REL_SPEND_CAT_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / cat_spend_mean

    age_spend_mean = df_eng.groupby('AGE_NEW')['PURCHASE_AMOUNT_(USD)'].transform('mean')
    df_eng['REL_SPEND_AGE_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / age_spend_mean

    loc_freq_mean = df_eng.groupby('CLIMATE_GROUP_NEW')['FREQUENCY_VALUE_NEW'].transform('mean')
    df_eng['REL_FREQ_CLIMATE_NEW'] = df_eng['FREQUENCY_VALUE_NEW'] / loc_freq_mean
    
    df_eng["PROMO_X_LOYALTY"] = ((df_eng["PROMO_CODE_USED"] == "Yes").astype(int) * df_eng["LOYALTY_SCORE_NEW"])
    df_eng["PROMO_X_FREQ"] = ((df_eng["PROMO_CODE_USED"] == "Yes").astype(int) * df_eng["FREQUENCY_VALUE_NEW"])

    if 'TEMP_TARGET' in df_eng.columns: df_eng.drop(columns=['TEMP_TARGET'], inplace=True)

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

# =============================================================================
# UYGULAMA ARAYÜZÜ
# =============================================================================

st.title("🛍️ Alışveriş Davranışları: Hibrit Analitik Paneli")
st.markdown("""
Bu panel; **Gelişmiş EDA**, **Segmentasyon** ve **Çoklu Model Karşılaştırması** ile 
Precision odaklı bir strateji sunar.
""")

# --- SIDEBAR: VERİ YÜKLEME ---
st.sidebar.header("📂 Veri Yönetimi")
uploaded_file = st.sidebar.file_uploader("CSV Dosyanızı Buraya Sürükleyin", type=["csv"])

if uploaded_file is None:
    st.info("Analize başlamak için lütfen 'shopping_behavior_updated.csv' dosyasını yükleyin.")
    st.stop()

# --- VERİ YÜKLEME VE İŞLEME (CACHE) ---
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

# Sekmeler
tab_eda, tab_seg, tab_model, tab_comp, tab_sim = st.tabs([
    "📊 EDA Gelişmiş", 
    "🧩 Segmentasyon", 
    "🎯 Model (RF+Precision)", 
    "🔄 Model Karşılaştırma",
    "🧪 Simülatör"
])

# VERİ İŞLEME
with st.spinner('Veri işleniyor...'):
    df_rare = rare_encoder(df_raw, 0.01)
    if 'DISCOUNT_APPLIED' in df_rare.columns and 'PROMO_CODE_USED' in df_rare.columns:
        cv_score = cramers_v(df_rare['DISCOUNT_APPLIED'], df_rare['PROMO_CODE_USED'])
        if cv_score > 0.8:
            df_rare.drop(columns=['DISCOUNT_APPLIED'], inplace=True)
            
    df_eng, df_encoded = process_data_pipeline(df_rare)

# =============================================================================
# TAB 1: GELIŞMIŞ EDA
# =============================================================================
with tab_eda:
    st.header("📊 Gelişmiş Keşifsel Veri Analizi")
    
    # Genel Metrikler
    st.subheader("📈 Genel Metrikler")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Toplam Müşteri", df_raw.shape[0])
    col2.metric("Ortalama Yaş", f"{df_raw['AGE'].mean():.1f}")
    col3.metric("Abonelik Oranı", f"%{(df_raw['SUBSCRIPTION_STATUS']=='Yes').mean()*100:.1f}")
    col4.metric("Ortalama Harcama", f"${df_raw['PURCHASE_AMOUNT_(USD)'].mean():.1f}")
    col5.metric("Toplam Gelir", f"${df_raw['PURCHASE_AMOUNT_(USD)'].sum():,.0f}")
    
    st.divider()
    
    # Görselleştirmeler
    st.subheader("📊 Abonelik Odaklı Görselleştirmeler")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("**Abonelik Durumuna Göre Harcama Dağılımı**")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        for status in df_raw['SUBSCRIPTION_STATUS'].unique():
            data = df_raw[df_raw['SUBSCRIPTION_STATUS'] == status]['PURCHASE_AMOUNT_(USD)']
            sns.kdeplot(data, ax=ax1, label=status, fill=True, alpha=0.5)
        ax1.set_xlabel('Harcama Tutarı ($)')
        ax1.set_ylabel('Yoğunluk')
        ax1.set_title('Abonelik Durumuna Göre Harcama Dağılımı')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        st.markdown("**Kategori Bazlı Abonelik Oranları**")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        category_sub = df_raw.groupby('CATEGORY')['SUBSCRIPTION_STATUS'].apply(lambda x: (x=='Yes').sum() / len(x) * 100).sort_values(ascending=True)
        sns.barplot(x=category_sub.values, y=category_sub.index, ax=ax2, palette='viridis')
        ax2.set_xlabel('Abonelik Oranı (%)')
        ax2.set_ylabel('Kategori')
        ax2.set_title('Kategori Bazında Abonelik Oranları')
        ax2.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig2)
    
    with viz_col2:
        st.markdown("**Abonelik Durumuna Göre Yaş Dağılımı**")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=df_raw, x='SUBSCRIPTION_STATUS', y='AGE', ax=ax3, palette=['#d62828', '#28a745'])
        ax3.set_xlabel('Abonelik Durumu')
        ax3.set_ylabel('Yaş')
        ax3.set_title('Abonelik Durumuna Göre Yaş Dağılımı')
        ax3.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig3)
        
        st.markdown("**Geçmiş Alışveriş vs Abonelik**")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_raw, x='SUBSCRIPTION_STATUS', y='PREVIOUS_PURCHASES', ax=ax4, palette=['#d62828', '#28a745'])
        ax4.set_xlabel('Abonelik Durumu')
        ax4.set_ylabel('Geçmiş Alışveriş Sayısı')
        ax4.set_title('Geçmiş Alışveriş ve Abonelik İlişkisi')
        ax4.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig4)
    
    st.divider()
    
    # Abonelik İstatistikleri
    st.subheader("📈 Abonelik İstatistikleri")
    
    stat_col1, stat_col2 = st.columns(2)
    
    with stat_col1:
        st.markdown("**Promosyon Kullanımı vs Abonelik**")
        promo_sub = pd.crosstab(df_raw['PROMO_CODE_USED'], df_raw['SUBSCRIPTION_STATUS'], normalize='index') * 100
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        promo_sub.plot(kind='bar', ax=ax5, color=['#d62828', '#28a745'], rot=0)
        ax5.set_xlabel('Promosyon Kullanımı')
        ax5.set_ylabel('Yüzde (%)')
        ax5.set_title('Promosyon Kullanımı ve Abonelik İlişkisi')
        ax5.legend(title='Abonelik', labels=['No', 'Yes'])
        ax5.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig5)
    
    with stat_col2:
        st.markdown("**Cinsiyet Bazlı Abonelik Dağılımı**")
        gender_sub = pd.crosstab(df_raw['GENDER'], df_raw['SUBSCRIPTION_STATUS'], normalize='index') * 100
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        gender_sub.plot(kind='bar', ax=ax6, color=['#d62828', '#28a745'], rot=0)
        ax6.set_xlabel('Cinsiyet')
        ax6.set_ylabel('Yüzde (%)')
        ax6.set_title('Cinsiyet Bazında Abonelik Dağılımı')
        ax6.legend(title='Abonelik', labels=['No', 'Yes'])
        ax6.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig6)
    
    st.divider()
    
    # Korelasyon Matrisi
    st.subheader("🔥 Korelasyon Analizi")
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    corr_matrix = df_raw[numeric_cols].corr()
    
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax7, center=0)
    ax7.set_title('Korelasyon Matrisi')
    st.pyplot(fig7)

# =============================================================================
# TAB 2: SEGMENTASYON
# =============================================================================
with tab_seg:
    st.header("🧩 K-Means Müşteri Segmentasyonu")
    
    segmentation_cols = [
        'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 
        'FREQUENCY_VALUE_NEW', 'PROMO_CODE_USED', 
        'SPEND_PER_PURCHASE_NEW', 'LOYALTY_SCORE_NEW', 'CLIMATE_LOYALTY_NEW'
    ]
    
    X_seg = df_encoded[[c for c in segmentation_cols if c in df_encoded.columns]].copy()
    X_seg.fillna(0, inplace=True)
    
    scaler_seg = StandardScaler()
    X_scaled = scaler_seg.fit_transform(X_seg)
    
    # Elbow
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
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info(f"**Optimal Küme Sayısı (K): {optimal_k}**")
        fig_elb, ax = plt.subplots()
        plt.plot(k_range, wcss, 'bo--', linewidth=2, markersize=8)
        plt.axvline(optimal_k, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Küme Sayısı (K)')
        plt.ylabel('WCSS')
        plt.title("Elbow Method")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_elb)
        
    with c2:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = clusters
        
        fig_pca, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], 
                            cmap='viridis', s=50, alpha=0.6, edgecolors='w')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varyans)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varyans)')
        plt.title(f"Segment Dağılımı (K={optimal_k})")
        st.pyplot(fig_pca)
        
    # Profil
    df_report = df_eng.copy()
    df_report['Cluster'] = clusters
    df_report['PROMO_USED_VAL'] = df_report['PROMO_CODE_USED'].apply(lambda x: 1 if x=='Yes' else 0)
    
    st.subheader("📊 Segment Profilleri")
    profile = df_report.groupby('Cluster')[['AGE', 'TOTAL_SPEND_WEIGHTED_NEW', 'PROMO_USED_VAL']].mean()
    
    # Segment isimlendirme fonksiyonu (sütun rename'den ÖNCE)
    def name_segment(row):
        spend = row['TOTAL_SPEND_WEIGHTED_NEW']
        age = row['AGE']
        promo = row['PROMO_USED_VAL'] * 100
        
        # Harcama seviyesi
        if spend > df_report['TOTAL_SPEND_WEIGHTED_NEW'].quantile(0.75):
            spend_level = "VIP"
        elif spend > df_report['TOTAL_SPEND_WEIGHTED_NEW'].quantile(0.50):
            spend_level = "Yüksek Değerli"
        elif spend > df_report['TOTAL_SPEND_WEIGHTED_NEW'].quantile(0.25):
            spend_level = "Orta Değerli"
        else:
            spend_level = "Potansiyel"
        
        # Yaş grubu
        if age < 30:
            age_group = "Genç"
        elif age < 45:
            age_group = "Orta Yaş"
        else:
            age_group = "Olgun"
        
        # Promo kullanımı
        if promo > 60:
            promo_type = "Fırsat Avcısı"
        elif promo > 30:
            promo_type = "Promosyon Duyarlı"
        else:
            promo_type = "Sadık"
        
        return f"{spend_level} {age_group} {promo_type}"
    
    # Segment isimlerini ekle (rename'den ÖNCE)
    profile['Segment İsmi'] = profile.apply(name_segment, axis=1)
    
    # Şimdi sütun isimlerini değiştir
    profile = profile.rename(columns={
        'AGE': 'Ortalama Yaş',
        'TOTAL_SPEND_WEIGHTED_NEW': 'Toplam Harcama',
        'PROMO_USED_VAL': 'Promo Kullanım Oranı (%)'
    })
    profile['Promo Kullanım Oranı (%)'] = profile['Promo Kullanım Oranı (%)'] * 100
    
    # Sıralamayı değiştir: İsim önce
    profile = profile[['Segment İsmi', 'Ortalama Yaş', 'Toplam Harcama', 'Promo Kullanım Oranı (%)']]
    
    st.dataframe(profile.style.background_gradient(cmap='Blues', subset=['Ortalama Yaş', 'Toplam Harcama', 'Promo Kullanım Oranı (%)']).format({
        'Ortalama Yaş': '{:.1f}',
        'Toplam Harcama': '${:.2f}',
        'Promo Kullanım Oranı (%)': '{:.1f}%'
    }))
    
    st.divider()
    
    # RİSK ANALİZİ: Kaybetme Riski Yüksek Müşteriler
    st.subheader("⚠️ Risk Altındaki Müşteriler (Churn Risk)")
    
    # Abonelik durumu ile segment analizi
    df_report['SUBSCRIPTION'] = df_report['SUBSCRIPTION_STATUS'].map({'Yes': 1, 'No': 0})
    
    # Her segment için abonelik oranı
    segment_sub_rate = df_report.groupby('Cluster').agg({
        'SUBSCRIPTION': 'mean',
        'CUSTOMER_ID': 'count',
        'TOTAL_SPEND_WEIGHTED_NEW': 'mean',
        'PREVIOUS_PURCHASES': 'mean',
        'REVIEW_RATING': 'mean'
    }).round(3)
    
    segment_sub_rate.columns = ['Abonelik Oranı', 'Müşteri Sayısı', 'Ort. Harcama', 'Ort. Alışveriş Sayısı', 'Ort. Rating']
    segment_sub_rate['Abonelik Oranı'] = segment_sub_rate['Abonelik Oranı'] * 100
    
    # Segment isimlerini ekle
    segment_names = profile['Segment İsmi'].to_dict()
    segment_sub_rate['Segment İsmi'] = segment_sub_rate.index.map(segment_names)
    
    # Sıralama: Cluster numarasına göre (default)
    segment_sub_rate = segment_sub_rate.sort_index()
    segment_sub_rate = segment_sub_rate[['Segment İsmi', 'Müşteri Sayısı', 'Abonelik Oranı', 'Ort. Harcama', 'Ort. Alışveriş Sayısı', 'Ort. Rating']]
    
    st.dataframe(segment_sub_rate.style.background_gradient(cmap='RdYlGn', subset=['Abonelik Oranı', 'Ort. Rating']).format({
        'Abonelik Oranı': '{:.1f}%',
        'Ort. Harcama': '${:.2f}',
        'Ort. Alışveriş Sayısı': '{:.1f}',
        'Ort. Rating': '{:.2f}'
    }))
    
    # Aksiyon Önerileri
    st.subheader("💡 Önerilen Aksiyonlar")
    
    # Düşük abonelik oranlı segmentler
    low_sub_segments = segment_sub_rate[segment_sub_rate['Abonelik Oranı'] < segment_sub_rate['Abonelik Oranı'].mean()]
    
    if len(low_sub_segments) > 0:
        st.warning(f"⚠️ **{len(low_sub_segments)} segment ortalamanın altında abonelik oranına sahip!**")
        
        for idx, row in low_sub_segments.iterrows():
            with st.expander(f"📌 Cluster {idx}: {row['Segment İsmi']}"):
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    st.metric("Müşteri Sayısı", f"{row['Müşteri Sayısı']:.0f}")
                    st.metric("Abonelik Oranı", f"{row['Abonelik Oranı']:.1f}%")
                    st.metric("Ort. Harcama", f"${row['Ort. Harcama']:.2f}")
                
                with col_exp2:
                    st.metric("Ort. Alışveriş Sayısı", f"{row['Ort. Alışveriş Sayısı']:.1f}")
                    st.metric("Ort. Rating", f"{row['Ort. Rating']:.2f}")
                
                st.markdown("**🎯 Önerilen Aksiyonlar:**")
                
                if row['Abonelik Oranı'] < 30:
                    st.write("✅ Agresif abonelik kampanyası (ilk 3 ay %50 indirim)")
                elif row['Abonelik Oranı'] < 50:
                    st.write("✅ Orta düzey abonelik teşviki (ilk ay %30 indirim)")
                
                if row['Ort. Harcama'] > segment_sub_rate['Ort. Harcama'].mean():
                    st.write("✅ VIP müşteri programı sun (premium avantajlar)")
                
                if row['Ort. Rating'] < 3.5:
                    st.write("✅ Müşteri memnuniyeti anketleri ve iyileştirme planı")
                
                if row['Ort. Alışveriş Sayısı'] < 20:
                    st.write("✅ Sadakat programı ve tekrar satın alma teşvikleri")
                else:
                    st.write("✅ Sadık müşteri ödüllendirme programı")
    else:
        st.success("✅ Tüm segmentler ortalamanın üzerinde abonelik oranına sahip!")
    
    # Cluster boyutları
    st.subheader("📏 Segment Boyutları")
    cluster_sizes = pd.DataFrame(df_report['Cluster'].value_counts().sort_index())
    cluster_sizes.columns = ['Müşteri Sayısı']
    cluster_sizes['Yüzde'] = (cluster_sizes['Müşteri Sayısı'] / cluster_sizes['Müşteri Sayısı'].sum() * 100).round(2)
    st.dataframe(cluster_sizes.style.background_gradient(cmap='Greens'))

# =============================================================================
# TAB 3: MODEL (RF + PRECISION)
# =============================================================================
with tab_model:
    st.header("🎯 Random Forest Modeli (Precision Odaklı)")
    
    # Veri Hazırlığı
    leakage_cols = [c for c in df_encoded.columns if 'SUB_FREQ_NEW' in c or 'PROMO_NO_SUB_NEW' in c or 'SHIP_SUB_NEW' in c]
    X_temp = df_encoded.drop(columns=['SUBSCRIPTION_STATUS'] + leakage_cols)
    y = df_encoded['SUBSCRIPTION_STATUS']
    
    # Feature Selection
    rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_sel.fit(X_temp, y)
    importances = pd.Series(rf_sel.feature_importances_, index=X_temp.columns)
    keep_cols = importances[importances >= 0.01].index.tolist()
    
    X = X_temp[keep_cols]
    
    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler_model = StandardScaler()
    X_train_s = scaler_model.fit_transform(X_train)
    X_test_s = scaler_model.transform(X_test)
    
    # Model Eğitimi
    with st.spinner("Random Forest optimize ediliyor..."):
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [8, 12, 16],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                            rf_params, cv=3, scoring='precision', n_jobs=-1)
        grid.fit(X_train_s, y_train)
        final_model = grid.best_estimator_
        
        y_proba = final_model.predict_proba(X_test_s)[:, 1]
    
    st.success("✅ Model eğitimi tamamlandı!")
    st.info(f"**En İyi Parametreler:** {grid.best_params_}")
    
    # Threshold Optimizasyonu
    st.subheader("🎯 Otomatik Threshold Optimizasyonu")
    
    def eval_thr(y_true, y_prob, thr):
        y_p = (y_prob >= thr).astype(int)
        return {
            "thr": thr, 
            "precision": precision_score(y_true, y_p, zero_division=0), 
            "recall": recall_score(y_true, y_p, zero_division=0),
            "f1": f1_score(y_true, y_p, zero_division=0)
        }
        
    thresholds = np.linspace(0.05, 0.95, 19)
    res = [eval_thr(y_test, y_proba, t) for t in thresholds]
    df_thr = pd.DataFrame(res)
    
    target_recall = 0.80
    candidates = df_thr[df_thr["recall"] >= target_recall].sort_values("precision", ascending=False)
    
    if not candidates.empty:
        best_thr = candidates.iloc[0]["thr"]
        best_prec = candidates.iloc[0]["precision"]
        best_rec = candidates.iloc[0]["recall"]
        best_f1 = candidates.iloc[0]["f1"]
    else:
        best_thr = 0.50
        best_prec = 0.0
        best_rec = 0.0
        best_f1 = 0.0
    
    st.session_state['best_threshold'] = best_thr
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Önerilen Threshold", f"{best_thr:.2f}")
    col2.metric("Precision", f"%{best_prec*100:.1f}")
    col3.metric("Recall", f"%{best_rec*100:.1f}")
    col4.metric("F1-Score", f"%{best_f1*100:.1f}")
    
    # Threshold grafiği
    fig_thr, ax_thr = plt.subplots(figsize=(10, 5))
    ax_thr.plot(df_thr["thr"], df_thr["precision"], 'b-o', label='Precision', linewidth=2)
    ax_thr.plot(df_thr["thr"], df_thr["recall"], 'r-s', label='Recall', linewidth=2)
    ax_thr.plot(df_thr["thr"], df_thr["f1"], 'g-^', label='F1-Score', linewidth=2)
    ax_thr.axvline(best_thr, color='orange', linestyle='--', linewidth=2, label=f'Optimal: {best_thr:.2f}')
    ax_thr.set_xlabel('Threshold')
    ax_thr.set_ylabel('Score')
    ax_thr.set_title('Threshold vs Performance Metrics')
    ax_thr.legend()
    ax_thr.grid(True, alpha=0.3)
    st.pyplot(fig_thr)
    
    st.divider()
    
    # Confusion Matrix ve Metrikler
    st.subheader("📊 Model Performans Metrikleri")
    
    y_pred = (y_proba >= best_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        ax_cm.set_xlabel('Tahmin')
        ax_cm.set_ylabel('Gerçek')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)
        
    with col_m2:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)
    
    # Classification Report
    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='RdYlGn').format('{:.2f}'))
    
    st.divider()
    
    # Feature Importance
    st.subheader("🔥 Feature Importance (Top 20)")
    imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
    sns.barplot(x=imp.values, y=imp.index, palette='viridis', ax=ax_imp)
    ax_imp.set_xlabel('Importance')
    ax_imp.set_ylabel('Feature')
    ax_imp.set_title('Top 20 Feature Importance')
    st.pyplot(fig_imp)

# =============================================================================
# TAB 4: MODEL KARŞILAŞTIRMA
# =============================================================================
with tab_comp:
    st.header("🔄 Çoklu Model Karşılaştırması")
    st.markdown("Farklı makine öğrenmesi algoritmalarının performansını karşılaştırın.")
    
    if st.button("🚀 Tüm Modelleri Eğit ve Karşılaştır"):
        with st.spinner("Modeller eğitiliyor... Bu biraz zaman alabilir."):
            
            # Veri hazırlığı (önceki tab'dan)
            leakage_cols = [c for c in df_encoded.columns if 'SUB_FREQ_NEW' in c or 'PROMO_NO_SUB_NEW' in c or 'SHIP_SUB_NEW' in c]
            X_temp = df_encoded.drop(columns=['SUBSCRIPTION_STATUS'] + leakage_cols)
            y = df_encoded['SUBSCRIPTION_STATUS']
            
            rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_sel.fit(X_temp, y)
            importances = pd.Series(rf_sel.feature_importances_, index=X_temp.columns)
            keep_cols = importances[importances >= 0.01].index.tolist()
            
            X = X_temp[keep_cols]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Modeller
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight='balanced'),
                'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
                'LightGBM': LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
            }
            
            results = []
            model_predictions = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (name, model) in enumerate(models.items()):
                status_text.text(f"Eğitiliyor: {name}...")
                
                # Eğitim
                model.fit(X_train_s, y_train)
                
                # Tahmin
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_s)[:, 1]
                else:
                    y_proba = model.decision_function(X_test_s)
                
                # Optimal threshold bulma (Recall >= 80%)
                thresholds = np.linspace(0.05, 0.95, 19)
                best_thr = 0.5
                best_prec = 0
                
                for thr in thresholds:
                    y_pred_temp = (y_proba >= thr).astype(int)
                    rec = recall_score(y_test, y_pred_temp, zero_division=0)
                    prec = precision_score(y_test, y_pred_temp, zero_division=0)
                    if rec >= 0.80 and prec > best_prec:
                        best_thr = thr
                        best_prec = prec
                
                y_pred = (y_proba >= best_thr).astype(int)
                
                # Metrikler
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_proba)
                
                results.append({
                    'Model': name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1-Score': f1,
                    'ROC-AUC': roc_auc,
                    'Threshold': best_thr
                })
                
                model_predictions[name] = {
                    'y_proba': y_proba,
                    'y_pred': y_pred,
                    'threshold': best_thr
                }
                
                progress_bar.progress((idx + 1) / len(models))
            
            status_text.text("✅ Tüm modeller eğitildi!")
            st.session_state['model_results'] = results
            st.session_state['model_predictions'] = model_predictions
        
        # Sonuçları göster
        st.success("Karşılaştırma tamamlandı!")
        
        results_df = pd.DataFrame(st.session_state['model_results'])
        
        st.subheader("📊 Model Performans Tablosu")
        st.dataframe(results_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']).format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}',
            'Threshold': '{:.2f}'
        }))
        
        # En iyi model
        best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        st.success(f"🏆 **En İyi Model (F1-Score):** {best_model_name}")
        
        st.divider()
        
        # Karşılaştırma grafikleri
        st.subheader("📈 Model Karşılaştırma Grafikleri")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("**Metrik Karşılaştırması**")
            fig_comp1, ax_comp1 = plt.subplots(figsize=(10, 6))
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            x = np.arange(len(results_df['Model']))
            width = 0.15
            
            for i, metric in enumerate(metrics_to_plot):
                ax_comp1.bar(x + i*width, results_df[metric], width, label=metric)
            
            ax_comp1.set_xlabel('Model')
            ax_comp1.set_ylabel('Score')
            ax_comp1.set_title('Model Performance Comparison')
            ax_comp1.set_xticks(x + width * 2)
            ax_comp1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
            ax_comp1.legend()
            ax_comp1.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig_comp1)
        
        with col_c2:
            st.markdown("**ROC Curves Karşılaştırması**")
            fig_roc_comp, ax_roc_comp = plt.subplots(figsize=(10, 6))
            
            for name, preds in st.session_state['model_predictions'].items():
                fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
                roc_auc = auc(fpr, tpr)
                ax_roc_comp.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            
            ax_roc_comp.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            ax_roc_comp.set_xlabel('False Positive Rate')
            ax_roc_comp.set_ylabel('True Positive Rate')
            ax_roc_comp.set_title('ROC Curves Comparison')
            ax_roc_comp.legend(loc='lower right')
            ax_roc_comp.grid(True, alpha=0.3)
            st.pyplot(fig_roc_comp)
        
        # Confusion Matrices
        st.subheader("🎯 Confusion Matrices")
        cols_cm = st.columns(len(models))
        
        for idx, (name, preds) in enumerate(st.session_state['model_predictions'].items()):
            with cols_cm[idx]:
                st.markdown(f"**{name}**")
                cm = confusion_matrix(y_test, preds['y_pred'])
                fig_cm_small, ax_cm_small = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm_small,
                           xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], cbar=False)
                ax_cm_small.set_xlabel('Pred')
                ax_cm_small.set_ylabel('True')
                st.pyplot(fig_cm_small)
    
    elif 'model_results' in st.session_state and st.session_state['model_results']:
        # Önceki sonuçları göster
        results_df = pd.DataFrame(st.session_state['model_results'])
        st.subheader("📊 Önceki Model Karşılaştırma Sonuçları")
        st.dataframe(results_df.style.background_gradient(cmap='RdYlGn').format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}',
            'Threshold': '{:.2f}'
        }))
    else:
        st.info("👆 Modelleri eğitmek için yukarıdaki butona tıklayın.")

# =============================================================================
# TAB 5: SİMÜLATÖR
# =============================================================================
with tab_sim:
    st.header(f"🧪 Canlı Tahmin Simülatörü (Threshold: {st.session_state['best_threshold']:.2f})")
    
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
        
        item = df_raw['ITEM_PURCHASED'].mode()[0]
        color = df_raw['COLOR'].mode()[0]
        size = df_raw['SIZE'].mode()[0]
        season = df_raw['SEASON'].mode()[0]
        pay = df_raw['PAYMENT_METHOD'].mode()[0]
        ship = df_raw['SHIPPING_TYPE'].mode()[0]
        
        btn = st.form_submit_button("🔮 Tahmin Et")
        
    if btn:
        input_row = pd.DataFrame({
            'CUSTOMER_ID': [999999],
            'AGE': [age], 'GENDER': [gender], 'ITEM_PURCHASED': [item],
            'CATEGORY': [cat], 'PURCHASE_AMOUNT_(USD)': [spend],
            'LOCATION': [loc], 'SIZE': [size], 'COLOR': [color],
            'SEASON': [season], 'REVIEW_RATING': [rating],
            'SHIPPING_TYPE': [ship], 'DISCOUNT_APPLIED': ['No'],
            'PROMO_CODE_USED': [promo], 'PREVIOUS_PURCHASES': [prev],
            'PAYMENT_METHOD': [pay], 'FREQUENCY_OF_PURCHASES': [freq],
            'SUBSCRIPTION_STATUS': ['No']
        })
        
        full_df = pd.concat([df_raw, input_row], axis=0, ignore_index=True)
        _, full_encoded = process_data_pipeline(full_df)
        
        leakage_cols = [c for c in full_encoded.columns if 'SUB_FREQ_NEW' in c or 'PROMO_NO_SUB_NEW' in c or 'SHIP_SUB_NEW' in c]
        user_row = full_encoded.iloc[[-1]].drop(columns=['SUBSCRIPTION_STATUS'] + leakage_cols)
        user_row = user_row.reindex(columns=X.columns, fill_value=0)
        
        # Abonelik tahmini
        user_s = scaler_model.transform(user_row)
        prob = final_model.predict_proba(user_s)[0][1]
        
        # Cluster tahmini
        segmentation_cols = [
            'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 
            'FREQUENCY_VALUE_NEW', 'PROMO_CODE_USED', 
            'SPEND_PER_PURCHASE_NEW', 'LOYALTY_SCORE_NEW', 'CLIMATE_LOYALTY_NEW'
        ]
        
        # Kullanıcının segmentasyon için gerekli özelliklerini al
        user_seg_data = full_encoded.iloc[[-1]][[c for c in segmentation_cols if c in full_encoded.columns]]
        user_seg_data.fillna(0, inplace=True)
        
        # Scale et (segmentasyon için kullanılan scaler ile)
        user_seg_scaled = scaler_seg.transform(user_seg_data)
        
        # Cluster tahmini yap
        predicted_cluster = kmeans.predict(user_seg_scaled)[0]
        
        # Segment ismini al
        segment_name = profile.loc[predicted_cluster, 'Segment İsmi']
        
        thr = st.session_state['best_threshold']
        
        st.divider()
        
        # 3 kolonlu layout: Abonelik, Cluster, Profil
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
            st.info(f"### Cluster {predicted_cluster}")
            st.success(f"**{segment_name}**")
            
            # Cluster istatistikleri
            if predicted_cluster in segment_sub_rate.index:
                cluster_info = segment_sub_rate.loc[predicted_cluster]
                st.metric("Segment Abonelik Oranı", f"{cluster_info['Abonelik Oranı']:.1f}%")
                st.metric("Segment Müşteri Sayısı", f"{cluster_info['Müşteri Sayısı']:.0f}")
        
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
        
        st.divider()
        
        # Segment karşılaştırması
        st.subheader("📊 Segment Profili ve Karşılaştırma")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown(f"**Cluster {predicted_cluster} ({segment_name}) Profili:**")
            if predicted_cluster in segment_sub_rate.index:
                cluster_profile = segment_sub_rate.loc[predicted_cluster]
                st.write(f"• Ortalama Harcama: ${cluster_profile['Ort. Harcama']:.2f}")
                st.write(f"• Ortalama Alışveriş: {cluster_profile['Ort. Alışveriş Sayısı']:.1f}")
                st.write(f"• Ortalama Rating: {cluster_profile['Ort. Rating']:.2f}")
                st.write(f"• Abonelik Oranı: {cluster_profile['Abonelik Oranı']:.1f}%")
        
        with comp_col2:
            st.markdown("**🎯 Öneriler:**")
            
            if predicted_cluster in segment_sub_rate.index:
                cluster_info = segment_sub_rate.loc[predicted_cluster]
                
                if cluster_info['Abonelik Oranı'] < 40:
                    st.warning("⚠️ Bu segment düşük abonelik oranına sahip")
                    st.write("💡 Agresif abonelik kampanyası uygulayın")
                elif cluster_info['Abonelik Oranı'] < 60:
                    st.info("ℹ️ Orta düzey abonelik potansiyeli")
                    st.write("💡 Kişiselleştirilmiş teklifler sunun")
                else:
                    st.success("✅ Yüksek abonelik potansiyeli")
                    st.write("💡 Sadakat programı ile uzun vadeli bağ kurun")
                
                if prob >= thr and cluster_info['Abonelik Oranı'] >= 50:
                    st.success("🎉 Hem model hem de segment abone olma olasılığı yüksek!")
                elif prob < thr and cluster_info['Abonelik Oranı'] < 40:
                    st.error("⚠️ Hem model hem de segment düşük abonelik gösteriyor - Dikkatli yaklaşın")
