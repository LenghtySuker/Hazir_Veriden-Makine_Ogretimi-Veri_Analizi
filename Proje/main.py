import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, davies_bouldin_score, r2_score
import logging
import warnings
import lightgbm as lgb
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from datetime import timedelta, date
import requests
import feedparser
# Polars ve GPU desteği
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# Logging konfigürasyonu
logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== VERİ YÜKLEME (ORİJİNAL PERFORMANSTA) ====================
def load_data_optimized(dosya_yolu, nrows=None):
    print("📊 Veri yükleniyor...")
    
    if POLARS_AVAILABLE:
        try:
            print("⚡ Polars kullanılıyor")
            df_polars = pl.read_ndjson(dosya_yolu, n_rows=nrows)
            df_polars = df_polars.rename({
                'browser': 'Browser Name and Version', 'os': 'OS Name and Version',
                'loginTime': 'Login Timestamp', 'clientName': 'Client Name'
            })
            df = df_polars.to_pandas()
        except:
            print("⚠️ Polars hatası, Pandas kullanılıyor...")
            df = pd.read_json(dosya_yolu, lines=True, nrows=nrows)
    else:
        df = pd.read_json(dosya_yolu, lines=True, nrows=nrows)
    
    # Ortak preprocessing
    df = df.rename(columns={
        'browser': 'Browser Name and Version', 'os': 'OS Name and Version',
        'loginTime': 'Login Timestamp', 'clientName': 'Client Name'
    })
    
    df["Login Successful"] = 1
    df = df.fillna("Bilinmiyor")
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')
    df['saat'] = df['Login Timestamp'].dt.hour
    df['gun'] = df['Login Timestamp'].dt.dayofweek
    
    # Encoding
    encoders = {}
    for col in ["OS Name and Version", "Browser Name and Version", "Client Name"]:
        encoders[col] = LabelEncoder()
        df[col+'_enc'] = encoders[col].fit_transform(df[col].astype(str))
    
    print(f"✓ {len(df):,} kayıt yüklendi | {df['Login Timestamp'].min()} → {df['Login Timestamp'].max()}")
    return df
# ==================== 1. OS/BROWSER YOĞUNLUK ANALİZİ (ORİJİNAL) ====================
def os_browser_yogunluk_analizi(df):
    print("\n📊 OS/BROWSER BAZLI LOGİN YOĞUNLUK ANALİZİ")
    
    for col, icon in [("OS Name and Version", "💻"), ("Browser Name and Version", "🌐")]:
        stats = df.groupby(col).agg({'email': 'nunique', 'Login Successful': 'sum'}).reset_index()
        stats.columns = [col, 'Kullanici_Sayisi', 'Login_Sayisi']
        stats['Ort_Login'] = (stats['Login_Sayisi'] / stats['Kullanici_Sayisi']).round(2)
        
        print(f"\n{icon} {col.upper()} İSTATİSTİK (Top 10)")
        print(stats.sort_values('Login_Sayisi', ascending=False).head(10).to_string(index=False))
# ==================== 2. CLIENT BAZLI ANALİZ (ORİJİNAL) ====================
def client_os_browser_analizi(df):
    print("\n📱 CLIENT BAZLI LOGİN DAĞILIMI")
    
    client_stats = df.groupby("Client Name").agg({
        'email': 'nunique', 'Login Successful': 'sum'
    }).rename(columns={'email': 'Kullanici', 'Login Successful': 'Login'}).sort_values('Login', ascending=False)
    print(client_stats.to_string())
    
    df_temp = df.copy()
    df_temp['Browser_Base'] = df_temp['Browser Name and Version'].str.split().str[0]
    
    print("\n🌐 TOP 5 CLIENT İÇİN BROWSER DAĞILIMI")
    for client in client_stats.head(5).index:
        df_client = df_temp[df_temp['Client Name'] == client]
        browser_dist = df_client['Browser_Base'].value_counts().head(3)
        
        print(f"\n🔹 {client} ({len(df_client):,} login):")
        for browser, count in browser_dist.items():
            print(f"   • {browser}: {count:,} (%{count/len(df_client)*100:.1f})")
# ==================== 3. SAAT/GÜN TAHMİNİ (ORİJİNAL PERFORMANSTA) ====================
def create_optimized_models():
    if GPU_AVAILABLE:
        return {
            "LGBM": lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=64,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20, min_split_gain=0.001,
                device='gpu', gpu_platform_id=0, gpu_device_id=0, verbosity=-1, random_state=42
            ),
            "XGB": xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7, min_child_weight=5, gamma=0.1,
                subsample=0.8, colsample_bytree=0.8, tree_method='gpu_hist', predictor='gpu_predictor',
                gpu_id=0, verbosity=0, random_state=42
            )
        }
    else:
        return {
            "LGBM": lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=64,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20, min_split_gain=0.001,
                verbosity=-1, n_jobs=-1, random_state=42
            ),
            "XGB": xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7, min_child_weight=5, gamma=0.1,
                subsample=0.8, colsample_bytree=0.8, verbosity=0, n_jobs=-1, random_state=42
            )
        }
def create_features(data):
    """Feature engineering - orijinal mantık korundu"""
    data = data.copy()
    data['hafta_sonu'] = (data['gun'] >= 5).astype(int)
    data['mevsim'] = data['ay'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    data['saat_sin'] = np.sin(2 * np.pi * data['saat']/24)
    data['saat_cos'] = np.cos(2 * np.pi * data['saat']/24)
    data['gun_sin'] = np.sin(2 * np.pi * data['gun']/7)
    data['gun_cos'] = np.cos(2 * np.pi * data['gun']/7)
    data['ay_sin'] = np.sin(2 * np.pi * data['ay']/12)
    data['ay_cos'] = np.cos(2 * np.pi * data['ay']/12)
    data['is_saati'] = ((data['saat'] >= 9) & (data['saat'] <= 17) & (data['hafta_sonu']==0)).astype(int)
    return data
def saat_gun_model(df):
    """Orijinal saat/gün tahmini - performans korundu"""
    df = df.copy()
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')
    df = df.dropna(subset=['Login Timestamp'])

    # Saatlik veri hazırlığı
    full_range = pd.date_range(df['Login Timestamp'].min().floor('H'), df['Login Timestamp'].max().ceil('H'), freq='H')
    hourly_counts = df.groupby(pd.Grouper(key='Login Timestamp', freq='H')).size().reindex(full_range, fill_value=0)
    hourly_counts = hourly_counts.reset_index().rename(columns={'index':'Login Timestamp', 0:'gercek_login'})
    
    # Temel feature'lar
    hourly_counts['gun'] = hourly_counts['Login Timestamp'].dt.dayofweek
    hourly_counts['saat'] = hourly_counts['Login Timestamp'].dt.hour
    hourly_counts['ay'] = hourly_counts['Login Timestamp'].dt.month
    hourly_counts['yil'] = hourly_counts['Login Timestamp'].dt.year
    hourly_counts['hafta'] = hourly_counts['Login Timestamp'].dt.isocalendar().week

    # Train/test split
    split_date = hourly_counts['Login Timestamp'].quantile(0.85)
    train_data = hourly_counts[hourly_counts['Login Timestamp'] < split_date].copy()
    test_data = hourly_counts[hourly_counts['Login Timestamp'] >= split_date].copy()

    # Feature engineering
    train_data = create_features(train_data)
    test_data = create_features(test_data)

    # Lag features
    lag_list = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lag_list:
        train_data[f'lag_{lag}'] = train_data['gercek_login'].shift(lag).fillna(0)
        test_data[f'lag_{lag}'] = pd.concat([train_data['gercek_login'].iloc[-lag:], test_data['gercek_login']]).shift(lag).iloc[lag:].fillna(0).values

    # Rolling features
    rolling_windows = [6, 12, 24, 168]
    for w in rolling_windows:
        train_data[f'rolling_mean_{w}'] = train_data['gercek_login'].rolling(window=w, min_periods=1).mean()
        test_data[f'rolling_mean_{w}'] = pd.concat([train_data['gercek_login'].iloc[-w:], test_data['gercek_login']]).rolling(window=w, min_periods=1).mean().iloc[w:].values

    # Difference features
    for data in [train_data, test_data]:
        data['diff_1'] = data['gercek_login'].diff(1).fillna(0)
        data['diff_24'] = data['gercek_login'].diff(24).fillna(0)

    # Haftalık istatistikler
    global_mean = train_data['gercek_login'].mean()
    weekly_mean_dict = train_data.groupby('hafta')['gercek_login'].mean().to_dict()
    weekly_std_dict = train_data.groupby('hafta')['gercek_login'].std().fillna(0).to_dict()
    weekly_max_dict = train_data.groupby('hafta')['gercek_login'].max().to_dict()

    for data in [train_data, test_data]:
        data['weekly_mean'] = data['hafta'].map(weekly_mean_dict).fillna(global_mean)
        data['weekly_std'] = data['hafta'].map(weekly_std_dict).fillna(0)
        data['weekly_max'] = data['hafta'].map(weekly_max_dict).fillna(global_mean)

    # Gün/saat istatistikleri
    for data in [train_data, test_data]:
        data['gun_saat_key'] = data['gun'].astype(str) + '_' + data['saat'].astype(str)

    gun_saat_mean_dict = train_data.groupby('gun_saat_key')['gercek_login'].mean().to_dict()
    gun_saat_std_dict = train_data.groupby('gun_saat_key')['gercek_login'].std().fillna(0).to_dict()
    gun_saat_max_dict = train_data.groupby('gun_saat_key')['gercek_login'].max().to_dict()

    for data in [train_data, test_data]:
        data['gun_saat_mean'] = data['gun_saat_key'].map(gun_saat_mean_dict).fillna(global_mean)
        data['gun_saat_std'] = data['gun_saat_key'].map(gun_saat_std_dict).fillna(0)
        data['gun_saat_max'] = data['gun_saat_key'].map(gun_saat_max_dict).fillna(global_mean)

    # Feature columns
    feature_cols = [
        'gun', 'saat', 'ay', 'yil', 'hafta', 'hafta_sonu', 'mevsim', 'is_saati',
        'saat_sin', 'saat_cos', 'gun_sin', 'gun_cos', 'ay_sin', 'ay_cos',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24', 'lag_48', 'lag_168',
        'rolling_mean_6', 'rolling_mean_12', 'rolling_mean_24', 'rolling_mean_168',
        'diff_1', 'diff_24', 'weekly_mean', 'weekly_std', 'weekly_max',
        'gun_saat_mean', 'gun_saat_std', 'gun_saat_max'
    ]

    X_train, y_train = train_data[feature_cols], train_data['gercek_login']
    X_test, y_test = test_data[feature_cols], test_data['gercek_login']

    # Model eğitimi
    models = create_optimized_models()
    predictions = {}
    
    print("⏳ Model eğitimi başlıyor...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0, None)
        predictions[name] = np.round(preds).astype(int)

    # Metrik hesaplama
    def safe_mape(y_true, y_pred, epsilon=1e-10):
        mask = y_true > epsilon
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf

    model_metrics = {}
    for name, preds in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = safe_mape(y_test.values, preds)
        model_metrics[name] = {'RMSE': rmse, 'R2': r2, 'MAPE': mape}

    # Sonuçları göster
    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['MAPE'])
    best_model, second_best_model = sorted_models[0][0], sorted_models[1][0]

    results_df = test_data[['Login Timestamp', 'gun', 'saat', 'gercek_login']].copy()
    gun_isimleri = {0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe', 4: 'Cuma', 5: 'Cumartesi', 6: 'Pazar'}
    results_df['gun_ismi'] = results_df['gun'].map(gun_isimleri)

    for name in [best_model, second_best_model]:
        results_df[f'{name.lower()}_tahmin'] = predictions[name]
        results_df[f'{name.lower()}_hata%'] = np.where(
            results_df['gercek_login'] > 0,
            (np.abs(results_df['gercek_login'] - predictions[name]) / results_df['gercek_login'] * 100).round(2),
            0
        )

    top_10 = results_df.nlargest(20, 'gercek_login').drop_duplicates(subset=['Login Timestamp'])
    display_cols = ['Login Timestamp', 'gun_ismi', 'saat', 'gercek_login',
                   f'{best_model.lower()}_tahmin', f'{best_model.lower()}_hata%',
                   f'{second_best_model.lower()}_tahmin', f'{second_best_model.lower()}_hata%']

    print(f"\n🔝 EN YOĞUN 10 SAAT - EN İYİ 2 MODEL ({best_model} & {second_best_model})")
    print(top_10[display_cols].to_string(index=False))

    return results_df, model_metrics
# ==================== 4. HAFTALIK TAHMİN ====================
def haftalik_login_tahmini(df, min_train=None):
    df = df.dropna(subset=['Login Timestamp']).copy()
    haftalik = df.set_index('Login Timestamp').resample('W-MON')['Login Successful'].sum().reset_index()
    haftalik.rename(columns={'Login Timestamp':'ds','Login Successful':'y'}, inplace=True)
    haftalik['gercek_login'] = haftalik['y'].astype(int)

    if min_train is None:
        min_train = 80

    if len(haftalik) < (min_train + 10):
        print(f"❌ Yetersiz veri: en az {min_train + 10} hafta gerekli")
        return None

    # -----------------------------
    # Extreme weeks detection
    # -----------------------------
    def detect_extreme_weeks(data, percentile=8):
        low_threshold = data['y'].quantile(percentile/100)
        return set(data[data['y'] < low_threshold]['ds'].dt.strftime('%Y-%m-%d').tolist())

    # -----------------------------
    # Feature creation
    # -----------------------------
    def create_features(train_data, base_date):
        row = {}
        current_date = train_data.iloc[-1]['ds']

        # --- Time features ---
        row['ay'] = current_date.month
        row['gun'] = current_date.day
        row['hafta'] = current_date.isocalendar().week
        row['yilin_gunu'] = current_date.timetuple().tm_yday
        row['ceyrek'] = (current_date.month-1)//3 +1

        # --- Trig features ---
        row['week_sin'] = np.sin(2*np.pi*row['hafta']/52)
        row['week_cos'] = np.cos(2*np.pi*row['hafta']/52)
        row['month_sin'] = np.sin(2*np.pi*row['ay']/12)
        row['month_cos'] = np.cos(2*np.pi*row['ay']/12)
        row['quarter_sin'] = np.sin(2*np.pi*row['ceyrek']/4)
        row['quarter_cos'] = np.cos(2*np.pi*row['ceyrek']/4)

        # --- Dynamic cycle ---
        weeks_since_start = int((current_date - base_date).days / 7)
        row['cycle_position'] = weeks_since_start
        cycle_window = min(weeks_since_start, 66)
        row['cycle_multiplier'] = train_data['y'].iloc[-cycle_window:].mean()/(train_data['y'].mean()+1e-6) if cycle_window>=2 else 1.0
        row['cycle_sin'] = np.sin(2*np.pi*row['cycle_position']/66)
        row['cycle_cos'] = np.cos(2*np.pi*row['cycle_position']/66)

        # --- Season flags ---
        rolling_52 = train_data['y'].rolling(min(52,len(train_data))).mean()
        row['is_low_season'] = int(rolling_52.iloc[-1]<=rolling_52.quantile(0.25))
        row['is_high_season'] = int(rolling_52.iloc[-1]>=rolling_52.quantile(0.75))
        row['is_yaz'] = int(row['ay'] in [7,8] or (row['ay']==6 and current_date.day>=25))
        row['is_kis'] = int(row['ay'] in [12,1,2])
        row['is_ilkbahar'] = int(row['ay'] in [3,4,5])

        # --- Lag features ---
        for lag in [1,2,3,4,8,12,26,52,66]:
            row[f'lag_{lag}'] = train_data['y'].iloc[-lag] if len(train_data)>=lag else train_data['y'].mean()

        # --- Lag ratio ---
        row['lag1_to_lag66_ratio'] = train_data['y'].iloc[-1]/(train_data['y'].iloc[-66]+1e-6) if len(train_data)>=66 else 1.0

        # --- Rolling stats ---
        for window in [4,8,12,26]:
            w = min(window,len(train_data))
            w_data = train_data['y'].iloc[-w:]
            row[f'roll{window}_mean'] = w_data.mean()
            row[f'roll{window}_std'] = w_data.std()
            row[f'roll{window}_cv'] = w_data.std()/(w_data.mean()+1e-6)

        # --- EMA ---
        ema_window = min(8,len(train_data))
        weights = np.exp(np.linspace(-1,0,ema_window))
        weights/=weights.sum()
        row['ema_8'] = np.sum(train_data['y'].iloc[-ema_window:].values*weights)

        # --- Trend & momentum ---
        trend_window = min(12,len(train_data))
        if trend_window>=2:
            y_vals = train_data['y'].iloc[-trend_window:].values
            row['trend_12'] = np.polyfit(range(len(y_vals)),y_vals,1)[0]
            row['momentum_12'] = y_vals[-1]-y_vals[0]
        else:
            row['trend_12'] = 0
            row['momentum_12'] = 0

        # --- Cycle stats ---
        if len(train_data)>=66:
            same_cycle_vals = train_data['y'].iloc[-(66*min(4,len(train_data)//66)):].values[::66]
            row['cycle_avg'] = same_cycle_vals.mean()
            row['cycle_std'] = same_cycle_vals.std()
            row['cycle_cv'] = row['cycle_std']/(row['cycle_avg']+1e-6)
        else:
            row['cycle_avg'] = row['roll8_mean']
            row['cycle_std'] = row['roll8_std']
            row['cycle_cv'] = row['roll8_cv']

        # --- Volatility & value-to-cycle ---
        vol_data = train_data['y'].iloc[-min(8,len(train_data)):]
        row['volatility_8'] = vol_data.std()/(vol_data.mean()+1e-6)
        row['value_to_cycle_ratio'] = train_data['y'].iloc[-1]/(row['cycle_avg']+1e-6)

        return row

    # -----------------------------
    # Walk-forward loop
    # -----------------------------
    predictions = []
    for i in range(len(haftalik)):
        if i<min_train:
            predictions.append({
                'ds': haftalik.iloc[i]['ds'],
                'gercek_login': haftalik.iloc[i]['gercek_login'],
                'xgb_tahmin': None,
                'lgbm_tahmin': None,
                'is_yaz': None
            })
            continue

        train = haftalik.iloc[:i].copy()
        test_date = haftalik.iloc[i]['ds']
        gercek_deger = haftalik.iloc[i]['gercek_login']
        base_date = train.iloc[0]['ds']
        extreme_weeks = detect_extreme_weeks(train)

        # --- Train feature matrix ---
        X_train_list, y_train_list = [],[]
        for idx in range(66,len(train)):
            subset = train.iloc[:idx]
            X_train_list.append(create_features(subset,base_date))
            y_train_list.append(train.iloc[idx]['y'])
        if not X_train_list:
            predictions.append({'ds':test_date,'gercek_login':gercek_deger,'xgb_tahmin':None,'lgbm_tahmin':None,'is_yaz':None})
            continue
        X_train = pd.DataFrame(X_train_list).fillna(0)
        y_train = np.array(y_train_list)

        # --- Test features ---
        X_test = pd.DataFrame([create_features(train,base_date)]).fillna(0)
        X_test = X_test[X_train.columns]

        # --- XGBoost ---
        try:
            xgb_model = XGBRegressor(
                n_estimators=500, max_depth=12, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, gamma=0,
                reg_alpha=1.5, reg_lambda=2.0,
                random_state=42, n_jobs=-1
            )
            xgb_model.fit(X_train,y_train)
            xgb_tahmin = xgb_model.predict(X_test)[0]
        except:
            xgb_tahmin = train['y'].tail(8).mean()

        # --- LightGBM ---
        try:
            lgbm_model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=12, learning_rate=0.05,
                num_leaves=31, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=1.5, reg_lambda=2.0, min_child_samples=20,
                n_jobs=-1, random_state=42, verbosity=-1, force_col_wise=True
            )
            lgbm_model.fit(X_train,y_train)
            lgbm_tahmin = lgbm_model.predict(X_test)[0]
        except:
            lgbm_tahmin = train['y'].tail(8).mean()

        # ===== POST-PROCESSING =====
        features_today = create_features(train,base_date)
        test_date_str = test_date.strftime('%Y-%m-%d')

        # --- Extreme weeks adjustment ---
        if test_date_str in extreme_weeks:
            hist_values = [train['y'].iloc[-lag] for lag in [66,132,198] if len(train)>=lag]
            if hist_values:
                hist_avg = np.mean(hist_values)
                xgb_tahmin = 0.25*xgb_tahmin + 0.75*hist_avg
                lgbm_tahmin = 0.25*lgbm_tahmin + 0.75*hist_avg
        # --- Low season adjustment ---
        elif features_today['is_low_season']==1:
            hist_values = [train['y'].iloc[-lag] for lag in [66,132] if len(train)>=lag]
            if hist_values:
                pos_avg = np.mean(hist_values)
                xgb_tahmin = 0.5*xgb_tahmin + 0.5*pos_avg
                lgbm_tahmin = 0.5*lgbm_tahmin + 0.5*pos_avg

        # --- Min/max clipping ---
        min_val = train['y'].quantile(0.02)
        max_val = train['y'].quantile(0.98)*1.15
        xgb_tahmin = np.clip(xgb_tahmin,min_val,max_val)
        lgbm_tahmin = np.clip(lgbm_tahmin,min_val,max_val)

        # --- Z-score adjustment ---
        recent_mean = train['y'].tail(12).mean()
        recent_std = train['y'].tail(12).std()+1e-6
        for model_name,value in [('xgb',xgb_tahmin),('lgbm',lgbm_tahmin)]:
            z_score = abs(value-recent_mean)/recent_std
            if z_score>3.0:
                if model_name=='xgb':
                    xgb_tahmin = 0.65*value+0.35*recent_mean
                else:
                    lgbm_tahmin = 0.65*value+0.35*recent_mean

        predictions.append({
            'ds':test_date,
            'gercek_login':gercek_deger,
            'xgb_tahmin':int(round(xgb_tahmin)),
            'lgbm_tahmin':int(round(lgbm_tahmin)),
            'is_yaz':int(features_today['is_yaz'])
        })

    # -----------------------------
    # Results & metrics
    # -----------------------------
    results = pd.DataFrame(predictions)
    results['xgb_hata%'] = (abs(results['xgb_tahmin']-results['gercek_login'])/results['gercek_login']*100).round(2)
    results['lgbm_hata%'] = (abs(results['lgbm_tahmin']-results['gercek_login'])/results['gercek_login']*100).round(2)
    valid = results.dropna(subset=['xgb_tahmin']).copy()

    print("\n✅ VERİ SIZINTISI DÜZELTİLMİŞ - XGBoost + LightGBM")
    display_cols = ['ds','gercek_login','xgb_tahmin','xgb_hata%','lgbm_tahmin','lgbm_hata%','is_yaz']
    print(valid[display_cols].to_string(index=False))

    print("\n📊 PERFORMANS METRİKLERİ")
    for model in ['xgb','lgbm']:
        rmse = np.sqrt(np.mean((valid['gercek_login']-valid[f'{model}_tahmin'])**2))
        mae = np.mean(abs(valid['gercek_login']-valid[f'{model}_tahmin']))
        mape = valid[f'{model}_hata%'].mean()
        r2 = 1-(np.sum((valid['gercek_login']-valid[f'{model}_tahmin'])**2)/
                 np.sum((valid['gercek_login']-valid['gercek_login'].mean())**2))
        print(f"{model.upper():4s} → RMSE: {rmse:>8.0f} | MAE: {mae:>8.0f} | MAPE: {mape:6.2f}% | R²: {r2:6.3f}")

    return valid
# ==================== 5. GERÇEK ZAMANLI GELECEK HAFTA TAHMİNİ ====================
class SchoolCalendar:
    @staticmethod
    def get_period_info(date):
        """Tarihe göre okul dönemi ve beklenen aktivite seviyesi"""
        month, day = date.month, date.day
        
        # YAZ TATİLİ (15 Haziran - Temmuz)
        if month == 6 and day >= 15:
            return {
                'period': 'haziran_sonu',
                'factor': 0.80,
                'min': 18000,
                'max': 32000,
                'baseline': 24000
            }
        
        elif month == 7:
            if day <= 15:
                # Temmuz başı daha yüksek
                return {
                    'period': 'temmuz_basi',
                    'factor': 1.10,
                    'min': 28000,
                    'max': 42000,
                    'baseline': 34000
                }
            else:
                # Temmuz ortası-sonu çok yüksek! (2023-2024: 45-48K)
                return {
                    'period': 'temmuz_yuksek',
                    'factor': 1.35,
                    'min': 36000,
                    'max': 52000,
                    'baseline': 43000
                }
        
        # AĞUSTOS OKUL HAZIRLIĞI (1-21: tatil, 22-31: hazırlık BAŞLADI)
        elif month == 8:
            if day <= 21:
                return {
                    'period': 'agustos_tatil',
                    'factor': 0.85,
                    'min': 22000,
                    'max': 36000,
                    'baseline': 29000
                }
            else:
                return {
                    'period': 'okul_hazirlik_baslangic',
                    'factor': 1.30,
                    'min': 35000,
                    'max': 48000,
                    'baseline': 40000
                }
        
        # EYLÜL BAŞI (1-14: HAZIRLIK DEVAM EDİYOR)
        elif month == 9 and day < 15:
            return {
                'period': 'eylul_hazirlik',
                'factor': 1.35,
                'min': 35000,
                'max': 50000,
                'baseline': 40000
            }
        
        # OKUL AÇILIŞI (15-30 Eylül) - ANİ YÜKSELIŞ
        elif month == 9 and 15 <= day <= 30:
            return {
                'period': 'okul_acilis',
                'factor': 1.45,
                'min': 32000,
                'max': 50000,
                'baseline': 40000
            }
        
        # EKİM PEAK
        elif month == 10:
            return {
                'period': 'ekim_peak',
                'factor': 1.35,
                'min': 32000,
                'max': 52000,
                'baseline': 40000
            }
        
        # KASIM AYRINTILI (gerçek veriye göre)
        elif month == 11:
            if day <= 10:
                return {
                    'period': 'kasim_basi',
                    'factor': 1.30,
                    'min': 32000,
                    'max': 45000,
                    'baseline': 37000
                }
            elif day <= 24:
                return {
                    'period': 'kasim_orta',
                    'factor': 0.90,
                    'min': 18000,
                    'max': 30000,
                    'baseline': 23000
                }
            else:
                return {
                    'period': 'kasim_sonu',
                    'factor': 1.20,
                    'min': 28000,
                    'max': 40000,
                    'baseline': 33000
                }
        
        # ARALIK-OCAK FİNAL DÖNEMİ
        elif month in [12, 1]:
            return {
                'period': 'final_donemi',
                'factor': 1.10,
                'min': 28000,
                'max': 38000,
                'baseline': 32000
            }
        
        # ŞUBAT (Yarıyıl tatili etkisi)
        elif month == 2:
            if day <= 15:
                return {
                    'period': 'subat_normal',
                    'factor': 1.05,
                    'min': 22000,
                    'max': 32000,
                    'baseline': 26000
                }
            else:
                return {
                    'period': 'subat_sonu',
                    'factor': 1.10,
                    'min': 24000,
                    'max': 34000,
                    'baseline': 28000
                }
        
        # MART (Bayram etkisi olabilir)
        elif month == 3:
            return {
                'period': 'mart',
                'factor': 1.05,
                'min': 22000,
                'max': 38000,
                'baseline': 30000
            }
        
        # NİSAN (Bahar dönemi yüksek aktivite)
        elif month == 4:
            return {
                'period': 'nisan_yuksek',
                'factor': 1.35,
                'min': 33000,
                'max': 50000,
                'baseline': 42000
            }
        
        # MAYIS-HAZİRAN ORTASI (Final hazırlık)
        elif month == 5 or (month == 6 and day < 15):
            return {
                'period': 'bahar_final',
                'factor': 1.15,
                'min': 28000,
                'max': 40000,
                'baseline': 33000
            }
        
        # NORMAL DÖNEM
        return {'period': 'normal', 'factor': 1.0, 'min': 25000, 'max': 38000, 'baseline': 30000}
    
    @staticmethod
    def get_historical_baseline(date, historical_data):
        """Aynı döneme ait geçmiş 3 yıl ortalaması"""
        target_week = date.isocalendar().week
        
        same_period = historical_data[
            historical_data['ds'].dt.isocalendar().week.between(target_week-2, target_week+2)
        ]
        
        if len(same_period) >= 3:
            sorted_data = same_period.sort_values('ds', ascending=False)
            
            if len(sorted_data) >= 12:
                recent_3y = sorted_data.head(12)
                weights = np.exp(np.linspace(-1, 0, len(recent_3y)))
                weights /= weights.sum()
                weighted_avg = np.average(recent_3y['y'].values, weights=weights)
                return {
                    'weighted_mean': weighted_avg,
                    'median': recent_3y['y'].median(),
                    'q25': recent_3y['y'].quantile(0.25),
                    'q75': recent_3y['y'].quantile(0.75),
                    'max': recent_3y['y'].max(),
                    'count': len(recent_3y)
                }
        
        return {
            'weighted_mean': historical_data['y'].mean(),
            'median': historical_data['y'].median(),
            'q25': historical_data['y'].quantile(0.25),
            'q75': historical_data['y'].quantile(0.75),
            'max': historical_data['y'].max(),
            'count': 0
        }
class AnomalyManager:
    @staticmethod
    def detect_regime_change(historical_data, window=26):
        """Rejim değişikliği tespit et (deprem, pandemi, vb.)"""
        if len(historical_data) < window * 2:
            return False, 'insufficient_data'
        
        recent = historical_data['y'].iloc[-window:].mean()
        previous = historical_data['y'].iloc[-window*2:-window].mean()
        
        change_pct = (recent - previous) / (previous + 1e-6)
        
        if change_pct < -0.30:
            return True, 'major_drop'
        elif change_pct > 0.30:
            return True, 'major_spike'
        
        return False, 'stable'
    
    @staticmethod
    def adjust_for_anomaly(prediction, anomaly_type, historical_baseline, period_info):
        """Anomali durumunda tahmini ayarla"""
        
        if anomaly_type == 'major_drop':
            safe_baseline = historical_baseline['q25']
            return 0.30 * prediction + 0.70 * safe_baseline
        
        elif anomaly_type == 'major_spike':
            high_baseline = historical_baseline['q75']
            return 0.50 * prediction + 0.50 * high_baseline
        
        return prediction
class GeographicImpactCalculator:
    """Olayların İstanbul'daki loginlere etkisini hesaplar"""
    
    DISTANCE_IMPACT = {
        'ISTANBUL': 1.0, 'KOCAELI': 0.8, 'SAKARYA': 0.7, 'BURSA': 0.6,
        'ANKARA': 0.4, 'IZMIR': 0.3, 'DIYARBAKIR': 0.1, 'default': 0.2
    }
    
    CITY_MAPPING = {
        'istanbul': 'ISTANBUL', 'i̇stanbul': 'ISTANBUL', 'ist': 'ISTANBUL',
        'ankara': 'ANKARA', 'izmir': 'IZMIR', 'bursa': 'BURSA',
        'kocaeli': 'KOCAELI', 'izmit': 'KOCAELI', 'sakarya': 'SAKARYA',
        'adapazari': 'SAKARYA', 'diyarbakir': 'DIYARBAKIR', 'diyarbakır': 'DIYARBAKIR'
    }
    
    @staticmethod
    def calculate_impact(location, event_type, severity):
        """Olayın İstanbul'daki loginlere etkisini hesapla"""
        location_impact = GeographicImpactCalculator._get_location_impact(location)
        
        base_impact = {
            'deprem': 0.8, 'yangin': 0.6, 'sel': 0.5, 'saglik_krizi': 0.4,
            'internet_kesintisi': 0.9, 'ekonomik_kriz': 0.3, 
            'sosyal_olay': 0.4, 'egitim_degisikligi': 0.2
        }.get(event_type, 0.5)
        
        final_impact = base_impact * location_impact * severity
        return min(0.95, final_impact)
    
    @staticmethod
    def _get_location_impact(location):
        """Lokasyona göre etki katsayısı"""
        if not location:
            return 0.3
        
        location_lower = location.lower().strip()
        
        for city_key, standard_city in GeographicImpactCalculator.CITY_MAPPING.items():
            if city_key in location_lower:
                return GeographicImpactCalculator.DISTANCE_IMPACT.get(standard_city, 0.2)
        
        for city in GeographicImpactCalculator.DISTANCE_IMPACT.keys():
            if city.lower() in location_lower:
                return GeographicImpactCalculator.DISTANCE_IMPACT[city]
        
        return GeographicImpactCalculator.DISTANCE_IMPACT['default']
class RealTimeWorldMonitor:
    """Tüm kategorilerde gerçek zamanlı izleme (ekonomi kaldırıldı, sessiz)"""
    
    @staticmethod
    def check_all_anomalies():
        """Tüm anomalileri gerçek zamanlı kontrol et"""
        anomalies = []
        anomalies.extend(RealTimeWorldMonitor._check_earthquakes())
        anomalies.extend(RealTimeWorldMonitor._check_weather_disasters())
        anomalies.extend(RealTimeWorldMonitor._check_health_crises())
        anomalies.extend(RealTimeWorldMonitor._check_tech_outages())
        anomalies.extend(RealTimeWorldMonitor._check_education_news())
        return anomalies
    
    @staticmethod
    def _check_earthquakes():
        """AFAD/USGS'den gerçek zamanlı deprem verisi"""
        earthquakes = []
        try:
            url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for feature in data.get('features', [])[:10]:
                    props = feature.get('properties', {})
                    geometry = feature.get('geometry', {})
                    magnitude = props.get('mag', 0)
                    location = props.get('place', 'Bilinmeyen')
                    if magnitude >= 4.5:
                        coords = geometry.get('coordinates', [])
                        if coords and len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            if 26.0 <= lon <= 45.0 and 36.0 <= lat <= 42.0:
                                earthquakes.append({
                                    'type': 'deprem',
                                    'category': 'afet_durumu',
                                    'severity': min(magnitude / 10.0, 0.9),
                                    'location': location,
                                    'magnitude': magnitude,
                                    'source': 'USGS',
                                    'description': f"{magnitude} büyüklüğünde deprem - {location}"
                                })
        except:
            pass
        return earthquakes
    
    @staticmethod
    def _check_weather_disasters():
        """Meteorolojik afet ve yangınlar - sessiz çalışır"""
        disasters = []
        try:
            url = "https://api.mgm.gov.tr/api/tahmin/il-ve-ilceler"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for city_data in data.get('data', [])[:10]:
                    city_name = city_data.get('il', '')
                    weather_status = city_data.get('durum', '')
                    critical_weather = ['fırtına', 'sağanak', 'yoğun kar', 'dolu']
                    if any(status in weather_status.lower() for status in critical_weather):
                        disasters.append({
                            'type': 'meteorolojik_afet',
                            'category': 'afet_durumu',
                            'severity': 0.6,
                            'location': city_name,
                            'source': 'MGM',
                            'description': f"Kritik hava durumu: {weather_status} - {city_name}"
                        })
        except:
            pass
        return disasters
    
    @staticmethod
    def _check_health_crises():
        """Sağlık krizleri ve salgınlar"""
        health_events = []
        try:
            who_feed = feedparser.parse('https://www.who.int/feeds/entity/csr/don/tr/rss.xml')
            for entry in who_feed.entries[:5]:
                if any(keyword in entry.title.lower() for keyword in ['salgın', 'pandemi', 'virüs', 'kriz']):
                    health_events.append({
                        'type': 'saglik_krizi',
                        'category': 'saglik_krizi',
                        'severity': 0.7,
                        'location': 'Türkiye',
                        'source': 'WHO',
                        'description': entry.title
                    })
        except:
            pass
        return health_events
    
    @staticmethod
    def _check_tech_outages():
        """İnternet ve teknoloji kesintileri"""
        outages = []
        try:
            services = ['turkcell', 'turktelekom', 'superonline', 'vodafone']
            for service in services:
                url = f"https://downdetector.com.tr/api/service/{service}/status"
                try:
                    response = requests.get(url, timeout=8)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('reports', 0) > 100:
                            outages.append({
                                'type': 'internet_kesintisi',
                                'category': 'teknoloji_kesintisi',
                                'severity': 0.8,
                                'location': 'Türkiye',
                                'source': 'Downdetector',
                                'description': f"{service} kesintisi - {data.get('reports', 0)} rapor"
                            })
                except:
                    continue
            cloud_services = {
                'aws': 'https://health.aws.amazon.com/health/status',
                'azure': 'https://status.azure.com/en-us/status',
                'google': 'https://www.google.com/appsstatus/dashboard/'
            }
            for service, status_url in cloud_services.items():
                try:
                    response = requests.get(status_url, timeout=5)
                    if response.status_code != 200:
                        outages.append({
                            'type': 'bulut_servis_kesintisi',
                            'category': 'teknoloji_kesintisi',
                            'severity': 0.9,
                            'location': 'Global',
                            'source': f'{service.upper()} Status',
                            'description': f'{service} erişim sorunu'
                        })
                except:
                    continue
        except:
            pass
        return outages
    
    @staticmethod
    def _check_education_news():
        """Eğitimdeki değişiklikler ve haberler"""
        education_events = []
        try:
            meb_feed = feedparser.parse('https://www.meb.gov.tr/rss')
            keywords = ['yeni sistem', 'müfredat', 'sınav', 'uzaktan eğitim', 'yüz yüze', 'değişiklik']
            for entry in meb_feed.entries[:10]:
                if any(keyword in entry.title.lower() for keyword in keywords):
                    severity = 0.2
                    if 'iptal' in entry.title.lower() or 'ertelendi' in entry.title.lower():
                        severity = 0.5
                    elif 'yeni' in entry.title.lower() or 'başlıyor' in entry.title.lower():
                        severity = 0.1
                    education_events.append({
                        'type': 'egitim_degisikligi',
                        'category': 'egitim_degisiklikleri',
                        'severity': severity,
                        'location': 'Türkiye', 
                        'source': 'MEB',
                        'description': entry.title
                    })
        except:
            pass
        return education_events
class RealTimeImpactCalculator:
    """Gerçek zamanlı etki analizi"""
    
    @staticmethod
    def calculate_real_time_impact(anomalies):
        """Tüm anomalilerin toplam etkisini hesapla"""
        
        if not anomalies:
            return 1.0, []
        
        total_impact = 1.0
        impact_report = []
        
        for anomaly in anomalies:
            geographic_impact = GeographicImpactCalculator.calculate_impact(
                anomaly['location'], 
                anomaly['type'],
                anomaly['severity']
            )
            
            impact_multiplier = 1 - geographic_impact
            total_impact *= impact_multiplier
            
            impact_report.append({
                'type': anomaly['type'],
                'location': anomaly['location'],
                'severity': anomaly['severity'],
                'geographic_impact': geographic_impact,
                'description': anomaly['description'],
                'source': anomaly['source']
            })
        
        final_impact = max(0.1, min(0.9, total_impact))
        return final_impact, impact_report
class ConfidenceScoreV2:
    
    @staticmethod
    def compute(prediction, historical_baseline, period_info, anomaly_type=None, real_time_impact=1.0):

        # 1) Tarihsel aralığa yakınlık (0-40 puan)
        q25, q75 = historical_baseline['q25'], historical_baseline['q75']
        
        if prediction < q25:
            hist_score = max(0, 40 - (q25 - prediction) / (q75 - q25 + 1e-6) * 40)
        elif prediction > q75:
            hist_score = max(0, 40 - (prediction - q75) / (q75 - q25 + 1e-6) * 40)
        else:
            hist_score = 40  # Tam aralıkta
        
        # 2) Okul dönemine uyum (0-25 puan)
        pmin, pmax = period_info['min'], period_info['max']
        if pmin <= prediction <= pmax:
            period_score = 25
        else:
            dist = min(abs(prediction - pmin), abs(prediction - pmax))
            period_score = max(0, 25 - dist / (pmax - pmin + 1e-6) * 25)
        
        # 3) Rejim değişikliği etkisi (0-15 puan)
        if anomaly_type == 'major_drop':
            anomaly_score = 5
        elif anomaly_type == 'major_spike':
            anomaly_score = 8
        else:
            anomaly_score = 15
        
        # ⭐⭐⭐ 4) DÜZELTME: Real-time impact (0-10 puan) ⭐⭐⭐
        if real_time_impact < 1.0:
            # Impact ne kadar düşükse skor o kadar düşer
            event_penalty = (1 - real_time_impact) * 10
        else:
            event_penalty = 0
        
        world_score = max(0, 10 - event_penalty)
        
        # 5) Median yakınlığı (0-10 puan)
        med = historical_baseline['median']
        median_score = max(0, 10 - abs(prediction - med) / (med + 1e-6) * 10)
        
        # TOPLAM SKOR
        raw_score = hist_score + period_score + anomaly_score + world_score + median_score
        final_score = max(0, min(99, raw_score))  # Max 99%
        return round(final_score, 1)
# ==================== GERÇEK ZAMANLI TARİH YÖNETİMİ ====================
def get_dynamic_prediction_dates(historical_data, weeks_ahead=52):
    """GERÇEK ZAMANLI tarih yönetimi - her çalıştırmada bugünden başlar"""
    
    today = date.today()
    last_data_date = historical_data['ds'].max().date()
    gap_days = (today - last_data_date).days
    
    print(f"🌍 GERÇEK ZAMANLI TARİH SİSTEMİ:")
    print(f"   Son veri: {last_data_date}")
    print(f"   Bugün: {today}")
    
    # Boşluk analizi (sadece bilgi amaçlı)
    if gap_days > 0:
        print(f"   ℹ️  Veri boşluğu: {gap_days} gün - model pattern'leri kullanacak")
    
    # Önümüzdeki pazartesiyi bul (GERÇEK ZAMANLI)
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:  # Zaten pazartesi
        start_date = today + timedelta(days=7)
    else:
        start_date = today + timedelta(days=days_until_monday)
    
    print(f"   Tahmin başlangıcı: {start_date}")
    return start_date
def get_decaying_impact(real_time_impact, week_num):
    """Zamanla azalan etki - hafta 1: %100, hafta 4: %25"""
    decay_factors = {
        1: 1.0,   # Hafta 1: %100 etki
        2: 0.75,  # Hafta 2: %75 etki
        3: 0.5,   # Hafta 3: %50 etki  
        4: 0.25   # Hafta 4: %25 etki
    }
    
    decay_factor = decay_factors.get(week_num, 0.0)  # 5+ hafta: %0 etki
    return 1.0 - ((1.0 - real_time_impact) * decay_factor)
# ==================== YENİ: AKILLI POST-PROCESSING FONKSİYONU ====================
def apply_intelligent_adjustments(prediction, historical_data, current_date):
    if len(historical_data) == 0:
        return prediction
    
    # 1. Extreme weeks detection
    low_threshold = historical_data['y'].quantile(0.08)
    extreme_weeks = set(historical_data[historical_data['y'] < low_threshold]['ds'].dt.strftime('%Y-%m-%d'))
    current_date_str = current_date.strftime('%Y-%m-%d')
    
    if current_date_str in extreme_weeks:
        hist_values = []
        for lag in [66, 132, 198]:
            if len(historical_data) >= lag:
                hist_values.append(historical_data['y'].iloc[-lag])
        if hist_values:
            hist_avg = np.mean(hist_values)
            prediction = 0.25 * prediction + 0.75 * hist_avg
    
    # 2. Low season adjustment
    rolling_52 = historical_data['y'].rolling(min(52, len(historical_data))).mean()
    if len(historical_data) > 0:
        is_low_season = rolling_52.iloc[-1] <= rolling_52.quantile(0.25)
    else:
        is_low_season = False
    
    if is_low_season:
        hist_values = []
        for lag in [66, 132]:
            if len(historical_data) >= lag:
                hist_values.append(historical_data['y'].iloc[-lag])
        if hist_values:
            pos_avg = np.mean(hist_values)
            prediction = 0.5 * prediction + 0.5 * pos_avg
    
    # 3. Z-score adjustment
    if len(historical_data) >= 12:
        recent_mean = historical_data['y'].tail(12).mean()
        recent_std = historical_data['y'].tail(12).std() + 1e-6
    else:
        recent_mean = historical_data['y'].mean()
        recent_std = historical_data['y'].std() + 1e-6
    
    z_score = abs(prediction - recent_mean) / recent_std
    
    if z_score > 3.0:
        prediction = 0.65 * prediction + 0.35 * recent_mean
    
    # 4. Min/max clipping
    min_val = historical_data['y'].quantile(0.02)
    max_val = historical_data['y'].quantile(0.98) * 1.15
    prediction = np.clip(prediction, min_val, max_val)
    
    return prediction
def fill_missing_weeks(historical_data, target_start_date):
    last_date = historical_data['ds'].max()
    
    # Timestamp ise date'e çevir
    if hasattr(last_date, 'date'):
        last_date = last_date.date()
    if hasattr(target_start_date, 'date'):
        target_start_date = target_start_date.date()
    
    gap_days = (target_start_date - last_date).days
    
    if gap_days <= 7:
        print("   ✅ Veri güncel - synthetic data gerekmedi")
        return historical_data
    
    print(f"   📊 {gap_days} günlük boşluk tespit edildi - synthetic data oluşturuluyor...")
    
    missing_weeks = []
    weeks_to_fill = gap_days // 7
    
    for week_offset in range(1, weeks_to_fill + 1):
        new_date = last_date + timedelta(weeks=week_offset)
        
        same_week_data = historical_data[
            historical_data['ds'].dt.isocalendar().week == new_date.isocalendar().week
        ]['y'].tail(3)
        
        if len(same_week_data) >= 2:
            weights = np.exp(np.linspace(-1, 0, len(same_week_data)))
            weights /= weights.sum()
            synthetic_value = np.average(same_week_data.values, weights=weights)
        else:
            synthetic_value = historical_data['y'].tail(12).mean()
        
        # ⭐ DEBUG EKLE:
        print(f"   Synthetic hafta {week_offset} ({new_date}): {synthetic_value:.0f}")
        
        missing_weeks.append({
            'ds': pd.Timestamp(new_date),
            'y': synthetic_value
        })

    
    print(f"   ✅ {weeks_to_fill} haftalık synthetic data eklendi")
    
    filled_data = pd.concat([
        historical_data, 
        pd.DataFrame(missing_weeks)
    ], ignore_index=True)
    
    return filled_data
# ==================== GELİŞMİŞ FEATURE ENGINEERING ====================
def create_features2(target_date, historical_data, base_date):
    row = {}
    
    # ⭐ BASE_DATE TİP DÜZELTMESİ
    if hasattr(base_date, 'date'):
        base_date = base_date.date()
    elif hasattr(base_date, 'timestamp'):
        base_date = base_date.date()
    
    # --- Time features ---
    row['ay'] = target_date.month
    row['gun'] = target_date.day
    row['hafta'] = target_date.isocalendar().week
    row['yilin_gunu'] = target_date.timetuple().tm_yday
    row['ceyrek'] = (target_date.month-1)//3 + 1
    
    # --- Trig features ---
    row['week_sin'] = np.sin(2*np.pi*row['hafta']/52)
    row['week_cos'] = np.cos(2*np.pi*row['hafta']/52)
    row['month_sin'] = np.sin(2*np.pi*row['ay']/12)
    row['month_cos'] = np.cos(2*np.pi*row['ay']/12)
    row['quarter_sin'] = np.sin(2*np.pi*row['ceyrek']/4)
    row['quarter_cos'] = np.cos(2*np.pi*row['ceyrek']/4)
    
    # ⭐⭐⭐ CYCLE FEATURE DÜZELTMESİ ⭐⭐⭐
    weeks_since_start = int((target_date - base_date).days / 7)
    
    # MODULO ile döngüsel hale getir (extrapolation önlenir)
    row['cycle_position'] = weeks_since_start % 66
    
    # Cycle multiplier
    cycle_window = min(weeks_since_start, 66)
    if len(historical_data) > 0:
        row['cycle_multiplier'] = historical_data['y'].iloc[-cycle_window:].mean() / (historical_data['y'].mean() + 1e-6) if cycle_window >= 2 else 1.0
    else:
        row['cycle_multiplier'] = 1.0
    
    # Trigonometric cycle features
    row['cycle_sin'] = np.sin(2*np.pi*row['cycle_position']/66)
    row['cycle_cos'] = np.cos(2*np.pi*row['cycle_position']/66)
    
    # --- Season flags ---
    if len(historical_data) > 0:
        rolling_52 = historical_data['y'].rolling(min(52, len(historical_data))).mean()
        row['is_low_season'] = int(rolling_52.iloc[-1] <= rolling_52.quantile(0.25)) if len(rolling_52) > 0 else 0
        row['is_high_season'] = int(rolling_52.iloc[-1] >= rolling_52.quantile(0.75)) if len(rolling_52) > 0 else 0
    else:
        row['is_low_season'] = 0
        row['is_high_season'] = 0
    
    row['is_yaz'] = int(target_date.month in [7,8] or (target_date.month==6 and target_date.day>=25))
    row['is_kis'] = int(target_date.month in [12,1,2])
    row['is_ilkbahar'] = int(target_date.month in [3,4,5])
    
    # --- Lag features ---
    for lag in [1, 2, 3, 4, 8, 12, 26, 52, 66]:
        if len(historical_data) >= lag:
            row[f'lag_{lag}'] = historical_data['y'].iloc[-lag]
        else:
            row[f'lag_{lag}'] = historical_data['y'].mean() if len(historical_data) > 0 else 0
    
    # --- Lag ratio ---
    if len(historical_data) >= 66:
        row['lag1_to_lag66_ratio'] = historical_data['y'].iloc[-1] / (historical_data['y'].iloc[-66] + 1e-6)
    else:
        row['lag1_to_lag66_ratio'] = 1.0
    
    # --- Rolling stats ---
    for window in [4, 8, 12, 26]:
        w = min(window, len(historical_data))
        if w > 0:
            w_data = historical_data['y'].iloc[-w:]
            row[f'roll{window}_mean'] = w_data.mean()
            row[f'roll{window}_std'] = w_data.std()
            row[f'roll{window}_cv'] = w_data.std() / (w_data.mean() + 1e-6)
        else:
            row[f'roll{window}_mean'] = 0
            row[f'roll{window}_std'] = 0
            row[f'roll{window}_cv'] = 0
    
    # --- EMA ---
    if len(historical_data) > 0:
        ema_window = min(8, len(historical_data))
        weights = np.exp(np.linspace(-1, 0, ema_window))
        weights /= weights.sum()
        row['ema_8'] = np.sum(historical_data['y'].iloc[-ema_window:].values * weights)
    else:
        row['ema_8'] = 0
    
    # --- Trend & momentum ---
    if len(historical_data) >= 12:
        y_vals = historical_data['y'].iloc[-12:].values
        row['trend_12'] = np.polyfit(range(len(y_vals)), y_vals, 1)[0]
        row['momentum_12'] = y_vals[-1] - y_vals[0]
    else:
        row['trend_12'] = 0
        row['momentum_12'] = 0
    
    # --- Cycle stats ---
    if len(historical_data) >= 66:
        same_cycle_vals = historical_data['y'].iloc[-(66*min(4, len(historical_data)//66)):].values[::66]
        row['cycle_avg'] = same_cycle_vals.mean()
        row['cycle_std'] = same_cycle_vals.std()
        row['cycle_cv'] = row['cycle_std'] / (row['cycle_avg'] + 1e-6)
    else:
        row['cycle_avg'] = row['roll8_mean']
        row['cycle_std'] = row['roll8_std']
        row['cycle_cv'] = row['roll8_cv']
    
    # --- Volatility ---
    if len(historical_data) >= 8:
        vol_data = historical_data['y'].iloc[-8:]
        row['volatility_8'] = vol_data.std() / (vol_data.mean() + 1e-6)
    else:
        row['volatility_8'] = 0
    
    # --- Value to cycle ratio ---
    if len(historical_data) > 0:
        row['value_to_cycle_ratio'] = historical_data['y'].iloc[-1] / (row['cycle_avg'] + 1e-6)
    else:
        row['value_to_cycle_ratio'] = 1.0
    
    return row
# ==================== ANA TAHMİN FONKSİYONU (TAM ENTEGRE) ====================
def gelecek_hafta_login_tahmini(df, weeks_ahead=52):
    print("\n" + "="*80)
    print("🌍 GELİŞMİŞ GERÇEK ZAMANLI TAHMİN SİSTEMİ v2.0")
    print("="*80)
    
    # ========== GERÇEK ZAMANLI ANOMALİ TARAMA ==========
    world_monitor = RealTimeWorldMonitor()
    impact_calculator = RealTimeImpactCalculator()
    
    anomalies = world_monitor.check_all_anomalies()
    real_time_impact, impact_report = impact_calculator.calculate_real_time_impact(anomalies)
    
    # ========== ETKİ RAPORU ==========
    if impact_report:
        print(f"\n⚠️  GERÇEK ZAMANLI ETKİ RAPORU:")
        print(f"   Toplam Etki: Önümüzdeki 4 hafta için %{(1-real_time_impact)*100:.1f} düşürülecek")
        
        for report in impact_report[:3]:
            print(f"   - {report['type'].upper()}: {report['location']} (%{report['geographic_impact']*100:.1f} etki)")
    else:
        print("\n✅ Anomali tespit edilmedi - normal tahmin modu")
    
    # ========== VERİ HAZIRLAMA ==========
    df = df.dropna(subset=['Login Timestamp']).copy()
    haftalik = df.set_index('Login Timestamp').resample('W-MON')['Login Successful'].sum().reset_index()
    haftalik.rename(columns={'Login Timestamp':'ds','Login Successful':'y'}, inplace=True)
    
    print(f"\n📊 Mevcut veri: {len(haftalik)} hafta")
    print(f"📅 Veri aralığı: {haftalik['ds'].min().date()} → {haftalik['ds'].max().date()}")
    
    # ⭐⭐⭐ YENİ: GERÇEK ZAMANLI TARİH YÖNETİMİ ⭐⭐⭐
    start_date = get_dynamic_prediction_dates(haftalik, weeks_ahead)
    
    # ⭐⭐⭐ YENİ: SYNTHETIC DATA İLE VERİ BOŞLUĞUNU DOLDUR ⭐⭐⭐
    haftalik = fill_missing_weeks(haftalik, start_date)
    
    print(f"📊 Güncel veri: {len(haftalik)} hafta (synthetic data dahil)")
    
    # ========== BAYRAM TATİLLERİ ==========
    future_holidays = {
        '2025-03-30': 'ramazan', '2025-03-31': 'ramazan', '2025-04-01': 'ramazan',
        '2025-06-06': 'kurban', '2025-06-07': 'kurban', '2025-06-08': 'kurban', '2025-06-09': 'kurban',
        '2025-08-30': 'zafer', '2025-10-29': 'cumhuriyet',
        '2026-01-01': 'yilbasi',
        '2026-03-20': 'ramazan', '2026-03-21': 'ramazan', '2026-03-22': 'ramazan',
        '2026-05-27': 'kurban', '2026-05-28': 'kurban', '2026-05-29': 'kurban', '2026-05-30': 'kurban',
        '2026-08-30': 'zafer', '2026-10-29': 'cumhuriyet'
    }
    
    # ========== ANOMALİ YÖNETİMİ ==========
    anomaly_manager = AnomalyManager()
    has_anomaly, anomaly_type = anomaly_manager.detect_regime_change(haftalik)
    
    if has_anomaly:
        print(f"⚠️  TARİHSEL ANOMALİ TESPİT EDİLDİ: {anomaly_type}")
    else:
        print("✅ Tarihsel veri stabil")
    
    # ========== BASE DATE ==========
    base_date = haftalik.iloc[0]['ds'].date()
    
    # ========== MODEL EĞİTİMİ ==========
    X_train_list, y_train_list = [], []
    
    for idx in range(52, len(haftalik)):
        historical_data = haftalik.iloc[:idx]
        target_date = haftalik.iloc[idx]['ds'].date()
        X_train_list.append(create_features2(target_date, historical_data, base_date))
        y_train_list.append(haftalik.iloc[idx]['y'])
    
    X_train = pd.DataFrame(X_train_list).fillna(0)
    y_train = np.array(y_train_list)
    
    # XGBoost model
    model = XGBRegressor(
        n_estimators=500, 
        max_depth=12, 
        learning_rate=0.05,
        subsample=0.85, 
        colsample_bytree=0.85, 
        gamma=0,
        reg_alpha=1.5, 
        reg_lambda=2.0,
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # ========== GELECEK TAHMİNLERİ ==========
    calendar = SchoolCalendar()
    future_predictions = []
    current_data = haftalik.copy()
    
    for week_num in range(1, weeks_ahead + 1):
        if week_num == 1:
            next_date = start_date
        else:
            next_date = future_predictions[-1]['tarih'] + timedelta(days=7)
        
        # Feature creation
        features = create_features2(next_date, current_data, base_date)
        X_future = pd.DataFrame([features])[X_train.columns].fillna(0)
        
        # ML Prediction
        ml_prediction = model.predict(X_future)[0]
        
        # Post-processing
        ml_prediction = apply_intelligent_adjustments(ml_prediction, current_data, next_date)  # ✅ Parametre kaldırıldı
        
        # Historical & Calendar baselines
        historical_baseline = calendar.get_historical_baseline(next_date, current_data)
        historical_pred = historical_baseline['weighted_mean']
        
        period_info = calendar.get_period_info(next_date)
        calendar_pred = period_info['baseline']
        
        # ⭐⭐⭐ YENİ: HİBRİT AĞIRLIK SİSTEMİ (Normal tahmin performansına göre) ⭐⭐⭐
        if historical_baseline['count'] >= 6:
            # Yeterli tarihsel veri var - ML'e güven artır (normal tahmin %5.82 MAPE)
            if next_date.month == 8 and next_date.day <= 21:
                # Ağustos tatil - tarihsel ağırlık biraz artır
                hybrid_prediction = 0.50 * ml_prediction + 0.35 * historical_pred + 0.15 * calendar_pred
            elif next_date.month == 9 and next_date.day < 15:
                # Eylül hazırlık - tarihsel ağırlık yüksek
                hybrid_prediction = 0.45 * ml_prediction + 0.40 * historical_pred + 0.15 * calendar_pred
            else:
                # Normal dönem - ML ağırlığı yüksek
                hybrid_prediction = 0.65 * ml_prediction + 0.25 * historical_pred + 0.10 * calendar_pred
        else:
            # Az veri - tarihsel ve takvime güven
            hybrid_prediction = 0.30 * ml_prediction + 0.45 * historical_pred + 0.25 * calendar_pred
        
        # Period factor
        hybrid_prediction *= period_info['factor']
        
        # Anomaly adjustment
        hybrid_prediction = anomaly_manager.adjust_for_anomaly(
            hybrid_prediction, anomaly_type, historical_baseline, period_info
        )
        
        # ⭐⭐⭐ YENİ: GERÇEK ZAMANLI ETKİ (Sadece ilk 4 hafta) ⭐⭐⭐
        if week_num <= 4 and real_time_impact < 0.95:
            decaying_impact = get_decaying_impact(real_time_impact, week_num)
            original_pred = hybrid_prediction
            hybrid_prediction *= decaying_impact
            
            if real_time_impact < 0.9:
                impact_pct = (1 - decaying_impact) * 100
                print(f"   📉 Hafta {week_num} gerçek dünya etkisi: %{impact_pct:.1f} ({original_pred:.0f} → {hybrid_prediction:.0f})")
        
        # Bayram kontrolü
        next_date_str = next_date.strftime('%Y-%m-%d')
        is_holiday = next_date_str in future_holidays
        holiday_type = future_holidays.get(next_date_str, '')
        
        if is_holiday:
            if 'ramazan' in holiday_type or 'kurban' in holiday_type:
                hybrid_prediction *= 0.45
            else:
                hybrid_prediction *= 0.70
        
        # Final clipping
        final_prediction = np.clip(hybrid_prediction, period_info['min'], period_info['max'])
        lower_bound = np.clip(final_prediction * 0.80, period_info['min'] * 0.75, period_info['max'])
        upper_bound = np.clip(final_prediction * 1.30, period_info['min'], period_info['max'] * 1.15)
        
        # ⭐⭐⭐ YENİ: DÜZELTİLMİŞ CONFIDENCE SCORE ⭐⭐⭐
        confidence = ConfidenceScoreV2.compute(
            final_prediction, 
            historical_baseline, 
            period_info, 
            anomaly_type, 
            real_time_impact if week_num <= 4 else 1.0  # 5+ hafta sonra etki yok
        )
        
        future_predictions.append({
            'hafta': week_num,
            'tarih': next_date,
            'tahmin': int(round(final_prediction)),
            'alt_sinir': int(round(lower_bound)),
            'ust_sinir': int(round(upper_bound)),
            'ml_tahmin': int(round(ml_prediction)),
            'gecmis_tahmin': int(round(historical_pred)),
            'takvim_tahmin': int(round(calendar_pred)),
            'guven': confidence,
            'veri_sayisi': historical_baseline['count'],
            'gercek_zamanli_etki': get_decaying_impact(real_time_impact, week_num) if week_num <= 4 else 1.0
        })
        
        # ⭐ Tahmin edilen değeri veri setine ekle (sonraki haftalar için lag)
        new_row = pd.DataFrame([{
            'ds': pd.Timestamp(next_date),
            'y': final_prediction
        }])
        current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    # ========== SONUÇLAR ==========
    results = pd.DataFrame(future_predictions)
    
    print("\n" + "="*80)
    print("📊 GELİŞMİŞ HİBRİT TAHMİN SONUÇLARI (v2.0)")
    print("="*80)
    
    if len(results) > 0:
        display_cols = ['hafta', 'tarih', 'tahmin', 'alt_sinir', 'ust_sinir', 'guven']
        results['guven'] = results['guven'].apply(lambda x: f"{x:.1f}%")
        print(results[display_cols].head(52).to_string(index=False))
    
    return results
# ==================== 6. OS 4 HAFTALIK TAHMİN (ORİJİNAL MANTIK) ====================
def os_4haftalik_tahmin(df):
    print("\n📊 OS BAZLI 4 HAFTALIK TAHMİN\n")
    
    populer_os = df["OS Name and Version"].value_counts().head(4).index
    all_results = []

    for os_name in populer_os:
        df_os = df[df["OS Name and Version"] == os_name].copy()
        if df_os.empty:
            continue

        # Haftalık veri hazırlama
        haftalik = (df_os.set_index("Login Timestamp")
                     .resample("W-MON")
                     .size()
                     .reset_index(name="gercek_login"))
        haftalik.rename(columns={"Login Timestamp": "ds"}, inplace=True)
        
        if len(haftalik) < 20:
            continue

        train_data = haftalik.iloc[:-4].copy()
        test_data = haftalik.iloc[-4:].copy()
        valid = test_data.copy()
        valid["OS"] = os_name

        # =============== XGBOOST ===============
        try:
            def create_features(df):
                df = df.copy()
                df['week'] = df['ds'].dt.isocalendar().week
                df['month'] = df['ds'].dt.month
                
                for lag in [1, 4, 52]:
                    if len(df) > lag:
                        df[f'lag_{lag}'] = df['gercek_login'].shift(lag)
                
                if len(df) > 4:
                    df['rolling_mean_4'] = df['gercek_login'].shift(1).rolling(window=4, min_periods=1).mean()
                
                return df

            train_features = create_features(train_data).dropna()
            
            feature_cols = [col for col in train_features.columns if col not in ['ds', 'gercek_login']]
            X_train = train_features[feature_cols]
            y_train = train_features['gercek_login']
            
            xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.3, max_depth=6, random_state=42)
            xgb_model.fit(X_train, y_train)
            
            xgb_predictions = []
            current_data = train_data.copy()
            
            for i in range(4):
                current_features = create_features(current_data)
                latest_features = current_features.iloc[[-1]][feature_cols]
                pred = xgb_model.predict(latest_features)[0]
                pred = max(0, round(pred))
                xgb_predictions.append(pred)
                
                new_date = current_data.iloc[-1]['ds'] + pd.Timedelta(weeks=1)
                new_row = pd.DataFrame({'ds': [new_date], 'gercek_login': [np.nan]})
                current_data = pd.concat([current_data, new_row], ignore_index=True)
            
            valid["xgb_tahmin"] = xgb_predictions
            
        except Exception as e:
            avg = train_data['gercek_login'].tail(13).mean()
            valid["xgb_tahmin"] = [int(avg)] * 4

        # =============== SARIMA ===============
        try:
            best_model, best_params = optimize_sarima(train_data['gercek_login'])
            sarima_pred = best_model.get_forecast(steps=4).predicted_mean.round().astype(int).clip(lower=0)
            valid["sarima_tahmin"] = sarima_pred.values
        except:
            avg = train_data['gercek_login'].tail(13).mean()
            valid["sarima_tahmin"] = [int(avg)] * 4

        # =============== AĞUSTOS DÜZELTMESİ ===============
        for idx, row in valid.iterrows():
            if row['ds'].month == 8:
                august_values = train_data[train_data['ds'].dt.month == 8]['gercek_login']
                if len(august_values) > 0:
                    august_median = august_values.median()
                    # XGBoost'u güçlü düzelt
                    valid.loc[idx, 'xgb_tahmin'] = int(0.2 * valid.loc[idx, 'xgb_tahmin'] + 0.8 * august_median)
                    # SARIMA'yı hafif düzelt
                    valid.loc[idx, 'sarima_tahmin'] = int(0.7 * valid.loc[idx, 'sarima_tahmin'] + 0.3 * august_median)

        # Hata hesaplama
        valid["xgb_mutlak_hata"] = abs(valid["gercek_login"] - valid["xgb_tahmin"])
        valid["xgb_hata_yuzdesi"] = (valid["xgb_mutlak_hata"] / valid["gercek_login"] * 100).round(2)
        valid["sarima_mutlak_hata"] = abs(valid["gercek_login"] - valid["sarima_tahmin"])
        valid["sarima_hata_yuzdesi"] = (valid["sarima_mutlak_hata"] / valid["gercek_login"] * 100).round(2)

        all_results.append(valid)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        
        column_order = ['ds', 'OS', 'gercek_login', 'xgb_tahmin', 'sarima_tahmin', 
                       'xgb_mutlak_hata', 'xgb_hata_yuzdesi', 'sarima_mutlak_hata', 'sarima_hata_yuzdesi']
        df_all = df_all[column_order]
        
        print(df_all.to_string(index=False))
        
        # Özet
        print(f"\n📊 ORTALAMA HATA:")
        print(f"   XGBoost: %{df_all['xgb_hata_yuzdesi'].mean():.1f}")
        print(f"   SARIMA: %{df_all['sarima_hata_yuzdesi'].mean():.1f}")
        
        return df_all

    print("⚠️ Hiçbir OS için geçerli tahmin üretilemedi.")
    return None
def optimize_sarima(train_series, seasonal_period=13, max_combinations=20):
    """Basitleştirilmiş SARIMA optimizasyonu"""
    import itertools
    import warnings
    
    # Temel parametreler
    p_range, d_range, q_range = range(0, 3), range(0, 2), range(0, 3)
    P_range, D_range, Q_range = range(0, 2), range(0, 2), range(0, 2)
    
    pdq = list(itertools.product(p_range, d_range, q_range))
    seasonal_pdq = list(itertools.product(P_range, D_range, Q_range, [seasonal_period]))
    
    combinations = list(itertools.product(pdq, seasonal_pdq))[:max_combinations]
    
    best_aic, best_model, best_params = np.inf, None, None
    
    for param, param_seasonal in combinations:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(train_series, order=param, seasonal_order=param_seasonal,
                              enforce_stationarity=False, enforce_invertibility=False)
                fitted_model = model.fit(disp=False, maxiter=100)
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    best_params = (param, param_seasonal)
        except:
            continue
    
    # Fallback
    if best_model is None:
        model = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))
        best_model = model.fit(disp=False)
        best_params = ((1,1,1), (1,1,1,seasonal_period))
    
    return best_model, best_params
# ==================== 7. ANOMALİ TESPİTİ ====================
def anomali_tespiti(df, contamination=None):
    df = df.rename(columns={
        "loginTime": "Login Timestamp", 
        "browser": "Browser Name and Version",
        "os": "OS Name and Version",
        "clientName": "Client Name"
    })

    df["Login Timestamp"] = pd.to_datetime(df["Login Timestamp"])
    df["Tarih"] = df["Login Timestamp"].dt.date
    df["Saat"] = df["Login Timestamp"].dt.hour

    user_daily = df.groupby(["key", "Tarih"]).agg({
        "Saat": ["count", "std"],
        "Browser Name and Version": "nunique", 
        "OS Name and Version": "nunique",
        "Client Name": "nunique"
    }).reset_index()
    user_daily.columns = ["ID", "Tarih", "Toplam_Login", "Login_Std", "Farkli_Browser", "Farkli_OS", "Farkli_App"]
    user_daily["Login_Std"] = user_daily["Login_Std"].fillna(0)

    # ⭐⭐⭐ İSİM DEĞİŞTİ: "Gece" → "Gece_Login_Sayisi" ⭐⭐⭐
    gece_loginleri = df[(df["Saat"] >= 0) & (df["Saat"] <= 5)].groupby(["key", "Tarih"]).size().reset_index(name="Gece_Login_Sayisi")
    gece_loginleri.rename(columns={"key": "ID"}, inplace=True)
    user_daily = user_daily.merge(gece_loginleri, on=["ID", "Tarih"], how="left").fillna({"Gece_Login_Sayisi": 0})

    # ⭐⭐⭐ KOLON İSİMLERİ GÜNCELLENDİ ⭐⭐⭐
    numeric_cols = ["Toplam_Login", "Login_Std", "Farkli_Browser", "Farkli_OS", "Farkli_App", "Gece_Login_Sayisi"]
    varyans = user_daily[numeric_cols].var()
    aktif_kolonlar = varyans[varyans > 0].index.tolist()

    X = StandardScaler().fit_transform(user_daily[aktif_kolonlar])

    if contamination is None:
        contamination = min(0.05, max(0.01, 100 / len(user_daily)))

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    user_daily["Anormal"] = np.where(iso.fit_predict(X) == -1, "Anormal", "Normal")

    user_daily["Tarih"] = pd.to_datetime(user_daily["Tarih"])
    user_daily["Ay"] = user_daily["Tarih"].dt.month_name(locale="tr_TR")

    aylik_anomali = (
        user_daily[user_daily["Anormal"] == "Anormal"]
        .groupby("Ay")
        .size()
        .reindex([
            "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
            "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"  
        ])
        .fillna(0)
        .astype(int)
        .reset_index(name="Toplam_Anomali_Sayısı")
    )

    # RASTGELE 20 ANOMALİ KAYIT
    anormal_kayitlar = user_daily[user_daily["Anormal"] == "Anormal"].copy()
    
    if len(anormal_kayitlar) > 20:
        rastgele_anomaliler = anormal_kayitlar.sample(n=20, random_state=42)
    else:
        rastgele_anomaliler = anormal_kayitlar.copy()

    print("\n" + "="*100)
    print("🔍 RASTGELE SEÇİLMİŞ 20 ANOMALİ KAYDI")
    print("="*100)
    
    # ⭐⭐⭐ DAHA ANLAŞILIR KOLON İSİMLERİ ⭐⭐⭐
    print(rastgele_anomaliler[['ID', 'Tarih', 'Toplam_Login', 'Gece_Login_Sayisi', 'Farkli_Browser', 'Farkli_OS', 'Farkli_App']].to_string(index=False))

    print("\n" + "="*100) 
    print("📊 ANOMALİ SEBEPLERİ ANALİZİ")
    print("="*100)
    
    return aylik_anomali, rastgele_anomaliler
# ==================== 8. KÜMELEME ====================
def benzer_login_siniflandir(df, n_clusters=7):
    temp_df = df.rename(columns={
        'browser': 'Browser Name and Version',
        'os': 'OS Name and Version',
        'loginTime': 'Login Timestamp',
        'clientName': 'Client Name'
    }).copy()

    temp_df["Login Timestamp"] = pd.to_datetime(temp_df["Login Timestamp"])
    temp_df["Tarih"] = temp_df["Login Timestamp"].dt.date
    temp_df["Saat"] = temp_df["Login Timestamp"].dt.hour
    temp_df["Gun"] = temp_df["Login Timestamp"].dt.dayofweek

    user_daily = temp_df.groupby(["key","Tarih"]).agg({
        "Saat": ["count","mean","max","min","std"],
        "OS Name and Version": "nunique",
        "Browser Name and Version": "nunique",
        "Client Name": "nunique"
    }).reset_index()

    user_daily.columns = ["ID","Tarih","Login","Saat_Ort","Saat_Max","Saat_Min","Saat_Std","OS","Browser","App"]
    user_daily = user_daily.fillna(0)

    # Saat aralığı hesapla
    user_daily["Saat_Aralik"] = user_daily["Saat_Max"] - user_daily["Saat_Min"]
    
    # Std değeri NaN olanları 0 yap (tek login günleri için)
    user_daily["Saat_Std"] = user_daily["Saat_Std"].fillna(0)

    gece = temp_df[(temp_df["Saat"] >= 0) & (temp_df["Saat"] < 6)].groupby(["key","Tarih"]).size().reset_index(name="Gece")
    gece.rename(columns={"key":"ID"}, inplace=True)
    user_daily = user_daily.merge(gece, on=["ID","Tarih"], how="left").fillna({"Gece":0})

    user_daily["HaftaDavranisi"] = user_daily["Tarih"].apply(
        lambda x: "Hafta Sonu" if pd.Timestamp(x).weekday() >= 5 else "Hafta İçi"
    )

    def gunun_zamani(saat):
        if 5 <= saat < 12: return "Sabah"
        elif 12 <= saat < 17: return "Öğle"
        elif 17 <= saat < 21: return "Akşam"
        else: return "Gece"
    user_daily["GunZamani"] = user_daily["Saat_Ort"].apply(gunun_zamani)

    gunler = {0:"Pazartesi",1:"Salı",2:"Çarşamba",3:"Perşembe",4:"Cuma",5:"Cumartesi",6:"Pazar"}
    user_daily["OrtalamaGun"] = user_daily["Tarih"].apply(lambda x: gunler[pd.Timestamp(x).weekday()])

    # Yeni özellikleri feature_cols'a ekle
    feature_cols = ["Login","Saat_Ort","Saat_Max","Saat_Std","Saat_Aralik","OS","Browser","App","Gece"]
    X = StandardScaler().fit_transform(user_daily[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    user_daily["Grup"] = kmeans.fit_predict(X) + 1

    # -----------------------------
    # temp_df ile user_daily gruplarını birleştir
    # -----------------------------
    temp_df_with_groups = temp_df.merge(
        user_daily[['ID', 'Tarih', 'Grup']],
        left_on=['key', 'Tarih'],
        right_on=['ID', 'Tarih'],
        how='left'
    )

    sample_size = min(80000, len(X))
    sample_idx = np.random.choice(len(X), sample_size, replace=False) if len(X) > sample_size else slice(None)
    silhouette = silhouette_score(X[sample_idx], kmeans.labels_[sample_idx])
    db_index = davies_bouldin_score(X, kmeans.labels_)

    print("\n" + "="*80)
    print("🎯 KÜMELEME SONUÇLARI (Geliştirilmiş - Standart Saat Farkı Eklendi)")
    print("="*80)
    print(f"Silhouette Skoru : {silhouette:.4f}")
    print(f"Davies-Bouldin   : {db_index:.4f}")
    print("="*80)

    for i in range(1, n_clusters+1):
        grup = user_daily[user_daily["Grup"]==i]
        grup_temp = temp_df_with_groups[temp_df_with_groups['Grup']==i]

        print(f"\n📊 Grup {i} ({len(grup):,} kayıt):")
        print(f"Login Ort      : {grup['Login'].mean():.1f}")
        print(f"Ortalama Saat  : {grup['Saat_Ort'].mean():.1f}")
        print(f"Saat Std       : {grup['Saat_Std'].mean():.1f}")  # YENİ
        print(f"Saat Aralık    : {grup['Saat_Aralik'].mean():.1f}")  # YENİ
        print(f"Günün Zamanı   : {grup['GunZamani'].mode()[0]}")
        print(f"En sık HaftaDavranisi : {grup['HaftaDavranisi'].mode()[0]}")
        print(f"Ortalama Gün   : {grup['OrtalamaGun'].mode()[0]}")
        print(f"En sık OS       : {grup_temp['OS Name and Version'].mode()[0]}")
        print(f"En sık Browser  : {grup_temp['Browser Name and Version'].mode()[0]}")
        print(f"En sık Client   : {grup_temp['Client Name'].mode()[0]}")
        print("-"*60)

    return user_daily
# ==================== ANA PROGRAM ====================
def main():
    print("🚀 LOGIN ANALİZ SİSTEMİ - PERFORMANS ODAKLI")
    dosya_yolu = r"C:\Users\Aykut\AppData\Local\Programs\Microsoft VS Code\2login_data_5years_10M.jsonl"
    
    df = load_data_optimized(dosya_yolu)
    
    menu_options = {
        '1': ("OS/Browser Yoğunluk", os_browser_yogunluk_analizi),
        '2': ("Client Analizi", client_os_browser_analizi),
        '3': ("Saat/Gün Tahmini", saat_gun_model),
        '4': ("Haftalık Tahmin", haftalik_login_tahmini),
        '5': ("Gelecek Hafta Tahmini", gelecek_hafta_login_tahmini),
        '6': ("OS 4 Haftalık Tahmin", os_4haftalik_tahmin),
        '7': ("Anomali Tespiti", anomali_tespiti),
        '8': ("Kümeleme", benzer_login_siniflandir),
        '9': ("Çıkış", exit)
    }
    
    while True:
        print("\n" + "="*50)
        print("LOGIN ANALİZ SİSTEMİ")
        print("="*50)
        
        for key, (label, _) in menu_options.items():
            print(f"{key}- {label}")
        
        secim = input("\nSeçim: ").strip()
        
        if secim in menu_options:
            if secim == '9':
                print("Çıkış yapılıyor...")
                break
            menu_options[secim][1](df)
        else:
            print("⚠️ Geçersiz seçim!")

if __name__ == "__main__":
    main()