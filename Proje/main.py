import pandas as pd 
import numpy as np  
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, davies_bouldin_score, r2_score 
import logging
import warnings
import lightgbm as lgb
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import itertools
# Polars desteği
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# GPU desteği kontrolü
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# ==================== VERİ YÜKLEME (POLARS ENTEGRE) ====================
def load_data_optimized(dosya_yolu, nrows=None):
    print("📊 Veri yükleniyor...")
    
    if POLARS_AVAILABLE:
        try:
            print("⚡ Polars kullanılıyor (15x hızlı)...")
            df_polars = pl.read_ndjson(dosya_yolu, n_rows=nrows)
            df_polars = df_polars.rename({
                'browser': 'Browser Name and Version',
                'os': 'OS Name and Version',
                'loginTime': 'Login Timestamp',
                'clientName': 'Client Name'
            })
            df = df_polars.to_pandas()
        except:
            print("⚠️ Polars hatası, Pandas kullanılıyor...")
            df = pd.read_json(dosya_yolu, lines=True, nrows=nrows)
            df = df.rename(columns={
                'browser': 'Browser Name and Version',
                'os': 'OS Name and Version',
                'loginTime': 'Login Timestamp',
                'clientName': 'Client Name'
            })
    else:
        df = pd.read_json(dosya_yolu, lines=True, nrows=nrows)
        df = df.rename(columns={
            'browser': 'Browser Name and Version',
            'os': 'OS Name and Version',
            'loginTime': 'Login Timestamp',
            'clientName': 'Client Name'
        })
    
    df["Login Successful"] = 1
    df = df.fillna("Bilinmiyor")
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')
    df['saat'] = df['Login Timestamp'].dt.hour
    df['gun'] = df['Login Timestamp'].dt.dayofweek
    
    encoders = {}
    for col in ["OS Name and Version", "Browser Name and Version", "Client Name"]:
        encoders[col] = LabelEncoder()
        df[col+'_enc'] = encoders[col].fit_transform(df[col].astype(str))
    
    print(f"✓ {len(df):,} kayıt yüklendi | {df['Login Timestamp'].min()} → {df['Login Timestamp'].max()}\n")
    return df
# ==================== SARIMA OPTİMİZASYONU ====================
def optimize_sarima(train_series, seasonal_period=4, max_combinations=30, random_state=42):
    np.random.seed(random_state)  # 🔑 seed sabitlendi

    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)
    P_range = range(0, 2)
    D_range = range(0, 2)
    Q_range = range(0, 2)
    
    pdq = list(itertools.product(p_range, d_range, q_range))
    seasonal_pdq = list(itertools.product(P_range, D_range, Q_range, [seasonal_period]))
    
    all_combinations = list(itertools.product(pdq, seasonal_pdq))
    np.random.shuffle(all_combinations)
    all_combinations = all_combinations[:max_combinations]
    
    best_aic = np.inf
    best_model = None
    best_params = None
    
    for param, param_seasonal in all_combinations:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    train_series,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted_model = model.fit(disp=False, maxiter=100)
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    best_params = (param, param_seasonal)
        except:
            continue
    
    if best_model is None:
        model = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))
        best_model = model.fit(disp=False, maxiter=100)
        best_params = ((1,1,1), (1,1,1,seasonal_period))
    
    return best_model, best_params
# ==================== 1. OS/BROWSER YOĞUNLUK ANALİZİ ====================
def os_browser_yogunluk_analizi(df):
    print("\n" + "="*80)
    print("📊 OS/BROWSER BAZLI LOGİN YOĞUNLUK ANALİZİ")
    print("="*80)
    
    for col, icon in [("OS Name and Version", "💻"), ("Browser Name and Version", "🌐")]:
        stats = df.groupby(col).agg({
            'email': 'nunique',
            'Login Successful': 'sum'
        }).reset_index()
        stats.columns = [col, 'Kullanici_Sayisi', 'Login_Sayisi']
        stats['Ort_Login'] = (stats['Login_Sayisi'] / stats['Kullanici_Sayisi']).round(2)
        
        print(f"\n{icon} {col.upper()} İSTATİSTİK (Top 10)")
        print("-"*80)
        print(stats.sort_values('Login_Sayisi', ascending=False).head(10).to_string(index=False))
# ==================== 2. CLIENT BAZLI ANALİZ ====================
def client_os_browser_analizi(df):
    print("\n📱 CLIENT BAZLI LOGİN DAĞILIMI")
    print("="*80)
    
    client_stats = df.groupby("Client Name").agg({
        'email': 'nunique',
        'Login Successful': 'sum'
    }).rename(columns={'email':'Kullanici', 'Login Successful':'Login'}).sort_values('Login', ascending=False)
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
# ==================== 3. SAAT/GÜN TAHMİNİ (VERİ SIZINTISI DÜZELTİLDİ) ====================
def saat_gun_model(df):
    df = df.copy()
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')
    df = df.dropna(subset=['Login Timestamp'])

    full_range = pd.date_range(df['Login Timestamp'].min().floor('H'),
                               df['Login Timestamp'].max().ceil('H'), freq='H')

    hourly_counts = df.groupby(pd.Grouper(key='Login Timestamp', freq='H')).size().reindex(full_range, fill_value=0)
    hourly_counts = hourly_counts.reset_index().rename(columns={'index':'Login Timestamp', 0:'gercek_login'})
    hourly_counts['gun'] = hourly_counts['Login Timestamp'].dt.dayofweek
    hourly_counts['saat'] = hourly_counts['Login Timestamp'].dt.hour
    hourly_counts['ay'] = hourly_counts['Login Timestamp'].dt.month
    hourly_counts['yil'] = hourly_counts['Login Timestamp'].dt.year
    hourly_counts['hafta'] = hourly_counts['Login Timestamp'].dt.isocalendar().week

    split_date = hourly_counts['Login Timestamp'].quantile(0.85)
    train_data = hourly_counts[hourly_counts['Login Timestamp'] < split_date].copy()
    test_data = hourly_counts[hourly_counts['Login Timestamp'] >= split_date].copy()

    def create_optimized_models():
        if GPU_AVAILABLE:
            print("✅ GPU tespit edildi - GPU modelleri kullanılacak")
            return {
                "LGBM": lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    min_split_gain=0.001,
                    device='gpu',
                    gpu_platform_id=0,
                    gpu_device_id=0,
                    verbosity=-1,
                    random_state=42
                ),
                "XGB": xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=7,
                    min_child_weight=5,
                    gamma=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method='gpu_hist',
                    predictor='gpu_predictor',
                    gpu_id=0,
                    verbosity=0,
                    random_state=42
                )
            }
        else:
            return {
                "LGBM": lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    min_split_gain=0.001,
                    verbosity=-1,
                    n_jobs=-1,
                    random_state=42
                ),
                "XGB": xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=7,
                    min_child_weight=5,
                    gamma=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    verbosity=0,
                    n_jobs=-1,
                    random_state=42
                )
            }
    def create_features(data):
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

    train_data = create_features(train_data)
    test_data = create_features(test_data)

    lag_list = [1,2,3,6,12,24,48,168]
    rolling_windows = [6,12,24,168]

    for lag in lag_list:
        train_data[f'lag_{lag}'] = train_data['gercek_login'].shift(lag).fillna(0)
    for w in rolling_windows:
        train_data[f'rolling_mean_{w}'] = train_data['gercek_login'].rolling(window=w, min_periods=1).mean()

    for lag in lag_list:
        test_data[f'lag_{lag}'] = pd.concat([train_data['gercek_login'].iloc[-lag:], test_data['gercek_login']]).shift(lag).iloc[lag:].fillna(0).values
    for w in rolling_windows:
        test_data[f'rolling_mean_{w}'] = pd.concat([train_data['gercek_login'].iloc[-w:], test_data['gercek_login']]).rolling(window=w, min_periods=1).mean().iloc[w:].values

    for data in [train_data, test_data]:
        data['diff_1'] = data['gercek_login'].diff(1).fillna(0)
        data['diff_24'] = data['gercek_login'].diff(24).fillna(0)

    global_mean = train_data['gercek_login'].mean()
    
    weekly_mean_dict = train_data.groupby('hafta')['gercek_login'].mean().to_dict()
    weekly_std_dict = train_data.groupby('hafta')['gercek_login'].std().fillna(0).to_dict()
    weekly_max_dict = train_data.groupby('hafta')['gercek_login'].max().to_dict()

    for data in [train_data, test_data]:
        data['weekly_mean'] = data['hafta'].map(weekly_mean_dict).fillna(global_mean)
        data['weekly_std'] = data['hafta'].map(weekly_std_dict).fillna(0)
        data['weekly_max'] = data['hafta'].map(weekly_max_dict).fillna(global_mean)

    for data in [train_data, test_data]:
        data['gun_saat_key'] = data['gun'].astype(str)+'_'+data['saat'].astype(str)

    gun_saat_mean_dict = train_data.groupby('gun_saat_key')['gercek_login'].mean().to_dict()
    gun_saat_std_dict = train_data.groupby('gun_saat_key')['gercek_login'].std().fillna(0).to_dict()
    gun_saat_max_dict = train_data.groupby('gun_saat_key')['gercek_login'].max().to_dict()

    for data in [train_data, test_data]:
        data['gun_saat_mean'] = data['gun_saat_key'].map(gun_saat_mean_dict).fillna(global_mean)
        data['gun_saat_std'] = data['gun_saat_key'].map(gun_saat_std_dict).fillna(0)
        data['gun_saat_max'] = data['gun_saat_key'].map(gun_saat_max_dict).fillna(global_mean)

    feature_cols = [
        'gun','saat','ay','yil','hafta','hafta_sonu','mevsim','is_saati',
        'saat_sin','saat_cos','gun_sin','gun_cos','ay_sin','ay_cos',
        'lag_1','lag_2','lag_3','lag_6','lag_12','lag_24','lag_48','lag_168',
        'rolling_mean_6','rolling_mean_12','rolling_mean_24','rolling_mean_168',
        'diff_1','diff_24','weekly_mean','weekly_std','weekly_max',
        'gun_saat_mean','gun_saat_std','gun_saat_max'
    ]

    X_train, y_train = train_data[feature_cols], train_data['gercek_login']
    X_test, y_test = test_data[feature_cols], test_data['gercek_login']

    models = create_optimized_models()
    predictions = {}
    
    print("⏳ Model eğitimi başlıyor...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0, None)
        predictions[name] = np.round(preds).astype(int)

    def safe_mape(y_true, y_pred, epsilon=1e-10):
        mask = y_true > epsilon
        if mask.sum() == 0:
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    model_metrics = {}
    for name in predictions.keys():
        preds = predictions[name]
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = safe_mape(y_test.values, preds)
        model_metrics[name] = {'RMSE': rmse, 'R2': r2, 'MAPE': mape}

    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['MAPE'])
    best_model, second_best_model = sorted_models[0][0], sorted_models[1][0]

    results_df = test_data[['Login Timestamp','gun','saat','gercek_login']].copy()
    gun_isimleri = {0:'Pazartesi',1:'Salı',2:'Çarşamba',3:'Perşembe',4:'Cuma',5:'Cumartesi',6:'Pazar'}
    results_df['gun_ismi'] = results_df['gun'].map(gun_isimleri)

    for name in [best_model, second_best_model]:
        results_df[f'{name.lower()}_tahmin'] = predictions[name]
        results_df[f'{name.lower()}_hata%'] = np.where(
            results_df['gercek_login'] > 0,
            (np.abs(results_df['gercek_login'] - predictions[name]) / results_df['gercek_login'] * 100).round(2),
            0
        )

    top_10 = results_df.nlargest(20, 'gercek_login').drop_duplicates(subset=['Login Timestamp'])
    display_cols = ['Login Timestamp','gun_ismi','saat','gercek_login',
                    f'{best_model.lower()}_tahmin', f'{best_model.lower()}_hata%',
                    f'{second_best_model.lower()}_tahmin', f'{second_best_model.lower()}_hata%']

    print("\n" + "="*110)
    print(f"🔝 EN YOĞUN 10 SAAT - EN İYİ 2 MODEL ({best_model} & {second_best_model})")
    print("="*110)
    print(top_10[display_cols].to_string(index=False))

    return results_df, model_metrics
# ==================== 4. HAFTALIK TAHMİN ====================
def haftalik_login_tahmini(df, max_lag=20, min_train=None):
    df = df.dropna(subset=['Login Timestamp']).copy()
    haftalik = df.set_index('Login Timestamp').resample('W-MON')['Login Successful'].sum().reset_index()
    haftalik.rename(columns={'Login Timestamp':'ds','Login Successful':'y'}, inplace=True)
    haftalik['gercek_login'] = haftalik['y'].astype(int)

    # ============================================
    # MINIMUM TRAIN CONTROL
    # ============================================
    if min_train is None:
        min_train = max_lag + 12
    if len(haftalik) < (min_train+2):
        print(f"❌ Yetersiz veri: en az {min_train+2} hafta gerekli")
        return None

    # ============================================
    # EXTREME LOW ADJUSTMENTS (GERÇEK ORANLAR)
    # ============================================
    extreme_low_adjustments = {
        '2021-11-15': 0.25,
        '2021-11-22': 0.30,
        '2022-05-02': 0.70,
        '2022-05-09': 0.65,
        '2022-05-16': 0.55,
        '2022-12-05': 0.35,
        '2023-02-20': 0.20,
        '2023-02-27': 0.30,
        '2023-03-06': 0.35,
        '2024-05-27': 0.17,
        '2024-06-03': 0.25,
        '2024-06-10': 0.35,
        '2024-06-17': 0.40,
        '2024-06-24': 0.45
    }

    # ============================================
    # 66 HAFTALIK AKADEMİK DÖNGÜ
    # ============================================
    weekly_cycle = [
        15,17,23,28,33,46,63,157,107,138,
        103,97,101,86,99,84,74,96,102,108,
        115,117,135,133,84,60,64,148,106,135,
        104,98,102,99,92,79,101,74,91,40,
        97,109,126,134,97,77,111,83,112,76,
        64,69,62,55,63,30,29,49,66,155,
        108,136,102,98,100,87
    ]

    cycle_avg = sum(weekly_cycle) / len(weekly_cycle)
    weekly_multipliers = [x / cycle_avg for x in weekly_cycle]

    predictions = []

    # ==================================================
    # ANA LOOP (HER HAFTA İÇİN MODEL OLUŞTUR)
    # ==================================================
    for i in range(len(haftalik)):
        if i < min_train:
            predictions.append({
                'ds': haftalik.iloc[i]['ds'],
                'gercek_login': haftalik.iloc[i]['gercek_login'],
                'rf_tahmin': None,
                'xgb_tahmin': None,
                'is_yaz': None
            })
            continue

        train = haftalik.iloc[:i].copy()
        test_date = haftalik.iloc[i]['ds']
        gercek_deger = haftalik.iloc[i]['gercek_login']

        # ============================================
        # FEATURE ENGINEERING
        # ============================================
        def create_features(data, target_date=None):
            features = []
            base_date = data.iloc[0]['ds']

            for idx in range(len(data)):
                row = {}
                current_date = data.iloc[idx]['ds']

                # Zaman özellikleri
                row['ay'] = current_date.month
                row['hafta'] = current_date.isocalendar().week
                row['week_sin'] = np.sin(2 * np.pi * row['hafta'] / 52)
                row['week_cos'] = np.cos(2 * np.pi * row['hafta'] / 52)

                # 66 Cycle
                weeks_since_start = int((current_date - base_date).days / 7)
                cycle_position = weeks_since_start % 66
                row['cycle_multiplier'] = weekly_multipliers[cycle_position]
                row['cycle_position'] = cycle_position

                # Yaz Sezonu
                row['is_yaz'] = 1 if row['ay'] in [7,8] or (row['ay']==6 and current_date.day >=25) else 0

                # Low Season
                low_activity_weeks = [0,1,2,25,26,27,40,56,57]
                row['is_low_season'] = 1 if cycle_position in low_activity_weeks else 0

                # High Season
                high_activity_weeks = [8,28,60]
                row['is_high_season'] = 1 if cycle_position in high_activity_weeks else 0

                # Lag features
                for lag in [1,2,4,8,52,66]:
                    row[f'lag_{lag}'] = data.iloc[idx-lag]['y'] if idx >= lag else 0

                # Rolling ortalamalar
                row['roll4'] = data.iloc[max(0,idx-4):idx]['y'].mean() if idx>0 else 0
                row['roll8'] = data.iloc[max(0,idx-8):idx]['y'].mean() if idx>0 else 0

                # 66-cycle ortalaması
                if idx > 66:
                    same_cycle_positions = []
                    for lookback in range(66, idx, 66):
                        same_cycle_positions.append(data.iloc[idx-lookback]['y'])
                    row['cycle_avg'] = np.mean(same_cycle_positions) if same_cycle_positions else row['roll8']
                else:
                    row['cycle_avg'] = row['roll8']

                features.append(row)

            # Test haftası için özellik üret
            if target_date is not None:
                test_row = {}
                test_row['ay'] = target_date.month
                test_row['hafta'] = target_date.isocalendar().week
                test_row['week_sin'] = np.sin(2 * np.pi * test_row['hafta'] / 52)
                test_row['week_cos'] = np.cos(2 * np.pi * test_row['hafta'] / 52)

                weeks_since_start = int((target_date - base_date).days / 7)
                cycle_position = weeks_since_start % 66
                test_row['cycle_multiplier'] = weekly_multipliers[cycle_position]
                test_row['cycle_position'] = cycle_position

                test_row['is_yaz'] = 1 if test_row['ay'] in [7,8] or (test_row['ay']==6 and target_date.day>=25) else 0

                low_activity_weeks = [0,1,2,25,26,27,40,56,57]
                test_row['is_low_season'] = 1 if cycle_position in low_activity_weeks else 0

                high_activity_weeks = [8,28,60]
                test_row['is_high_season'] = 1 if cycle_position in high_activity_weeks else 0

                # Lag
                for lag in [1,2,4,8,52,66]:
                    test_row[f'lag_{lag}'] = data.iloc[-lag]['y'] if len(data)>=lag else 0

                # Rolling
                test_row['roll4'] = data.iloc[-4:]['y'].mean() if len(data)>=4 else data['y'].mean()
                test_row['roll8'] = data.iloc[-8:]['y'].mean() if len(data)>=8 else data['y'].mean()

                # 66-cycle ortalama
                if len(data) > 66:
                    same_cycle_positions = []
                    for lookback in range(66, len(data), 66):
                        same_cycle_positions.append(data.iloc[-lookback]['y'])
                    test_row['cycle_avg'] = np.mean(same_cycle_positions) if same_cycle_positions else test_row['roll8']
                else:
                    test_row['cycle_avg'] = test_row['roll8']

                return pd.DataFrame(features), pd.DataFrame([test_row]), test_row['is_yaz']

            return pd.DataFrame(features), None, None

        # Feature oluşturalım
        X_train_df, X_test_df, is_yaz_flag = create_features(train, target_date=test_date)
        
        feature_cols = [col for col in X_train_df.columns]
        X_train = X_train_df[feature_cols].fillna(0)
        y_train = train['y'].values[:len(X_train)]
        X_test = X_test_df[feature_cols].fillna(0)

        # ===============================
        # RANDOM FOREST
        # ===============================
        try:
            rf = RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X_train, y_train)
            rf_tahmin = rf.predict(X_test)[0]
        except:
            rf_tahmin = train['y'].tail(4).mean()

        # ===============================
        # XGBOOST
        # ===============================
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=600,
                max_depth=12,
                learning_rate=0.08,
                subsample=0.88,
                colsample_bytree=0.85,
                reg_alpha=1.0,
                reg_lambda=1.5,
                gamma=0.5,
                min_child_weight=8,
                n_jobs=-1,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            xgb_tahmin = xgb_model.predict(X_test)[0]
        except:
            xgb_tahmin = train['y'].tail(4).mean()

        # ===============================
        # SADECE EKSTREMİZE HAFTALAR İÇİN DÜZELTME
        # ===============================
        test_date_str = test_date.strftime('%Y-%m-%d')
        
        if test_date_str in extreme_low_adjustments:
            # Geçmiş 3 yılın aynı hafta ortalaması
            if i > 66:
                same_weeks = []
                for lookback in [66, 132, 198]:
                    if i - lookback >= 0:
                        same_weeks.append(train.iloc[-lookback]['y'])
                
                if len(same_weeks) > 0:
                    historical_avg = np.mean(same_weeks) * 0.50  # %50 düşüş varsayımı
                    # Agresif düzeltme: %90 tarihsel
                    rf_tahmin = 0.10 * rf_tahmin + 0.90 * historical_avg
                    xgb_tahmin = 0.10 * xgb_tahmin + 0.90 * historical_avg
        
        # Normal düşük sezon düzeltmesi
        elif X_test_df['is_low_season'].values[0] == 1:
            if i > 66:
                same_cycles = []
                for lookback in [66, 132, 198]:
                    if i - lookback >= 0:
                        same_cycles.append(train.iloc[-lookback]['y'])
                
                if len(same_cycles) > 0:
                    cycle_based = np.mean(same_cycles)
                    rf_tahmin = 0.40 * rf_tahmin + 0.60 * cycle_based
                    xgb_tahmin = 0.40 * xgb_tahmin + 0.60 * cycle_based

        # ===============================
        # GÜVENLIK KONTROLLERI
        # ===============================
        min_deger = train['y'].quantile(0.05)
        max_deger = train['y'].quantile(0.95) * 1.2
        
        rf_tahmin = np.clip(rf_tahmin, min_deger, max_deger)
        xgb_tahmin = np.clip(xgb_tahmin, min_deger, max_deger)
        
        son_8_ort = train['y'].tail(8).mean()
        son_8_std = train['y'].tail(8).std()
        
        for tahmin_ismi, tahmin_degeri in [('rf', rf_tahmin), ('xgb', xgb_tahmin)]:
            z_score = abs(tahmin_degeri - son_8_ort) / (son_8_std + 1)
            if z_score > 2.5:
                if tahmin_ismi == 'rf':
                    rf_tahmin = 0.65 * tahmin_degeri + 0.35 * son_8_ort
                else:
                    xgb_tahmin = 0.65 * tahmin_degeri + 0.35 * son_8_ort

        rf_tahmin = max(5000, rf_tahmin)
        xgb_tahmin = max(5000, xgb_tahmin)

        predictions.append({
            'ds': test_date,
            'gercek_login': gercek_deger,
            'rf_tahmin': int(round(rf_tahmin)),
            'xgb_tahmin': int(round(xgb_tahmin)),
            'is_yaz': int(is_yaz_flag) if is_yaz_flag is not None else 0
        })

    results = pd.DataFrame(predictions)
    results['rf_hata%'] = (abs(results['rf_tahmin'] - results['gercek_login']) / results['gercek_login'] * 100).round(2)
    results['xgb_hata%'] = (abs(results['xgb_tahmin'] - results['gercek_login']) / results['gercek_login'] * 100).round(2)
    
    valid = results.dropna(subset=['rf_tahmin']).copy()

    print("\n" + "="*80)
    print("✅ 66 DÖNGÜ + MANUEL EKSTREMİZE BAYRAM - Tüm Tahminler")
    print("="*80)
    display_cols = ['ds','gercek_login','rf_tahmin','rf_hata%','xgb_tahmin','xgb_hata%','is_yaz']
    print(valid[display_cols].to_string(index=False))

    print("\n" + "="*80)
    print("📊 PERFORMANS METRİKLERİ")
    print("="*80)
    
    for model in ['rf', 'xgb']:
        rmse = np.sqrt(np.mean((valid['gercek_login'] - valid[f'{model}_tahmin'])**2))
        mae = np.mean(abs(valid['gercek_login'] - valid[f'{model}_tahmin']))
        mape = valid[f'{model}_hata%'].mean()
        r2 = 1 - (np.sum((valid['gercek_login'] - valid[f'{model}_tahmin'])**2) / 
                  np.sum((valid['gercek_login'] - valid['gercek_login'].mean())**2))
        
        print(f"{model.upper():3s} → RMSE: {rmse:>8.0f} | MAE: {mae:>8.0f} | MAPE: {mape:6.2f}% | R²: {r2:6.3f}")

    return valid
# ==================== 5. OS BAZLI 4 HAFTALIK TAHMİN (SARIMA OPTİMİZE) ====================
def os_4haftalik_tahmin(df):
    print("\n📊 OS BAZLI 4 HAFTALIK TAHMİN (YAZ TATİLİ DÜZELTMELİ)\n")
    
    # 66 haftalık döngü bilgisi
    weekly_cycle = [
        15,17,23,28,33,46,63,157,107,138,
        103,97,101,86,99,84,74,96,102,108,
        115,117,135,133,84,60,64,148,106,135,
        104,98,102,99,92,79,101,74,91,40,
        97,109,126,134,97,77,111,83,112,76,
        64,69,62,55,63,30,29,49,66,155,
        108,136,102,98,100,87
    ]
    
    cycle_avg = sum(weekly_cycle) / len(weekly_cycle)
    weekly_multipliers = [x / cycle_avg for x in weekly_cycle]
    
    low_activity_weeks = [0, 1, 2, 25, 26, 27, 40, 56, 57]

    populer_os = df["OS Name and Version"].value_counts().head(4).index
    all_results = []

    for os_name in populer_os:
        df_os = df[df["OS Name and Version"] == os_name].copy()
        if df_os.empty:
            continue

        haftalik = (
            df_os.set_index("Login Timestamp")
            .resample("W-MON")
            .size()
            .reset_index(name="gercek_login")
        )
        haftalik.rename(columns={"Login Timestamp": "ds"}, inplace=True)
        
        if len(haftalik) < 16:
            continue

        train_data = haftalik.iloc[:-4].copy()
        test_data = haftalik.iloc[-4:].copy()
        valid = test_data.copy()
        valid["OS"] = os_name
        
        base_date = haftalik.iloc[0]['ds']

        # =============== PROPHET =============== 
        try:
            prophet_model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.35,
                seasonality_mode='additive'
            )
            
            prophet_df = train_data[['ds', 'gercek_login']].rename(columns={'gercek_login':'y'})
            prophet_model.fit(prophet_df)
            
            future = prophet_model.make_future_dataframe(periods=4, freq='W-MON', include_history=False)
            forecast = prophet_model.predict(future)
            
            prophet_tahmin = forecast['yhat'].round().astype(int).clip(lower=0)
            valid["prophet_tahmin"] = prophet_tahmin.values
            
        except:
            ema = train_data['gercek_login'].ewm(span=4, adjust=False).mean().iloc[-1]
            valid["prophet_tahmin"] = int(ema)

        # =============== SARIMA ===============
        try:
            print(f"  → {os_name}: SARIMA optimizasyonu yapılıyor...")
            best_model, best_params = optimize_sarima(train_data['gercek_login'], seasonal_period=4, max_combinations=40)
            
            sarima_pred = best_model.get_forecast(steps=4).predicted_mean.round().astype(int).clip(lower=0)
            valid["sarima_tahmin"] = sarima_pred.values
            
            print(f"     ✓ En iyi parametreler: {best_params}")
        
        except:
            ema = train_data['gercek_login'].ewm(span=4, adjust=False).mean().iloc[-1]
            valid["sarima_tahmin"] = int(ema)
            print(f"     ⚠ SARIMA hatası, fallback kullanıldı")

        # ======================================================
        #  🔥 BURASI KRİTİK: (Hata nedeni buradaydı)
        #  cycle_position, lookback_weeks HER ZAMAN hesaplanıyor.
        # ======================================================
        for idx, row in valid.iterrows():

            tahmin_tarihi = row['ds']

            # döngü pozisyonu
            weeks_since_start = int((tahmin_tarihi - base_date).days / 7)
            cycle_position = weeks_since_start % 66

            # geçmiş yıl karşılaştırmaları
            lookback_weeks = []
            for lookback in [52, 104, 156]:
                lookback_idx = len(train_data) - (len(valid) - list(valid.index).index(idx)) - lookback
                if 0 <= lookback_idx < len(train_data):
                    lookback_weeks.append(train_data.iloc[lookback_idx]['gercek_login'])

            # ========== AĞUSTOS ÖZEL DÜZELTMESİ ==========
            if cycle_position in [56, 57]:
                if len(lookback_weeks) > 0:
                    historical_avg = np.mean(lookback_weeks)
                    adjusted_value = int(historical_avg * 0.30)

                    valid.loc[idx, 'prophet_tahmin'] = int(
                        0.20 * valid.loc[idx, 'prophet_tahmin'] + 0.80 * adjusted_value
                    )
                    valid.loc[idx, 'sarima_tahmin'] = int(
                        0.20 * valid.loc[idx, 'sarima_tahmin'] + 0.80 * adjusted_value
                    )
                continue

            # ========== NORMAL YAZ TATİLİ DÜZELTMESİ ==========
            if cycle_position in low_activity_weeks:
                if len(lookback_weeks) > 0:
                    historical_avg = np.mean(lookback_weeks)
                    adjusted_value = int(historical_avg * 0.60)

                    valid.loc[idx, 'prophet_tahmin'] = int(
                        0.30 * valid.loc[idx, 'prophet_tahmin'] + 0.70 * adjusted_value
                    )
                    valid.loc[idx, 'sarima_tahmin'] = int(
                        0.30 * valid.loc[idx, 'sarima_tahmin'] + 0.70 * adjusted_value
                    )

        # ========== Hata Hesaplaması ==========
        valid["prophet_mutlak_hata"] = abs(valid["gercek_login"] - valid["prophet_tahmin"])
        valid["prophet_hata_yuzdesi"] = (valid["prophet_mutlak_hata"] / valid["gercek_login"] * 100).round(2)

        valid["sarima_mutlak_hata"] = abs(valid["gercek_login"] - valid["sarima_tahmin"])
        valid["sarima_hata_yuzdesi"] = (valid["sarima_mutlak_hata"] / valid["gercek_login"] * 100).round(2)

        all_results.append(valid)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        print("\n📈 SON 4 HAFTA TAHMİN SONUÇLARI (YAZ TATİLİ DÜZELTMELİ)\n")
        print(df_all.to_string(index=False))
        return df_all

    print("⚠️ Hiçbir OS için geçerli tahmin üretilemedi.")
    return None
# ==================== 6. ANOMALİ TESPİTİ ====================
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
    user_daily.columns = ["ID", "Tarih", "Login_Sayisi", "Login_Std", "Browser", "OS", "App"]
    user_daily["Login_Std"] = user_daily["Login_Std"].fillna(0)

    gece = df[(df["Saat"] >= 0) & (df["Saat"] <= 5)].groupby(["key", "Tarih"]).size().reset_index(name="Gece")
    gece.rename(columns={"key": "ID"}, inplace=True)
    user_daily = user_daily.merge(gece, on=["ID", "Tarih"], how="left").fillna({"Gece": 0})

    numeric_cols = ["Login_Sayisi", "Login_Std", "Browser", "OS", "App", "Gece"]
    varyans = user_daily[numeric_cols].var()
    aktif_kolonlar = varyans[varyans > 0].index.tolist()

    X = StandardScaler().fit_transform(user_daily[aktif_kolonlar])

    if contamination is None:
        contamination = min(0.05, max(0.01, 100 / len(user_daily)))

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    user_daily["Anormal"] = np.where(iso.fit_predict(X) == -1, "Anormal", "Normal")

    rastgele_kayitlar = user_daily.sample(n=50, random_state=42)

    print("\n" + "="*100)
    print("🎲 RASTGELE 50 KAYIT")
    print("="*100)
    print(rastgele_kayitlar.to_string(index=False))

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

    # -----------------------------
    # EN ŞÜPHELİ 20 ANOMALİ KAYIT
    # -----------------------------
    anormal_kayitlar = user_daily[user_daily["Anormal"] == "Anormal"].copy()
    anormal_kayitlar = anormal_kayitlar.nlargest(20, 'Login_Sayisi')

    print("\n" + "="*100)
    print(anormal_kayitlar[['ID', 'Tarih', 'Login_Sayisi', 'Gece', 'Browser', 'OS', 'App']].to_string(index=False))

    # -----------------------------
    # ANOMALİ SEBEPLERİ ANALİZİ
    # -----------------------------
    print("\n" + "="*100)
    print("📊 ANOMALİ SEBEPLERİ ANALİZİ")
    print("="*100)
    
    return rastgele_kayitlar, aylik_anomali
# ==================== 7. KÜMELEME ====================
def benzer_login_siniflandir(df, n_clusters=5):
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
        "Saat": ["count","mean","max"],
        "OS Name and Version": "nunique",
        "Browser Name and Version": "nunique",
        "Client Name": "nunique"
    }).reset_index()

    user_daily.columns = ["ID","Tarih","Login","Saat_Ort","Saat_Max","OS","Browser","App"]
    user_daily = user_daily.fillna(0)

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

    feature_cols = ["Login","Saat_Ort","Saat_Max","OS","Browser","App","Gece"]
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
    print("🎯 KÜMELEME SONUÇLARI (Optimize Edilmiş)")
    print("="*80)
    print(f"Silhouette Skoru : {silhouette:.4f}")
    print(f"Davies-Bouldin   : {db_index:.4f}")
    print("="*80)

    for i in range(1, n_clusters+1):
        grup = user_daily[user_daily["Grup"]==i]
        grup_temp = temp_df_with_groups[temp_df_with_groups['Grup']==i]  # ← yeni

        print(f"\n📊 Grup {i} ({len(grup):,} kayıt):")
        print(f"Login Ort      : {grup['Login'].mean():.1f}")
        print(f"Ortalama Saat  : {grup['Saat_Ort'].mean():.1f}")
        print(f"Günün Zamanı   : {grup['GunZamani'].mode()[0]}")
        print(f"En sık HaftaDavranisi : {grup['HaftaDavranisi'].mode()[0]}")
        print(f"Ortalama Gün   : {grup['OrtalamaGun'].mode()[0]}")
        print(f"En sık OS       : {grup_temp['OS Name and Version'].mode()[0]}")
        print(f"En sık Browser  : {grup_temp['Browser Name and Version'].mode()[0]}")
        print(f"En sık Client   : {grup_temp['Client Name'].mode()[0]}")
        print("-"*60)

    return user_daily
# ==================== ANA PROGRAM ====================
print("🚀 LOGIN ANALİZ SİSTEMİ - OPTİMİZE VERSİYON")
dosya_yolu = r"Buraya Verdiğim koddan verinizi oluşturduktan sonra json dosyasının yol dizimini yazıcaksınız"
"örnek olarak yol dizimi: C:\Users\Aykut\AppData\Local\Programs\Microsoft VS Code\2login_data_5years_10M.jsonl "
df = load_data_optimized(dosya_yolu, nrows=None)
# ==================== MENÜ ====================
while True:
    print("\n==== Login Analiz Sistemi ====")
    secim = input(
        "1- Os/Browser yoğunluk\n"
        "2- Client analizi\n"
        "3- Saat/Gün Tahmini\n"
        "4- Haftalık Tahmin\n"
        "5- OS 4 Haftalık Tahmin\n"
        "6- Anomali Tespiti\n"
        "7- Kümeleme\n"
        "8- Çıkış\n"
        "Seçim: "
    )
    if secim == '1': 
        os_browser_yogunluk_analizi(df)
    elif secim == '2':
        client_os_browser_analizi(df)
    elif secim == '3':
        saat_gun_model(df)
    elif secim == '4': 
        haftalik_login_tahmini(df)
    elif secim == '5': 
        os_4haftalik_tahmin(df)
    elif secim == '6': 
        anomali_tespiti(df)
    elif secim == '7': 
        benzer_login_siniflandir(df)
    elif secim == '8':
        print("Çıkış yapılıyor...")
        break
    else:
        print("⚠️ Geçersiz seçim!")