import pandas as pd 
import numpy as np  
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, davies_bouldin_score
from prophet import Prophet
from pykalman import KalmanFilter
import logging
import warnings
import json
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 🔇 Gereksiz logları tamamen kapat
logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').disabled = True  # 🔥 Bu satır en etkili olanı
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== VERİ YÜKLEME ====================
print("📊 Veri yükleniyor...")
dosya_yolu = r"C:\Users\Aykut\AppData\Local\Programs\Microsoft VS Code\NEW2_login_data_realistic.jsonl"

data_list = []
with open(dosya_yolu, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                continue

df = pd.DataFrame(data_list).rename(columns={
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

gun_dict = {0:'Pazartesi', 1:'Salı', 2:'Çarşamba', 3:'Perşembe', 4:'Cuma', 5:'Cumartesi', 6:'Pazar'}

# Label Encoding
encoders = {}
for col in ["OS Name and Version", "Browser Name and Version", "Client Name"]:
    encoders[col] = LabelEncoder()
    df[col+'_enc'] = encoders[col].fit_transform(df[col].astype(str))

print(f"✓ {len(df):,} kayıt yüklendi | {df['Login Timestamp'].min()} → {df['Login Timestamp'].max()}\n")

# ==================== YARDIMCI FONKSİYONLAR ====================
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return rmse, mape

# ==================== 1. OS/BROWSER YOĞUNLUK ANALİZİ ====================
def os_browser_yogunluk_analizi(df):
    print("\n" + "="*80)
    print("📊 OS/BROWSER BAZLI LOGİN YOĞUNLUK ANALİZİ")
    print("="*80)
    
    # Tek fonksiyonla her iki analiz
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
    
    # Browser dağılımı (sadece top 5 client)
    df_temp = df.copy()
    df_temp['Browser_Base'] = df_temp['Browser Name and Version'].str.split().str[0]
    
    print("\n🌐 TOP 5 CLIENT İÇİN BROWSER DAĞILIMI")
    for client in client_stats.head(5).index:
        df_client = df_temp[df_temp['Client Name'] == client]
        browser_dist = df_client['Browser_Base'].value_counts().head(3)
        
        print(f"\n🔹 {client} ({len(df_client):,} login):")
        for browser, count in browser_dist.items():
            print(f"   • {browser}: {count:,} (%{count/len(df_client)*100:.1f})")

# ==================== 3. SAAT/GÜN TAHMİNİ (GB + LightGBM) ====================
def saat_gun_model(df, encoders, gun_dict):
    print("\n⏰ SAAT/GÜN BAZLI LOGİN TAHMİNİ (GB + LightGBM)")
    print(f"📊 Toplam veri: {len(df):,} satır")
    
    df = df.copy()
    df['ay'] = df['Login Timestamp'].dt.month
    df['hafta_sonu'] = (df['gun'] >= 5).astype(int)
    df['mevsim'] = df['ay'].map({12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})

    cols = ['gun', 'saat', 'ay', 'hafta_sonu', 'mevsim']
    login_counts = df.groupby(cols).size().reset_index(name='Login_Sayisi')
    login_counts = login_counts.sort_values(['ay','gun','saat']).reset_index(drop=True)
    
    print(f"✅ {len(login_counts):,} kombinasyon bulundu")

    # Train/Test split
    split_idx = int(len(login_counts) * 0.85)
    train_data = login_counts.iloc[:split_idx].copy()
    test_data = login_counts.iloc[split_idx:].copy()
    
    print(f"🎯 Train: {len(train_data)}, Test: {len(test_data)}")

    # Feature engineering
    for data in [train_data]:
        data['lag_1'] = data['Login_Sayisi'].shift(1).fillna(0)
        data['lag_24'] = data['Login_Sayisi'].shift(24).fillna(0)
        data['lag_168'] = data['Login_Sayisi'].shift(168).fillna(0)
        data['rolling_mean_3'] = data['Login_Sayisi'].rolling(3, min_periods=1).mean()
        data['rolling_mean_24'] = data['Login_Sayisi'].rolling(24, min_periods=1).mean()
        data['rolling_std_24'] = data['Login_Sayisi'].rolling(24, min_periods=1).std().fillna(0)
        
        # Siklik encoding
        data['saat_sin'] = np.sin(2 * np.pi * data['saat'] / 24)
        data['saat_cos'] = np.cos(2 * np.pi * data['saat'] / 24)
        data['gun_sin'] = np.sin(2 * np.pi * data['gun'] / 7)
        data['gun_cos'] = np.cos(2 * np.pi * data['gun'] / 7)
    
    # Train'den öğrenilen istatistikler
    saat_avg = train_data.groupby('saat')['Login_Sayisi'].mean().to_dict()
    gun_avg = train_data.groupby('gun')['Login_Sayisi'].mean().to_dict()
    hafta_sonu_avg = train_data.groupby('hafta_sonu')['Login_Sayisi'].mean().to_dict()
    
    for data in [train_data, test_data]:
        data['saat_avg'] = data['saat'].map(saat_avg).fillna(train_data['Login_Sayisi'].mean())
        data['gun_avg'] = data['gun'].map(gun_avg).fillna(train_data['Login_Sayisi'].mean())
        data['hafta_sonu_avg'] = data['hafta_sonu'].map(hafta_sonu_avg).fillna(train_data['Login_Sayisi'].mean())

    # Test için lag hesaplama
    combined = pd.concat([train_data[['Login_Sayisi']], test_data[['Login_Sayisi']]], ignore_index=True)
    test_data['lag_1'] = combined['Login_Sayisi'].shift(1).iloc[len(train_data):].fillna(0).values
    test_data['lag_24'] = combined['Login_Sayisi'].shift(24).iloc[len(train_data):].fillna(0).values
    test_data['lag_168'] = combined['Login_Sayisi'].shift(168).iloc[len(train_data):].fillna(0).values
    test_data['rolling_mean_3'] = combined['Login_Sayisi'].rolling(3, min_periods=1).mean().iloc[len(train_data):].values
    test_data['rolling_mean_24'] = combined['Login_Sayisi'].rolling(24, min_periods=1).mean().iloc[len(train_data):].values
    test_data['rolling_std_24'] = combined['Login_Sayisi'].rolling(24, min_periods=1).std().fillna(0).iloc[len(train_data):].values
    
    test_data['saat_sin'] = np.sin(2 * np.pi * test_data['saat'] / 24)
    test_data['saat_cos'] = np.cos(2 * np.pi * test_data['saat'] / 24)
    test_data['gun_sin'] = np.sin(2 * np.pi * test_data['gun'] / 7)
    test_data['gun_cos'] = np.cos(2 * np.pi * test_data['gun'] / 7)

    # Model eğitimi
    feature_cols = cols + ['lag_1','lag_24','lag_168','rolling_mean_3','rolling_mean_24','rolling_std_24',
                           'saat_avg','gun_avg','hafta_sonu_avg','saat_sin','saat_cos','gun_sin','gun_cos']
    
    X_train, y_train = train_data[feature_cols], train_data['Login_Sayisi']
    X_test, y_test = test_data[feature_cols], test_data['Login_Sayisi']

    # Gradient Boosting
    print("🚀 GB modeli eğitiliyor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.2, max_depth=6,
        min_samples_split=10, min_samples_leaf=5, subsample=0.8,
        max_features='sqrt', random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = np.clip(gb_model.predict(X_test), 0, None)
    print("✅ GB tamamlandı")

    # LightGBM
    print("🚀 LightGBM modeli eğitiliyor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.2, max_depth=-1,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = np.clip(lgb_model.predict(X_test), 0, None)
    print("✅ LightGBM tamamlandı")

    # Test sonuçları
    test_results = test_data.copy()
    test_results['gercek_login'] = y_test.values
    test_results['gb_tahmin'] = np.round(gb_pred).astype(int)
    test_results['lgb_tahmin'] = np.round(lgb_pred).astype(int)
    
    test_results['gb_mutlak_hata'] = abs(test_results['gb_tahmin'] - test_results['gercek_login'])
    test_results['lgb_mutlak_hata'] = abs(test_results['lgb_tahmin'] - test_results['gercek_login'])
    
    test_results['gb_hata%'] = np.where(
        test_results['gercek_login'] > 0,
        (test_results['gb_mutlak_hata'] / test_results['gercek_login'] * 100).round(2), 0
    )
    test_results['lgb_hata%'] = np.where(
        test_results['gercek_login'] > 0,
        (test_results['lgb_mutlak_hata'] / test_results['gercek_login'] * 100).round(2), 0
    )

    test_results['gun_ismi'] = test_results['gun'].map(gun_dict)

    # EN YOĞUN 10 GÜN+SAAT
    print("\n" + "="*80)
    print("🔥 EN YOĞUN 10 GÜN+SAAT")
    print("="*80)
    print(test_results[['gun_ismi','saat','gercek_login','gb_tahmin','lgb_tahmin',
                        'gb_mutlak_hata','lgb_mutlak_hata','gb_hata%','lgb_hata%']]
          .sort_values('gercek_login', ascending=False).head(10).to_string(index=False))

    # Metrikler (calculate_metrics fonksiyonu kullanılacak)
    gb_rmse, gb_mape = calculate_metrics(test_results['gercek_login'], test_results['gb_tahmin'])
    lgb_rmse, lgb_mape = calculate_metrics(test_results['gercek_login'], test_results['lgb_tahmin'])
    
    print("\n📊 METRİKLER:")
    print(f"GB       → RMSE: {gb_rmse:,.2f} | MAPE: {gb_mape:.2f}%")
    print(f"LightGBM → RMSE: {lgb_rmse:,.2f} | MAPE: {lgb_mape:.2f}%")

    return gb_model, lgb_model, test_results

# ==================== 4. HAFTALIK TAHMİN ====================
def haftalik_login_tahmini(df):
    df = df.dropna(subset=['Login Timestamp']).copy()
    haftalik = df.set_index('Login Timestamp').resample('W-MON')['Login Successful'].sum().reset_index()
    haftalik.rename(columns={'Login Successful':'y', 'Login Timestamp':'ds'}, inplace=True)
    haftalik['ay'] = haftalik['ds'].dt.month
    haftalik['is_yil_sonu'] = haftalik['ay'].isin([12,1]).astype(int)
    haftalik['is_yaz'] = haftalik['ay'].isin([6,7,8]).astype(int)

    # Outlier temizleme
    Q1, Q3 = haftalik['y'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    haftalik = haftalik[(haftalik['y'] >= Q1-1.5*IQR) & (haftalik['y'] <= Q3+1.5*IQR)].reset_index(drop=True)
    
    if len(haftalik) < 8:
        print("❌ Yetersiz veri (min 8 hafta)")
        return None

    haftalik['gercek_login'] = haftalik['y'].astype(int)
    min_train = int(len(haftalik) * 0.5)
    prophet_pred, kalman_pred = [], []

    for i in range(len(haftalik)):
        if i < min_train:
            prophet_pred.append(None)
            kalman_pred.append(None)
            continue

        train = haftalik.iloc[:i].copy()

        # Prophet
        try:
            model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False,
                          seasonality_mode='additive', changepoint_prior_scale=0.3)
            model.add_regressor('is_yil_sonu', prior_scale=0.5)
            model.add_regressor('is_yaz', prior_scale=0.5)
            model.fit(train[['ds','y','is_yil_sonu','is_yaz']])
            forecast = model.predict(haftalik.iloc[[i]][['ds','is_yil_sonu','is_yaz']])
            prophet_pred.append(max(0, int(round(forecast['yhat'].values[0]))))
        except:
            prophet_pred.append(None)

        # Kalman
        window = min(4, i)
        train_vals = haftalik.iloc[i-window:i]['gercek_login'].values
        kf = KalmanFilter(transition_matrices=[[1,1],[0,1]], observation_matrices=[[1,0]],
                         initial_state_mean=[train_vals[0],0], transition_covariance=np.eye(2)*0.001,
                         observation_covariance=1.5)
        state_means, _ = kf.filter(train_vals)
        kalman_pred.append(max(0, int(round((kf.transition_matrices @ state_means[-1])[0]))))

    haftalik['prophet_tahmin'] = prophet_pred
    haftalik['kalman_tahmin'] = kalman_pred
    valid = haftalik.dropna(subset=['prophet_tahmin','kalman_tahmin']).copy()

    for model in ['prophet', 'kalman']:
        valid[f'{model}_hata%'] = (abs(valid[f'{model}_tahmin']-valid['gercek_login'])/valid['gercek_login']*100).round(2)

    print("\n" + "="*80)
    print("📈 HAFTALIK TAHMİN SONUÇLARI")
    print("="*80)
    print(valid[['ds','gercek_login','prophet_tahmin','kalman_tahmin','prophet_hata%','kalman_hata%']].to_string(index=False))
    
    for model in ['prophet','kalman']:
        rmse = np.sqrt(np.mean((valid['gercek_login']-valid[f'{model}_tahmin'])**2))
        mape = valid[f'{model}_hata%'].mean()
        print(f"\n{model.upper():10s} → RMSE: {rmse:,.2f} | MAPE: {mape:.2f}%")

    return valid

# ==================== 5. OS BAZLI 4 HAFTALIK TAHMİN ====================
def os_4haftalik_tahmin(df):
    print("\n📊 OS BAZLI 4 HAFTALIK TAHMİN (DETAYLI TABLO – XGBoost + CatBoost + Kalman)")

    populer_os = df["OS Name and Version"].value_counts().head(4).index
    all_results = []

    for os_name in populer_os:
        df_os = df[df["OS Name and Version"] == os_name].copy()
        haftalik = (
            df_os.set_index("Login Timestamp")
            .resample("W-MON")
            .size()
            .reset_index(name="y")
        )
        haftalik.rename(columns={"Login Timestamp": "ds"}, inplace=True)

        if len(haftalik) < 8:
            continue

        # Outlier temizleme
        Q1, Q3 = haftalik["y"].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        haftalik = haftalik[
            (haftalik["y"] >= Q1 - 1.5 * IQR) & (haftalik["y"] <= Q3 + 1.5 * IQR)
        ].reset_index(drop=True)

        haftalik["gercek"] = haftalik["y"].astype(int)
        haftalik["hafta_no"] = haftalik["ds"].dt.isocalendar().week
        min_train = int(len(haftalik) * 0.5)

        kalman_pred, xgb_pred, cat_pred = [], [], []

        for i in range(len(haftalik)):
            if i < min_train:
                kalman_pred.append(None)
                xgb_pred.append(None)
                cat_pred.append(None)
                continue

            train = haftalik.iloc[:i].copy()  # ⚡️copy() eklendi uyarı gider
            train["hafta_sonu"] = (train["ds"].dt.dayofweek >= 5).astype(int)
            train["ay"] = train["ds"].dt.month

            # ---------------- Kalman Filter ----------------
            try:
                window = min(4, i)
                kf = KalmanFilter(
                    transition_matrices=[[1, 1], [0, 1]],
                    observation_matrices=[[1, 0]],
                    initial_state_mean=[train.iloc[-window]["gercek"], 0],
                    transition_covariance=np.eye(2) * 0.01,
                    observation_covariance=1.5
                )
                state_means, _ = kf.filter(train.iloc[-window:]["gercek"].values)
                kalman_pred.append(max(0, int(round((kf.transition_matrices @ state_means[-1])[0]))))
            except:
                kalman_pred.append(None)

            # ---------------- XGBoost ----------------
            try:
                if i >= 3:
                    X_train = pd.DataFrame({
                        'prev1': train['gercek'].shift(1),
                        'prev2': train['gercek'].shift(2),
                        'prev3': train['gercek'].shift(3),
                        'hafta_no': train['hafta_no']
                    }).dropna()
                    y_train = train['gercek'].iloc[-len(X_train):]

                    model_xgb = XGBRegressor(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=42
                    )
                    model_xgb.fit(X_train, y_train)

                    X_pred = pd.DataFrame({
                        'prev1': [train['gercek'].iloc[-1]],
                        'prev2': [train['gercek'].iloc[-2]],
                        'prev3': [train['gercek'].iloc[-3]],
                        'hafta_no': [train['hafta_no'].iloc[-1]]
                    })
                    pred_val = model_xgb.predict(X_pred)[0]
                    xgb_pred.append(max(0, int(round(pred_val))))
                else:
                    xgb_pred.append(None)
            except:
                xgb_pred.append(None)

        haftalik['kalman_tahmin'] = kalman_pred
        haftalik['xgb_tahmin'] = xgb_pred

        valid = haftalik.dropna(subset=["kalman_tahmin", "xgb_tahmin"]).tail(4)
        if valid.empty:
            continue

        valid["mutlak_hata_kalman"] = abs(valid["gercek"] - valid["kalman_tahmin"])
        valid["mutlak_hata_xgb"] = abs(valid["gercek"] - valid["xgb_tahmin"])

        valid["hata%_kalman"] = (valid["mutlak_hata_kalman"] / valid["gercek"] * 100).round(2)
        valid["hata%_xgb"] = (valid["mutlak_hata_xgb"] / valid["gercek"] * 100).round(2)

        valid["OS"] = os_name
        valid = valid[[
            "OS","ds","gercek","kalman_tahmin","xgb_tahmin",
            "mutlak_hata_kalman","mutlak_hata_xgb",
            "hata%_kalman","hata%_xgb"
        ]]
        all_results.append(valid)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.rename(columns={"gercek": "gercek_login"}, inplace=True)
        print("\n📈 SON 4 HAFTA TAHMİN KARŞILAŞTIRMASI (Kalman + XGBoost)\n")
        print(df_all.to_string(index=False))
        return df_all
    else:
        print("⚠️ Hiçbir OS için geçerli tahmin üretilemedi.")
        return None

# ==================== 6. ANOMALİ TESPİTİ ====================
def anomali_tespiti(df, contamination=0.05):
    # Sütun adlarını uyumlu hale getir
    df = df.rename(columns={
        "loginTime": "Login Timestamp",
        "browser": "Browser Name and Version",
        "os": "OS Name and Version",
        "clientName": "Client Name"
    })

    # Zaman ve saat bilgisi
    df["Login Timestamp"] = pd.to_datetime(df["Login Timestamp"])
    df["Tarih"] = df["Login Timestamp"].dt.date
    df["Saat"] = df["Login Timestamp"].dt.hour

    # Günlük kullanıcı davranışı
    user_daily = df.groupby(["key", "Tarih"]).agg({
        "Saat": ["count", "std"],
        "Browser Name and Version": "nunique",
        "OS Name and Version": "nunique",
        "ip": "nunique",
        "Client Name": "nunique"
    }).reset_index()
    
    user_daily.columns = ["ID", "Tarih", "Login_Sayisi", "Login_Std", "Browser", "OS", "IP", "App"]
    user_daily["Login_Std"] = user_daily["Login_Std"].fillna(0)

    # Gece login sayısı
    gece = df[(df["Saat"] >= 0) & (df["Saat"] <= 5)].groupby(["key", "Tarih"]).size().reset_index(name="Gece")
    gece.rename(columns={"key": "ID"}, inplace=True)
    user_daily = user_daily.merge(gece, on=["ID", "Tarih"], how="left").fillna({"Gece": 0})

    # Anomali modelleme
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    import numpy as np

    X = StandardScaler().fit_transform(
        user_daily[["Login_Sayisi", "Login_Std", "Browser", "OS", "IP", "App", "Gece"]].fillna(0)
    )

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    user_daily["Anormal"] = np.where(iso.fit_predict(X) == -1, "Anormal", "Normal")

    # 🔥 RASTGELE 50 KAYIT GÖSTER (Alfabetik sıralama yok!)
    sample = user_daily.sample(n=50, random_state=42)
    
    # Kodunuzun sonuna ekleyin
    print("\n" + "="*80)
    print("⚠️ SADECE ANOMALİ KAYITLAR (İlk 50)")
    print("="*80)
    anomalies = user_daily[user_daily["Anormal"] == "Anormal"].head(50)
    print(anomalies.to_string(index=False))

# ==================== 7. KÜMELEME ====================
def benzer_login_siniflandir(df, n_clusters=5):
    df["Tarih"] = df["Login Timestamp"].dt.date
    df["Saat"] = df["Login Timestamp"].dt.hour
    df["Gun"] = df["Login Timestamp"].dt.dayofweek

    user_daily = df.groupby(["key","Tarih"]).agg({
        "Saat": ["count","mean","std"],
        "OS Name and Version": "nunique",
        "Browser Name and Version": "nunique",
        "Client Name": "nunique"
    }).reset_index()
    
    user_daily.columns = ["ID","Tarih","Login","Saat_Ort","Saat_Std","OS","Browser","App"]
    user_daily = user_daily.fillna(0)
    user_daily = user_daily[user_daily["Login"] < 100]  # Outlier filtre

    X = StandardScaler().fit_transform(user_daily[["Login","Saat_Ort","Saat_Std","OS","Browser","App"]])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_daily["Grup"] = kmeans.fit_predict(X) + 1

    # Metrikler
    sample_size = min(80000, len(X))
    sample_idx = np.random.choice(len(X), sample_size, replace=False) if len(X) > sample_size else slice(None)
    silhouette = silhouette_score(X[sample_idx], kmeans.labels_[sample_idx])
    db_index = davies_bouldin_score(X, kmeans.labels_)
    
    print("\n" + "="*80)
    print("🎯 KÜMELEME SONUÇLARI")
    print("="*80)
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}\n")

    for i in range(1, n_clusters+1):
        grup = user_daily[user_daily["Grup"]==i]
        print(f"Grup {i} ({len(grup):,} kayıt): Login={grup['Login'].mean():.1f}, Saat={grup['Saat_Ort'].mean():.1f}, OS={grup['OS'].mean():.1f}")

    return user_daily

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
        saat_gun_model(df, encoders, gun_dict)
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