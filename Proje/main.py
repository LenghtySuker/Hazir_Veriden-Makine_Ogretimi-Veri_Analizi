import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from prophet import Prophet
import logging
import warnings
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
logging.getLogger('prophet').setLevel(logging.ERROR) 
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").disabled = True

# ---------------- Veri Yükleme ----------------
dosya_yolu = r"C:\Users\Aykut\.cache\kagglehub\datasets\dasgroup\rba-dataset\versions\1\rba-dataset.csv"
df = pd.read_csv(dosya_yolu, nrows=2_000_000)

# Temizlik ve dönüşümler
df["Login Successful"] = df["Login Successful"].astype(int)
df = df.fillna("Bilinmiyor")
df["Round-Trip Time [ms]"] = pd.to_numeric(df["Round-Trip Time [ms]"], errors="coerce").fillna(0)
df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')  
df['saat'] = df['Login Timestamp'].dt.hour
df['gun'] = df['Login Timestamp'].dt.dayofweek

# 🔹 1 yıllık (2020-02-01 → 2021-02-01) veriyi filtrele
df_2020 = df[(df['Login Timestamp'] >= '2020-02-01') & (df['Login Timestamp'] <= '2021-02-01')]

gun_dict = {0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe', 4: 'Cuma', 5: 'Cumartesi', 6: 'Pazar'}

# Label Encoding
from sklearn.preprocessing import LabelEncoder
label_cols = ["OS Name and Version", "Device Type", "Browser Name and Version", "Country", "Region", "City", "ASN"]
encoders = {}
for col in label_cols:
    df[col] = df[col].astype(str)
    encoders[col] = LabelEncoder()
    df[col+'_enc'] = encoders[col].fit_transform(df[col])
# ---------------- Fonksiyonlar ----------------
def os_device_model(df):
    X = df[[col+'_enc' for col in ["OS Name and Version", "Device Type"]]]  # Veri setinden OS ve Cihaz tipine ait sayısal olarak encode edilmiş sütunları alıyoruz
    y = df["Login Successful"]  # Girişin başarılı olup olmadığını (1 veya 0) hedef değişken olarak seçiyoruz

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:,1] # predict_proba:iki sınıfın (0 ve 1) olasılıklarından sadece 1 sınıfının olasılıklarını alır
    sonuc = X_test.copy()  # Test verisinin bir kopyasını alıyoruz 

    for col in ["OS Name and Version", "Device Type"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc'])  # Encode ettiğimiz sütunları tekrar orijinal isimlerine çeviriyoruz
    # inverse_transform → sayısal kodları geri dönüştürüp gerçek isimleri elde eder

    sonuc['login_olasilik'] = y_pred_prob * 100  # sonuc tablosuna “login_olasilik” adlı yeni bir sütun ekliyoruz ve tahmin edilen olasılıkları %’ye çeviriyoruz
    # yani örn. 0.87 olasılığını 87.0 olarak kaydediyoruz

    print("---- OS/Device Bazlı Login Olasılıkları ----")  
    mse = mean_squared_error(y_test, y_pred_prob)

    rmse = np.sqrt(mse)
    print(f"Model Hatası (RMSE): {rmse:.4f}") 
    print(sonuc[["OS Name and Version", "Device Type", "login_olasilik"]].head(20))

def saat_gun_model(df):

    cols = ['gun','saat'] + [col+'_enc' for col in ["OS Name and Version", "Device Type", "Browser Name and Version"]]
    login_counts = df.groupby(cols).size().reset_index(name='login_sayisi')

    # 1️⃣ NORMAL RANDOM SPLIT MODEL

    X = login_counts[cols]
    y = login_counts['login_sayisi']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=1500,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), a_min=0, a_max=None)

    sonuc = X_test.copy()
    sonuc['gercek_login'] = y_test.values
    sonuc['tahmini_login'] = np.round(y_pred).astype(int)

    for col in ["OS Name and Version", "Device Type", "Browser Name and Version"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc'].astype(int))

    sonuc['gun_ismi'] = sonuc['gun'].map(gun_dict)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("\n---- Saat/Gün + Ek Özellikler Bazlı Tahmini Login Sayısı ----")
    print(f"Model Hatası (Genel RMSE): {rmse:.2f}")

    gun_saat_bazli = (
        sonuc.groupby(['gun_ismi', 'saat'])
        .agg({'tahmini_login':'sum','gercek_login':'sum'})
        .reset_index()
        .sort_values(by='tahmini_login', ascending=False)
    )

    gun_saat_bazli['mutlak_hata'] = abs(gun_saat_bazli['tahmini_login'] - gun_saat_bazli['gercek_login'])
    gun_saat_bazli['hata_%'] = (gun_saat_bazli['mutlak_hata'] / np.maximum(gun_saat_bazli['gercek_login'], 1)) * 100

    print("\n--- EN YOĞUN 10 GÜN+SAAT KOMBİNASYONU ---")
    print(gun_saat_bazli.head(10)[['gun_ismi','saat','gercek_login','tahmini_login','mutlak_hata','hata_%']].to_string(index=False))

    sonuc_en_yogun = sonuc.groupby(['gun_ismi','saat','OS Name and Version','Device Type','Browser Name and Version']) \
                          .agg({'tahmini_login':'sum','gercek_login':'sum'}) \
                          .reset_index() \
                          .sort_values(by='tahmini_login', ascending=False)
    
    print("\n--- ÖRNEK 10 TAHMİN (EN YOĞUN) ---")
    print(sonuc_en_yogun.head(10)[['gun_ismi','saat','OS Name and Version','Device Type','Browser Name and Version','gercek_login','tahmini_login']].to_string(index=False))

    # ======================================================
    # 2️⃣ EZBER KONTROLÜ (ZAMAN BAZLI TRAIN-TEST AYRIMI)
    # ======================================================
    gun_sirasi = login_counts['gun'].unique()
    gun_sirasi.sort()  # 0=Pts, 6=Paz vb.
    bolme_nokta = int(len(gun_sirasi) * 0.8)

    train_days = gun_sirasi[:bolme_nokta]
    test_days = gun_sirasi[bolme_nokta:]

    train_df = login_counts[login_counts['gun'].isin(train_days)]
    test_df = login_counts[login_counts['gun'].isin(test_days)]

    X_train2 = train_df[cols]
    y_train2 = train_df['login_sayisi']
    X_test2 = test_df[cols]
    y_test2 = test_df['login_sayisi']

    model2 = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

    model2.fit(X_train2, y_train2)
    y_pred2 = np.clip(model2.predict(X_test2), a_min=0, a_max=None)

    rmse2 = np.sqrt(mean_squared_error(y_test2, y_pred2))
    print("\n🧠 EZBER KONTROLÜ (Zaman Bazlı Test)")
    print(f"Geçmiş günlerle eğitilip geleceği tahmin etti → RMSE: {rmse2:.2f}")

    fark_orani = ((rmse2 - rmse) / rmse) * 100
    print(f"Fark Oranı: %{fark_orani:.2f} (yüksekse model biraz ezberliyor olabilir)")

    return gun_saat_bazli

def haftalik_login_tahmini(df):
    # --- 1️⃣ Veri Hazırlığı ---
    df = df.copy()
    df = df.dropna(subset=['Login Timestamp'])
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')

    haftalik_logins = (
        df.set_index('Login Timestamp')
          .resample('W-MON')['Login Successful']
          .sum()
          .reset_index()
    )
    haftalik_logins.rename(columns={'Login Successful': 'y', 'Login Timestamp': 'ds'}, inplace=True)

    # Veri filtreleme
    ortalama = haftalik_logins['y'].median()
    haftalik_logins = haftalik_logins[
        (haftalik_logins['y'] > ortalama * 0.3) &
        (haftalik_logins['y'] > 10000)
    ].reset_index(drop=True)

    print(f"✓ Toplam hafta: {len(haftalik_logins)}")
    print(f"✓ Tarih: {haftalik_logins['ds'].min()} → {haftalik_logins['ds'].max()}\n")

    if len(haftalik_logins) < 4:
        print("❌ Yeterli veri yok")
        return

    haftalik_logins['gercek_login'] = haftalik_logins['y'].astype(int)

    # --- 2️⃣ PROPHET - 1 Hafta İleri Tahmin ---
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    prophet_tahminler = []
    min_train_weeks = 3

    for i in range(len(haftalik_logins)):
        if i < min_train_weeks:
            prophet_tahminler.append(None)
            continue

        train_data = haftalik_logins.iloc[:i].copy()
        train_data['y_log'] = np.log1p(train_data['y'])

        model = Prophet(
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.5
        )
        model.fit(train_data[['ds', 'y_log']].rename(columns={'y_log': 'y'}))

        gelecek_tarih = train_data['ds'].max() + pd.Timedelta(weeks=1)
        future = pd.DataFrame({'ds': [gelecek_tarih]})
        tahmin = model.predict(future)
        tahmin_deger = int(round(np.expm1(tahmin['yhat'].values[0])))
        prophet_tahminler.append(tahmin_deger)

    haftalik_logins['tahmini_login'] = prophet_tahminler

    # --- 3️⃣ KALMAN - 1 Hafta İleri (Trendli) ---
    gercek_degerler = haftalik_logins['gercek_login'].values
    kalman_tahminler = []

    for i in range(len(gercek_degerler)):
        if i < min_train_weeks:
            kalman_tahminler.append(None)
            continue

        train_values = gercek_degerler[:i]

        # 🔹 Trendli Kalman: Değer + Trend Bileşeni
        kf = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],
            observation_matrices=[[1, 0]],
            initial_state_mean=[train_values[0], 0],
            transition_covariance=np.eye(2) * 0.01,
            observation_covariance=1.0
        )

        state_means, state_covs = kf.filter(train_values)
        pred_state = kf.transition_matrices @ state_means[-1]
        tahmin = pred_state[0]
        kalman_tahminler.append(int(round(tahmin)))

    haftalik_logins['kalman_tahmin'] = kalman_tahminler

    # --- 4️⃣ HATA HESAPLARI ---
    haftalik_logins_valid = haftalik_logins[
        haftalik_logins['tahmini_login'].notna() &
        haftalik_logins['kalman_tahmin'].notna()
    ].copy()

    haftalik_logins_valid['mutlak_hata'] = abs(
        haftalik_logins_valid['tahmini_login'] - haftalik_logins_valid['gercek_login']
    ).astype(int)
    haftalik_logins_valid['kalman_hata'] = abs(
        haftalik_logins_valid['kalman_tahmin'] - haftalik_logins_valid['gercek_login']
    ).astype(int)
    haftalik_logins_valid['hata_%'] = (haftalik_logins_valid['mutlak_hata'] / haftalik_logins_valid['gercek_login']) * 100
    haftalik_logins_valid['kalman_hata_%'] = (haftalik_logins_valid['kalman_hata'] / haftalik_logins_valid['gercek_login']) * 100

    prophet_rmse = np.sqrt(mean_squared_error(
        haftalik_logins_valid['gercek_login'], haftalik_logins_valid['tahmini_login']
    ))
    kalman_rmse = np.sqrt(mean_squared_error(
        haftalik_logins_valid['gercek_login'], haftalik_logins_valid['kalman_tahmin']
    ))

    print(f"Prophet (1 Hafta İleri) RMSE: {prophet_rmse:,.2f}")
    print(f"Kalman (1 Hafta İleri - Trendli) RMSE:   {kalman_rmse:,.2f}\n")

    # --- 5️⃣ SONUÇ TABLOSU ---
    sonuc = haftalik_logins_valid[['ds', 'gercek_login', 'tahmini_login', 'kalman_tahmin',
                                   'mutlak_hata', 'kalman_hata', 'hata_%', 'kalman_hata_%']].copy()
    sonuc['hata_%'] = sonuc['hata_%'].round(2)
    sonuc['kalman_hata_%'] = sonuc['kalman_hata_%'].round(2)
    pd.options.display.float_format = '{:.2f}'.format

    print("=" * 100)
    print(sonuc.to_string(index=False))
    print("=" * 100)
    print(f"\n📊 Prophet Ortalama Hata: %{sonuc['hata_%'].mean():.2f}")
    print(f"📊 Kalman Ortalama Hata:   %{sonuc['kalman_hata_%'].mean():.2f}")

    # --- 6️⃣ 52 HAFTALIK İLERİ TAHMİN ---
    gelecek_haftalar = 52  # 🔹 1 yıl ileri tahmin

    # Prophet
    final_model = Prophet(
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=True,
        seasonality_mode='additive',
        changepoint_prior_scale=0.5
    )
    final_model.fit(haftalik_logins[['ds', 'y']])
    son_tarih = haftalik_logins['ds'].max()
    gelecek_tarihler = pd.date_range(start=son_tarih + pd.Timedelta(weeks=1),
                                     periods=gelecek_haftalar, freq='W-MON')
    future_df = pd.DataFrame({'ds': gelecek_tarihler})
    gelecek_tahmin = final_model.predict(future_df)
    gelecek_tahmin['prophet_tahmin'] = gelecek_tahmin['yhat'].round().astype(int)

    # Kalman (Trendli - 52 hafta ileri)
    kalman_gelecek = []
    train_values = gercek_degerler
    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[train_values[0], 0],
        transition_covariance=np.eye(2) * 0.01,
        observation_covariance=1.0
    )
    state_means, state_covs = kf.filter(train_values)
    pred_state = state_means[-1]
    for _ in range(gelecek_haftalar):
        pred_state = kf.transition_matrices @ pred_state
        kalman_gelecek.append(int(round(pred_state[0])))

    gelecek_sonuc = pd.DataFrame({
        'Tarih': gelecek_tarihler,
        'Prophet_Tahmin': gelecek_tahmin['prophet_tahmin'].values,
        'Kalman_Tahmin': kalman_gelecek
    })
    print("\n📅 --- 52 HAFTALIK GELECEK TAHMİN ---")
    print(gelecek_sonuc.to_string(index=False))

    return sonuc

def os_4haftalik_tahmin(df):
    populer_os = df["OS Name and Version"].value_counts().head(3).index.tolist()
    tum_tahminler = []

    for os_name in populer_os:
        df_os = df[df["OS Name and Version"] == os_name].copy()
        df_os['tarih'] = pd.to_datetime(df_os['Login Timestamp'].dt.date)

        # Günlük login sayısı
        gunluk_logins = df_os.groupby('tarih').size().reset_index(name='gercek_login')

        if len(gunluk_logins) < 14:
            continue

        # Prophet modeli (optimize edilmiş)
        prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.2
        )
        prophet_model.fit(gunluk_logins.rename(columns={'tarih':'ds', 'gercek_login':'y'}))
        gelecek_28_gun = prophet_model.make_future_dataframe(periods=28)
        tahmin = prophet_model.predict(gelecek_28_gun)
        tahmin['tahmini_login'] = tahmin['yhat'].round().astype(int)

        # Hafta bazında topla
        tahmin['Hafta'] = tahmin['ds'].dt.to_period('W').astype(str)
        haftalik_tahmin = tahmin.groupby('Hafta').agg({'tahmini_login':'sum'}).reset_index()
        haftalik_tahmin['OS Name and Version'] = os_name

        # Gerçek login
        gunluk_logins['Hafta'] = gunluk_logins['tarih'].dt.to_period('W').astype(str)
        gercek_haftalik = gunluk_logins.groupby('Hafta').agg({'gercek_login':'sum'}).reset_index().rename(columns={'y':'gercek_login'})

        # Tahmin ve gerçek birleştir
        df_haftalik = pd.merge(haftalik_tahmin, gercek_haftalik, on='Hafta', how='left')
        df_haftalik = df_haftalik[df_haftalik['gercek_login'].notna()]  # Sadece gerçek veri olan haftalar

        # Hata hesapları
        df_haftalik['mutlak_hata'] = (df_haftalik['tahmini_login'] - df_haftalik['gercek_login']).abs()
        df_haftalik['hata_%'] = df_haftalik['mutlak_hata'] / df_haftalik['gercek_login'] * 100

        tum_tahminler.append(df_haftalik)

    if tum_tahminler:
        final_df = pd.concat(tum_tahminler, ignore_index=True)

        # Genel RMSE hesapla
        genel_rmse = np.sqrt(mean_squared_error(final_df['gercek_login'], final_df['tahmini_login']))
        print(f" Genel RMSE (4 Haftalık, optimize edilmiş): {genel_rmse:.2f}\n")
        # Mutlak hata ve gerçek karşılaştırma
        
        (final_df['tahmini_login'] == final_df['gercek_login']).all()
        final_df['fark'] = final_df['tahmini_login'] - final_df['gercek_login']
        print(final_df[final_df['fark'] != 0])


        # Sütunları düzenle
        final_df = final_df[['Hafta', 'OS Name and Version', 'gercek_login', 'tahmini_login', 'mutlak_hata', 'hata_%']]
        print("---- OS Bazlı 4 Haftalık Gerçek vs Tahmini Login ----")
        print(final_df.to_string(index=False))
    else:
        print("Hiçbir OS için yeterli veri bulunamadı.")

def anomali_tespiti(df, contamination=0.1, verbose=True):
    # Tarih ve saat sütunlarını oluşturuyoruz; saat sadece günlük davranışın varyansını ölçmek için
    df["Tarih"] = pd.to_datetime(df["Login Timestamp"]).dt.date
    df["Saat"] = pd.to_datetime(df["Login Timestamp"]).dt.hour

    # Kullanıcı-gün bazlı özet: login sayısı, saat std, farklı cihaz/OS/şehir sayısı
    user_daily = df.groupby(["User ID", "Tarih"]).agg({
        "Saat": ["count", "std"],
        "Device Type": "nunique",
        "OS Name and Version": "nunique",
        "City": "nunique"
    }).reset_index()

    # Kolon isimlerini anlaşılır hale getiriyoruz
    user_daily.columns = ["Kullanıcı ID", "Tarih", "Günlük Login Sayısı", 
                          "Login Saati Std", "Farklı Cihaz Sayısı", 
                          "Farklı OS Sayısı", "Farklı Şehir Sayısı"]
    
    user_daily["Login Saati Std"] = user_daily["Login Saati Std"].fillna(0)    # Std boşsa 0 olarak ayarlanır; model bozulmasın diye

    # Kullanıcı bazında ortalama ve std login sayısı hesaplıyoruz
    user_stats = user_daily.groupby("Kullanıcı ID")["Günlük Login Sayısı"].agg(["mean", "std"])
    user_stats.columns = ["Ort_Login", "Std_Login"]
    # Kullanıcı-gün dataframe'i ile birleştiriyoruz
    user_daily = user_daily.merge(user_stats, left_on="Kullanıcı ID", right_index=True)
    user_daily["Login Sapması"] = user_daily["Günlük Login Sayısı"] - user_daily["Ort_Login"]   # Login sapmasını hesaplıyoruz; kullanıcı ortalamasına göre fark

    feature_cols = ["Günlük Login Sayısı", "Login Sapması", "Login Saati Std",    # Anomali tespiti için kullanılacak özellikler
                    "Farklı Cihaz Sayısı", "Farklı OS Sayısı", "Farklı Şehir Sayısı"]

    X_scaled = StandardScaler().fit_transform(user_daily[feature_cols].fillna(0)) # Özellikleri ölçeklendiriyoruz; IsolationForest için gereklidir

    predictions = IsolationForest(contamination=contamination, random_state=42, n_estimators=200).fit_predict(X_scaled)
    user_daily["Anormal Mi?"] = np.where(predictions == -1, "Anormal", "Normal") # IsolationForest ile anomali tahmini; -1 anormal, 1 normal

    if verbose:
        print("\n--- Anomali Tespit Sonuçları ---")
        display_cols = ["Kullanıcı ID", "Tarih", "Günlük Login Sayısı", 
                        "Login Saati Std", "Farklı Cihaz Sayısı", "Farklı OS Sayısı", 
                        "Farklı Şehir Sayısı", "Anormal Mi?"]
        print(user_daily[display_cols].iloc[1:51]) 
    return user_daily    # Anormal etiketlenmiş dataframe'i döndürüyoruz

def benzer_login_siniflandir(df, n_clusters=7, verbose=True):
    # Tarih ve saat bilgilerini çıkarıyoruz
    df["Tarih"] = pd.to_datetime(df["Login Timestamp"]).dt.date
    df["Saat"] = pd.to_datetime(df["Login Timestamp"]).dt.hour
    df["Haftanin Gunu"] = pd.to_datetime(df["Login Timestamp"]).dt.dayofweek  # Pazartesi=0

    # Kullanıcı-gün bazlı özet oluşturuluyor
    user_daily = df.groupby(["User ID", "Tarih"]).agg({
        "Saat": ["count", "mean", "std"],
        "Device Type": "nunique",
        "OS Name and Version": "nunique",
        "City": "nunique"
    }).reset_index()

    # Kolon isimlerini anlamlı yapıyoruz
    user_daily.columns = ["Kullanıcı ID", "Tarih", "Günlük Login Sayısı", "Login Saati Ort",
                          "Login Saati Std", "Farklı Cihaz Sayısı", "Farklı OS Sayısı", "Farklı Şehir Sayısı"]

    # Standart sapmaları doldur
    user_daily["Login Saati Std"] = user_daily["Login Saati Std"].fillna(0)
    user_daily["Login Saati Ort"] = user_daily["Login Saati Ort"].fillna(0)

    # Ek özellikler
    user_daily["Cihaz Orani"] = user_daily["Farklı Cihaz Sayısı"] / user_daily["Günlük Login Sayısı"]
    user_daily["Sehir Orani"] = user_daily["Farklı Şehir Sayısı"] / user_daily["Günlük Login Sayısı"]

    # Haftanın günü bazlı aktivite
    haftanin_gunu = df.groupby(["User ID", "Haftanin Gunu"]).size().reset_index(name="Login Sayısı Gunu")
    gunluk_ortalama = haftanin_gunu.groupby("User ID")["Login Sayısı Gunu"].mean().reset_index()
    gunluk_ortalama.rename(columns={"Login Sayısı Gunu": "Haftanin Gunu Ort"}, inplace=True)
    user_daily = user_daily.merge(gunluk_ortalama.rename(columns={"User ID": "Kullanıcı ID"}), on="Kullanıcı ID", how="left")

    # Yoğun login saati (en sık yapılan saat)
    en_yogun_saat = df.groupby(["User ID", "Tarih", "Saat"]).size().reset_index(name="Saat_Sayisi")
    idx = en_yogun_saat.groupby(["User ID", "Tarih"])["Saat_Sayisi"].idxmax()
    en_yogun_saat = en_yogun_saat.loc[idx]
    user_daily = user_daily.merge(en_yogun_saat[["User ID", "Tarih", "Saat"]].rename(
        columns={"User ID": "Kullanıcı ID", "Saat": "Yoğun Saat"}), on=["Kullanıcı ID", "Tarih"], how="left")

    # Özellik listesi
    feature_cols = ["Günlük Login Sayısı", "Login Saati Ort", "Login Saati Std",
                    "Farklı Cihaz Sayısı", "Cihaz Orani",
                    "Farklı OS Sayısı", "Farklı Şehir Sayısı", "Sehir Orani",
                    "Haftanin Gunu Ort", "Yoğun Saat"]

    user_daily = user_daily[user_daily["Günlük Login Sayısı"] < 100]  # mantıklı bir eşik

    # Özellikleri ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_daily[feature_cols].fillna(0))

    # KMeans ile kullanıcıları gruplara ayırıyoruz
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_daily["Davranış Grubu"] = kmeans.fit_predict(X_scaled) + 1  # 1’den başlatıyoruz

    # Her grup için ortalama login sayısı
    group_mean = user_daily.groupby("Davranış Grubu")["Günlük Login Sayısı"].transform("mean")

    # MSE hesapla
    mse = mean_squared_error(user_daily["Günlük Login Sayısı"], group_mean)
    rmse = np.sqrt(mse)
    print(f"Model Hatası (RMSE): {rmse}")

    if verbose:
        print(f"Toplam kullanıcı-gün kayıt sayısı: {len(user_daily)}")
        print("\n--- Davranış Grupları Özeti ---")
        for i in range(1, n_clusters + 1):
            grup = user_daily[user_daily["Davranış Grubu"] == i]
            print(f"\nGrup {i} ({len(grup)} kullanıcı-gün):")
            print(f"  Ort Login: {grup['Günlük Login Sayısı'].mean():.1f}")
            print(f"  Ort Login Saati: {grup['Login Saati Ort'].mean():.1f}")
            print(f"  Ort Saat Std: {grup['Login Saati Std'].mean():.2f}")
            print(f"  Yoğun Saat: {grup['Yoğun Saat'].mean():.1f}")
            print(f"  Ort Cihaz: {grup['Farklı Cihaz Sayısı'].mean():.1f}, Cihaz Orani: {grup['Cihaz Orani'].mean():.2f}")
            print(f"  Ort OS: {grup['Farklı OS Sayısı'].mean():.1f}")
            print(f"  Ort Şehir: {grup['Farklı Şehir Sayısı'].mean():.1f}, Sehir Orani: {grup['Sehir Orani'].mean():.2f}")
            print(f"  Haftanin Gunu Ort: {grup['Haftanin Gunu Ort'].mean():.1f}")

    return scaler, user_daily

# ---------------- Menü ----------------
while True:
    print("\n---- Menü ----")
    secim = input(
        "1- OS/Device Bazlı Tahmin\n"
        "2- Saat/Gün Bazlı Tahmin\n"
        "3- Gelecek Hafta Tahmini\n"
        "4- OS/Device Bazlı Login Sayısı Tahmini (Zaman Serisi)\n"
        "5- Anomali tespiti\n"
        "6- Benzer Login Davranışları\n"
        "7- Çıkış\n"
        "Seçiminiz: "
    )
    if secim == '1': os_device_model(df)
    elif secim == '2': saat_gun_model(df)
    elif secim == '3': haftalik_login_tahmini(df_2020)
    elif secim == '4': os_4haftalik_tahmin(df)
    elif secim == '5': anomali_tespiti(df)
    elif secim == '6': benzer_login_siniflandir(df)
    elif secim == '7':
        print("Çıkış Yapılıyor...")
        break
    else:
        print("Geçersiz seçim! Tekrar deneyiniz.")