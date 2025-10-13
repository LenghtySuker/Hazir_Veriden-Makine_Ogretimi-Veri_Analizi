import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from prophet import Prophet
import logging
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


logging.getLogger('prophet').setLevel(logging.ERROR) 
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# ---------------- Veri Yükleme ----------------
dosya_yolu = r"C:\Users\Aykut\.cache\kagglehub\datasets\dasgroup\rba-dataset\versions\1\rba-dataset.csv"
df = pd.read_csv(dosya_yolu, nrows=2_000_000)

df["Login Successful"] = df["Login Successful"].astype(int)
df = df.fillna("Bilinmiyor")
df["Round-Trip Time [ms]"] = pd.to_numeric(df["Round-Trip Time [ms]"], errors="coerce").fillna(0)
df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')  
df['saat'] = df['Login Timestamp'].dt.hour
df['gun'] = df['Login Timestamp'].dt.dayofweek  

gun_dict = {0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe', 4: 'Cuma', 5: 'Cumartesi', 6: 'Pazar'} 

# ---------------- Label Encoding ----------------
label_cols = ["OS Name and Version", "Device Type", "Browser Name and Version", "Country", "Region", "City", "ASN"]  
encoders = {}
for col in label_cols:  
    le = LabelEncoder()
    df[col+'_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ---------------- Fonksiyonlar ----------------
def os_device_model(df):
    X = df[[col+'_enc' for col in ["OS Name and Version", "Device Type"]]]
    y = df["Login Successful"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Sınıf tahmini
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Olasılık tahmini
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    # Sonuç DataFrame
    sonuc = X_test.copy()
    for col in ["OS Name and Version", "Device Type"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc'])
    sonuc['login_olasilik'] = y_pred_prob*100
    
    print("---- OS/Device Bazlı Login Olasılıkları ----")
    print(f"Model Doğruluk Oranı: {accuracy:.4f}")
    
    mse = mean_squared_error(y_test, y_pred_prob)
    rmse = np.sqrt(mse)
    print(f"Model Hatası (RMSE): {rmse:.4f}")
    
    print(sonuc[["OS Name and Version", "Device Type", "login_olasilik"]].head(20))

def saat_gun_model(df):
    cols = ['gun','saat'] + [col+'_enc' for col in ["OS Name and Version", "Device Type", "Browser Name and Version"]]
    login_counts = df.groupby(cols).size().reset_index(name='login_sayisi')

    X = login_counts[cols]
    y = login_counts['login_sayisi']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=1500,       # Çok sayıda ağaç → daha stabil tahmin
        max_depth=40,         # Sınırsız derinlik → veri detaylarını yakalama
        min_samples_split=2,    # Dallanmada minimum örnek → daha hassas ağaçlar
        min_samples_leaf=1,     # Yaprakta minimum örnek → nadir olayları yakalar
        random_state=42,
        n_jobs=-1               # Tüm çekirdekleri kullan
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Clip düzeltmesi
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    
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

    # En yoğun 10 tahmini çıkarmak
    sonuc_en_yogun = sonuc.groupby(['gun_ismi','saat','OS Name and Version','Device Type','Browser Name and Version']) \
                          .agg({'tahmini_login':'sum','gercek_login':'sum'}) \
                          .reset_index() \
                          .sort_values(by='tahmini_login', ascending=False)
    
    print("\n--- ÖRNEK 10 TAHMİN (EN YOĞUN) ---")
    print(sonuc_en_yogun.head(10)[['gun_ismi','saat','OS Name and Version','Device Type','Browser Name and Version','gercek_login','tahmini_login']].to_string(index=False))

    return gun_saat_bazli

def haftalik_login_tahmini(df):
     # Günlük login sayıları
    gunluk_logins = df.groupby(df['Login Timestamp'].dt.date).size().reset_index(name='y')
    gunluk_logins['ds'] = pd.to_datetime(gunluk_logins['Login Timestamp'])

    # Eksik tarihleri doldur
    tarih_araligi = pd.date_range(start=gunluk_logins['ds'].min(), end=gunluk_logins['ds'].max())
    gunluk_logins = gunluk_logins.set_index('ds').reindex(tarih_araligi, fill_value=0).rename_axis('ds').reset_index()

    if len(gunluk_logins) < 14:
        print("Yeterli veri yok, tahmin üretilemiyor.")
        return

    # Log dönüşümü
    gunluk_logins['y'] = np.log1p(gunluk_logins['y'])

    # Prophet modeli
    prophet_model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )
    prophet_model.fit(gunluk_logins[['ds', 'y']])

    # Tahmin
    tahmin = prophet_model.predict(gunluk_logins[['ds']])
    tahmin['yhat_real'] = np.expm1(tahmin['yhat'])

    # Gerçek ve tahmini login
    gunluk_logins['tahmini_login'] = tahmin['yhat_real'].round().astype(int)
    gunluk_logins['gercek_login'] = np.expm1(gunluk_logins['y'])

    # Haftalık toplamlar
    gunluk_logins['ds'] = pd.to_datetime(gunluk_logins['ds'])
    haftalik = gunluk_logins[['ds', 'gercek_login', 'tahmini_login']].resample('W-MON', on='ds').sum().reset_index()

    # İlk haftayı kaldır (yeterli veri yok)
    if len(haftalik) > 1:
        haftalik = haftalik.iloc[1:].reset_index(drop=True)

    # Hata hesapları
    haftalik['mutlak_hata'] = abs(haftalik['tahmini_login'] - haftalik['gercek_login'])
    haftalik['hata_%'] = (haftalik['mutlak_hata'] / np.maximum(haftalik['gercek_login'], 1)) * 100

    # RMSE
    rmse = np.sqrt(mean_squared_error(haftalik['gercek_login'], haftalik['tahmini_login']))
    print(f"Model Hatası (RMSE - Haftalık): {rmse:.2f}\n")

    # Tabloyu ekrana yazdır
    print(haftalik[['ds', 'gercek_login', 'tahmini_login', 'mutlak_hata', 'hata_%']])

    return haftalik[['ds', 'gercek_login', 'tahmini_login', 'mutlak_hata', 'hata_%']]

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
    elif secim == '3': haftalik_login_tahmini(df)
    elif secim == '4': os_4haftalik_tahmin(df)
    elif secim == '5': anomali_tespiti(df)
    elif secim == '6': benzer_login_siniflandir(df)
    elif secim == '7':
        print("Çıkış Yapılıyor...")
        break
    else:
        print("Geçersiz seçim! Tekrar deneyiniz.")