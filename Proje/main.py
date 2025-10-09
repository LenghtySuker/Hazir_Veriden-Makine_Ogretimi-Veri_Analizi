import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from prophet import Prophet
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import warnings

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
    y_pred_prob = model.predict_proba(X_test)[:,1]

    sonuc = X_test.copy()
    for col in ["OS Name and Version", "Device Type"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc'])
    sonuc['login_olasilik'] = y_pred_prob*100

    print("---- OS/Device Bazlı Login Olasılıkları ----")
    print(sonuc[["OS Name and Version", "Device Type", "login_olasilik"]].head(20))

def saat_gun_model(df):
    cols = ['gun','saat'] + [col+'_enc' for col in ["OS Name and Version", "Device Type", "Browser Name and Version"]]
    login_counts = df.groupby(cols).size().reset_index(name='login_sayisi')
    X = login_counts[cols]
    y = login_counts['login_sayisi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    sonuc = X_test.copy()
    sonuc['tahmini_login'] = y_pred.round().astype(int)
    for col in ["OS Name and Version", "Device Type", "Browser Name and Version"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc'].astype(int))
    sonuc['gun'] = sonuc['gun'].map(gun_dict)

    print("---- Saat/Gün + Ek Özellikler Bazlı Tahmini Login Sayısı ----")
    print(sonuc[['gun','saat','OS Name and Version','Device Type','Browser Name and Version','tahmini_login']].head(20))

def gelecek_hafta_tahmin(df):
    gunluk_logins = df.groupby(df['Login Timestamp'].dt.date).size().reset_index(name='y')
    gunluk_logins['ds'] = pd.to_datetime(gunluk_logins['Login Timestamp'])
    tarih_araligi = pd.date_range(start=gunluk_logins['ds'].min(), end=gunluk_logins['ds'].max())
    gunluk_logins = gunluk_logins.set_index('ds').reindex(tarih_araligi, fill_value=0).rename_axis('ds').reset_index()
    if len(gunluk_logins) < 7:
        print("Yeterli veri yok, tahmin üretilemiyor.")
        return

    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(gunluk_logins[['ds', 'y']])
    gelecek_28_gun = prophet_model.make_future_dataframe(periods=28)
    tahmin = prophet_model.predict(gelecek_28_gun)

    haftalik_tahmin = tahmin[['ds', 'yhat']].set_index('ds').resample('W-MON').sum().reset_index().rename(columns={'ds': 'Hafta', 'yhat': 'Tahmini Login'})
    haftalik_tahmin['Tahmini Login'] = haftalik_tahmin['Tahmini Login'].round().astype(int)
    haftalik_tahmin = haftalik_tahmin.tail(4).reset_index(drop=True)

    plt.figure(figsize=(10,6))
    sns.barplot(data=haftalik_tahmin, x='Hafta', y='Tahmini Login', palette='crest')
    plt.title("Gelecek 4 Hafta Login Tahmini")
    plt.xlabel("Hafta Başlangıcı")
    plt.ylabel("Tahmini Login Sayısı")
    plt.xticks(rotation=45, ha='right')
    for idx, val in enumerate(haftalik_tahmin['Tahmini Login']):
        plt.text(idx, val + 0.5, str(val), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

    print("---- Gelecek 4 Hafta Login Tahmini ----")
    print(haftalik_tahmin.to_string(index=False))

def kullanici_saat_gun_model(df):
    cols = ['gun','saat'] + [col+'_enc' for col in ["OS Name and Version", "Device Type"]]
    X = df[cols]
    y = df['Login Successful']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:,1]

    sonuc = X_test.copy()
    sonuc['login_olasilik'] = y_pred_prob*100
    for col in ["OS Name and Version", "Device Type"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc'])
    sonuc['gun'] = sonuc['gun'].map(gun_dict)

    print("---- Kullanıcı Bazlı Saat/Gün Login Olasılıkları ----")
    print(sonuc[['gun','saat','OS Name and Version','Device Type','login_olasilik']].head(20))

def os_4haftalik_tahmin(df):
    populer_os = df["OS Name and Version"].value_counts().head(3).index.tolist()
    tum_tahminler = []

    for os_name in populer_os:
        df_os = df[df["OS Name and Version"] == os_name].copy()
        df_os['tarih'] = pd.to_datetime(df_os['Login Timestamp'].dt.date)
        gunluk_logins = df_os.groupby('tarih').size().reset_index(name='y')
        if len(gunluk_logins) < 7:
            continue

        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(gunluk_logins.rename(columns={'tarih':'ds'}))
        gelecek_28_gun = prophet_model.make_future_dataframe(periods=28)
        tahmin = prophet_model.predict(gelecek_28_gun)

        gelecek = tahmin[['ds','yhat']].tail(28).copy()
        gelecek['Tahmini Login'] = gelecek['yhat'].round().astype(int)
        gelecek['OS Name and Version'] = os_name
        gelecek['Hafta'] = gelecek['ds'].dt.to_period('W').astype(str)

        tum_tahminler.append(gelecek.groupby(['Hafta','OS Name and Version'])['Tahmini Login'].sum().reset_index())

    if tum_tahminler:
        os_tahmin_df = pd.concat(tum_tahminler, ignore_index=True)
        os_tahmin_df = os_tahmin_df.groupby(['OS Name and Version']).tail(4).reset_index(drop=True)

        haftalar = os_tahmin_df['Hafta'].unique()
        x = np.arange(len(haftalar))
        width = 0.3
        fig, ax = plt.subplots(figsize=(10,5))
        os_list = os_tahmin_df['OS Name and Version'].unique()
        n_os = len(os_list)

        for i, os_name in enumerate(os_list):
            df_plot = os_tahmin_df[os_tahmin_df['OS Name and Version'] == os_name]
            ax.bar(x + i*width, df_plot['Tahmini Login'], width, label=os_name)
            for j, val in enumerate(df_plot['Tahmini Login']):
                ax.text(x[j] + i*width, val + 0.5, str(int(val)), ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x + width*(n_os-1)/2)
        ax.set_xticklabels(haftalar)
        ax.set_xlabel("Hafta")
        ax.set_ylabel("Tahmini Login Sayısı")
        ax.set_title("En Popüler 3 OS için 4 Haftalık Tahmin")
        ax.legend(title='OS')
        plt.tight_layout()
        plt.show()
        print("---- OS Bazlı Gelecek 4 Hafta Login Tahmini ----")
        print(os_tahmin_df.iloc[0:12])
    else:
        print("Hiçbir OS için tahmin üretilemedi.")


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

    user_daily["Login Saati Std"] = user_daily["Login Saati Std"].fillna(0)

    # Yeni özellikler
    user_daily["Login Saati Ort"] = user_daily["Login Saati Ort"].fillna(0)
    user_daily["Cihaz Orani"] = user_daily["Farklı Cihaz Sayısı"] / user_daily["Günlük Login Sayısı"]
    user_daily["Sehir Orani"] = user_daily["Farklı Şehir Sayısı"] / user_daily["Günlük Login Sayısı"]

    feature_cols = ["Günlük Login Sayısı", "Login Saati Ort", "Login Saati Std",
                    "Farklı Cihaz Sayısı", "Cihaz Orani",
                    "Farklı OS Sayısı", "Farklı Şehir Sayısı", "Sehir Orani"]

    # Özellikleri ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_daily[feature_cols].fillna(0))

   # KMeans ile kullanıcıları gruplara ayırıyoruz
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_daily["Davranış Grubu"] = kmeans.fit_predict(X_scaled) + 1  # 1’den başlatıyoruz

    if verbose:
        print(f"Toplam kullanıcı-gün kayıt sayısı: {len(user_daily)}")
        print("\n--- Davranış Grupları Özeti ---")
        for i in range(1, n_clusters + 1):  # 🔹 1’den n_clusters’a kadar
            grup = user_daily[user_daily["Davranış Grubu"] == i]
            print(f"\nGrup {i} ({len(grup)} kullanıcı-gün):")
            print(f"  Ort Login: {grup['Günlük Login Sayısı'].mean():.1f}")
            print(f"  Ort Login Saati: {grup['Login Saati Ort'].mean():.1f}")
            print(f"  Ort Saat Std: {grup['Login Saati Std'].mean():.2f}")
            print(f"  Ort Cihaz: {grup['Farklı Cihaz Sayısı'].mean():.1f}, Cihaz Orani: {grup['Cihaz Orani'].mean():.2f}")
            print(f"  Ort OS: {grup['Farklı OS Sayısı'].mean():.1f}")
            print(f"  Ort Şehir: {grup['Farklı Şehir Sayısı'].mean():.1f}, Sehir Orani: {grup['Sehir Orani'].mean():.2f}")
    return scaler, user_daily        

# ---------------- Menü ----------------
while True:
    print("\n---- Menü ----")
    secim = input(
        "1- OS/Device Bazlı Tahmin\n"
        "2- Saat/Gün Bazlı Tahmin\n"
        "3- Gelecek Hafta Tahmini\n"
        "4- Kullanıcı Bazlı Saat/Gün Tahmini\n"
        "5- OS/Device Bazlı Login Sayısı Tahmini (Zaman Serisi)\n"
        "6- Anomali tespiti\n"
        "7- Benzer Login Davranışları\n"
        "8- Çıkış\n"
        "Seçiminiz: "
    )
    if secim == '1': os_device_model(df)
    elif secim == '2': saat_gun_model(df)
    elif secim == '3': gelecek_hafta_tahmin(df)
    elif secim == '4': kullanici_saat_gun_model(df)
    elif secim == '5': os_4haftalik_tahmin(df)
    elif secim == '6': anomali_tespiti(df)
    elif secim == '7': benzer_login_siniflandir(df)
    elif secim == '8':
        print("Çıkış Yapılıyor...")
        break
    else:
        print("Geçersiz seçim! Tekrar deneyiniz.")