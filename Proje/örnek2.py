import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split  # veriyi eğitim ve test setine bölmek için kullanılır
from sklearn.preprocessing import LabelEncoder, StandardScaler  # LabelEncoder: kategorik değişkenleri sayısal hale getirir, StandardScaler: veriyi normalize eder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest  # RandomForestClassifier/Regressor: sınıflandırma ve regresyon için, IsolationForest: anomali tespiti
from sklearn.cluster import KMeans  # KMeans: kullanıcı davranışlarını gruplamak için kümeleme algoritması
from prophet import Prophet  # Prophet: zaman serisi tahmini için kullanılan model
import logging  # logging: hata ve uyarıları yönetmek için kullanılır
import matplotlib.pyplot as plt
import seaborn as sns


# logging seviyelerini düşürerek fazla mesaj çıkmasını engelliyoruz
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)  # Prophet içindeki cmdstanpy kütüphanesinin hatalarını bastırır
logging.getLogger('prophet').setLevel(logging.ERROR)  # Prophet uyarılarını bastırır

# ---------------- Veri Yükleme ----------------
#Kullanırken veriyi kendi bilgisayarına yükleyerek kendi dosya yolunu gir
dosya_yolu = r"C:\Users\Aykut\.cache\kagglehub\datasets\dasgroup\rba-dataset\versions\1\rba-dataset.csv"
#Kullanırken veriyi kendi bilgisayarına yükleyerek kendi dosya yolunu gir

df = pd.read_csv(dosya_yolu, nrows=2_000_000)

df["Login Successful"] = df["Login Successful"].astype(int)  # Başarılı login bilgisini integer yapıyoruz, ML modelleri string kabul etmez.
df = df.fillna("Bilinmiyor")  # Boş değerleri "Bilinmiyor" ile dolduruyoruz, boş bırakılırsa encoding veya model çalışmaz.
df["Round-Trip Time [ms]"] = pd.to_numeric(df["Round-Trip Time [ms]"], errors="coerce").fillna(0)# Round-Trip Time sayısal olmayan değerleri zorla sayıya çeviriyoruz, NaN olursa 0 ile dolduruyoruz. Model sayısal değer bekler.

df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'], errors='coerce')  
# Tarihleri datetime formatına çeviriyoruz, böylece saat ve gün çıkarabiliriz. Olmazsa saat/gün hesaplanamaz.
df['saat'] = df['Login Timestamp'].dt.hour  # Saat bilgisini datetime'dan çekiyoruz, zaman bazlı tahmin için gerekli.
df['gun'] = df['Login Timestamp'].dt.dayofweek  # Haftanın gününü 0-6 arası sayısal olarak alıyoruz, zaman bazlı analiz için lazım.

gun_dict = {0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe', 4: 'Cuma', 5: 'Cumartesi', 6: 'Pazar'}  

# ---------------- Label Encoding ----------------
label_cols = ["OS Name and Version", "Device Type", "Browser Name and Version", "Country", "Region", "City", "ASN"]  
# Kategorik sütunları listeledik, çünkü ML modelleri sayısal veri ister.
encoders = {}  # Her sütun için encoder saklayacağız.

for col in label_cols:
    le = LabelEncoder()  # LabelEncoder ile kategorik veriyi sayısala çeviriyoruz.
    df[col+'_enc'] = le.fit_transform(df[col].astype(str))  
    # Sayısal olmayan kategorik veriyi sayısal koda çeviriyoruz. Olmazsa ML modeli hata verir.
    encoders[col] = le  # Daha sonra tersini almak için encoder kaydediyoruz.
# ---------------- Fonksiyonlar ----------------
def os_device_model(df):
    X = df[[col+'_enc' for col in ["OS Name and Version", "Device Type"]]]# Model için girdi değişkenleri: OS ve cihaz tipi sayısal kodları
    y = df["Login Successful"]# Hedef değişken: Login başarılı mı?

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# Veriyi eğitim ve test olarak ayırıyoruz. Modelin doğruluğunu test için kullanacağız.

    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)# Random Forest classifier: login olasılığı tahmini için. Çok sayıda ağaç ve maksimim derinlik belirleniyor.
    model.fit(X_train, y_train)# Modeli eğitiyoruz

    y_pred_prob = model.predict_proba(X_test)[:,1]# Login olasılığı tahmini, 0-1 arası

    sonuc = X_test.copy()  
    for col in ["OS Name and Version", "Device Type"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc']) # Sayısal kodları tekrar orijinal kategoriye çeviriyoruz, okunabilir çıktı için.
    sonuc['login_olasilik'] = y_pred_prob*100  # Olasılığı % olarak gösteriyoruz

    print("---- OS/Device Bazlı Login Olasılıkları ----")
    print(sonuc[[ "OS Name and Version", "Device Type", "login_olasilik"]].head(20))  # İlk 20 örneği gösteriyoruz

def saat_gun_model(df):
    cols = ['gun','saat'] + [col+'_enc' for col in ["OS Name and Version", "Device Type", "Browser Name and Version"]]     # Girdi değişkenleri: gün, saat ve ek kategorik özellikler
    login_counts = df.groupby(cols).size().reset_index(name='login_sayisi')     # Her kombinasyon için login sayısını sayıyoruz

    X = login_counts[cols]
    y = login_counts['login_sayisi']   # Model girdi ve hedef değişkenler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Eğitim/test ayırımı

    model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)   # Random Forest Regressor ile sayısal login tahmini yapıyoruz
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) # Test seti için tahmin

    sonuc = X_test.copy()
    sonuc['tahmini_login'] = y_pred.round().astype(int)   # Tahmini login sayısını yuvarlayıp integer yapıyoruz
    for col in ["OS Name and Version", "Device Type", "Browser Name and Version"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc'].astype(int)) # Kodları kategoriye geri çeviriyoruz
    sonuc['gun'] = sonuc['gun'].map(gun_dict)   # Günleri isimle eşleştiriyoruz

    print("---- Saat/Gün + Ek Özellikler Bazlı Tahmini Login Sayısı ----")
    print(sonuc[['gun','saat','OS Name and Version','Device Type','Browser Name and Version','tahmini_login']].head(20)) # İlk 20 tahmini gösteriyoruz

def gelecek_hafta_tahmin(df):
    # Günlük login sayısı
    gunluk_logins = df.groupby(df['Login Timestamp'].dt.date).size().reset_index(name='y')
    gunluk_logins['ds'] = pd.to_datetime(gunluk_logins['Login Timestamp'])

    # Eksik günleri doldur
    tarih_araligi = pd.date_range(start=gunluk_logins['ds'].min(), end=gunluk_logins['ds'].max())
    gunluk_logins = (
        gunluk_logins
        .set_index('ds')
        .reindex(tarih_araligi, fill_value=0)
        .rename_axis('ds')
        .reset_index()
    )

    # Veri yetersizse çık
    if len(gunluk_logins) < 7:
        print("Yeterli veri yok, tahmin üretilemiyor.")
        return

    # Prophet modeli ile tahmin
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(gunluk_logins[['ds', 'y']])

    # Gelecek 28 gün için tahmin üret
    gelecek_28_gun = prophet_model.make_future_dataframe(periods=28)
    tahmin = prophet_model.predict(gelecek_28_gun)

    # Haftalık toplamları Pazartesi başlangıçlı olarak grupla
    haftalik_tahmin = (
        tahmin[['ds', 'yhat']]
        .set_index('ds')
        .resample('W-MON')  # Pazartesi başlangıçlı haftalık örnekleme
        .sum()
        .reset_index()
        .rename(columns={'ds': 'Hafta', 'yhat': 'Tahmini Login'})
    )

    # Tahmini login sayısını tam sayıya çevir
    haftalik_tahmin['Tahmini Login'] = haftalik_tahmin['Tahmini Login'].round().astype(int)

    # Sadece son 4 haftayı al
    haftalik_tahmin = haftalik_tahmin.tail(4).reset_index(drop=True)

    # 🎨 Görselleştirme
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=haftalik_tahmin,
        x='Hafta',
        y='Tahmini Login',
        hue='Hafta',

        palette='crest',
        legend=False  # uyarıyı engeller
    )
    plt.title("Gelecek 4 Hafta Login Tahmini", fontsize=14, weight='bold', pad=20)
    plt.xlabel("Hafta Başlangıcı", fontsize=12)
    plt.ylabel("Tahmini Login Sayısı", fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Her sütunun üstüne değer yaz
    for index, value in enumerate(haftalik_tahmin['Tahmini Login']):
        plt.text(index, value + (value*0.02), str(value), ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.show()

    # Tablo olarak yazdır
    print("---- Gelecek 4 Hafta Login Tahmini ----")
    print(haftalik_tahmin.to_string(index=False))


def kullanici_saat_gun_model(df):
    cols = ['gun','saat'] + [col+'_enc' for col in ["OS Name and Version", "Device Type"]]   # Girdi değişkenleri: gün, saat ve OS + cihaz tipi kodları
    X = df[cols]
    y = df['Login Successful']  # Hedef değişken: login başarılı mı?

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Eğitim ve test seti oluştur

    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)  # Random Forest classifier
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:,1]   # Login olasılığı tahmini

    sonuc = X_test.copy()
    sonuc['login_olasilik'] = y_pred_prob*100   # % olarak göster

    for col in ["OS Name and Version", "Device Type"]:
        sonuc[col] = encoders[col].inverse_transform(sonuc[col+'_enc'])     # Kodları kategoriye geri çevir

    sonuc['gun'] = sonuc['gun'].map(gun_dict)  # Gün isimlerini yazdır

    print("---- Kullanıcı Bazlı Saat/Gün Login Olasılıkları ----")
    print(sonuc[['gun','saat','OS Name and Version','Device Type','login_olasilik']].head(20))

def os_4haftalik_tahmin(df):
    populer_os = df["OS Name and Version"].value_counts().head(3).index.tolist()# En popüler 3 işletim sistemi seçiliyor; en çok login yapan OS'ler için tahmin yapılacak
    tum_tahminler = []  # Tüm OS'ler için haftalık tahminler buraya eklenecek

    for os_name in populer_os:
        df_os = df[df["OS Name and Version"] == os_name].copy()# Sadece ilgili OS için veri alıyoruz
        df_os['tarih'] = pd.to_datetime(df_os['Login Timestamp'].dt.date)# Tarih sütunu oluşturuyoruz; Prophet için 'ds' sütunu gerekli
        gunluk_logins = df_os.groupby('tarih').size().reset_index(name='y')# Günlük login sayısını hesaplıyoruz

        # Veri yetersizse tahmin yapamayız, minimum 7 gün veri gerekiyor
        if len(gunluk_logins) < 7:
            continue

        prophet_model = Prophet(daily_seasonality=True)# Prophet modeli oluşturuluyor; günlük sezonluk davranışı yakalaması için daily_seasonality=True
        prophet_model.fit(gunluk_logins.rename(columns={'tarih':'ds'}))# Prophet modelini veriyle eğitiyoruz
        gelecek_28_gun = prophet_model.make_future_dataframe(periods=28) # Gelecek 28 gün için tahmin yapılacak dataframe    
        tahmin = prophet_model.predict(gelecek_28_gun)    # Tahminler oluşturuluyor

        gelecek = tahmin[['ds','yhat']].tail(28).copy()    # Sadece son 28 günün tahminini alıyoruz
        gelecek['Tahmini Login'] = gelecek['yhat'].round().astype(int)  # Tahmini login sayısını integer yapıyoruz      
        gelecek['OS Name and Version'] = os_name  # OS bilgisini ekliyoruz        
        gelecek['Hafta'] = gelecek['ds'].dt.to_period('W').astype(str)# Haftalık gruplama için tarihleri haftaya dönüştürüyoruz

        tum_tahminler.append(       # Haftalık toplam tahmini login sayısını ekliyoruz
            gelecek.groupby(['Hafta','OS Name and Version'])['Tahmini Login'].sum().reset_index()
        )

    if tum_tahminler:
        # Tüm OS'ler için tahminleri birleştiriyoruz
        os_tahmin_df = pd.concat(tum_tahminler, ignore_index=True)
        os_tahmin_df = os_tahmin_df.groupby(['OS Name and Version']).tail(4).reset_index(drop=True)
        
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
                            #----MENÜ----
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