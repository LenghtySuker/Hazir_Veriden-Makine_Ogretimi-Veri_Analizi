### Hazir\_Veriden-Makine\_Ogretimi-Veri\_Analizi



\# Login Tahmin ve Davranış Analizi Projesi



Bu proje, kullanıcı login verileri üzerinden makine öğrenmesi modelleri ile tahminler üretmeyi ve geleceğe dair veri öngörüsü sunmayı amaçlamaktadır.



\## 🔍 Proje Özellikleri



Aşağıdaki 8 sabit özellik üzerine analiz ve tahminler yapılmıştır:


1- OS / Browser Yoğunluk Analizi
2- Client Analizi
3- Saat / Gün Tahmini
4- Haftalık Login Tahmini
5- Gelecek Hafta Login Tahmini(Sistemi her çalıştırdığımızda o tarihten 1 yıllık gelecek tahmini)
6- OS 4 Haftalık Tahmin
7- Anomali Tespiti
8- Login Kümeleme Analizi


\## 🧪 Kullanılan Kütüphaneler
---

## Bu projede kullanılan Python kütüphaneleri

- **Veri İşleme ve Analiz**
  - `pandas`, `numpy`: Veri işleme ve sayısal analiz  
  - `polars`: Alternatif hızlı veri işleme (opsiyonel, varsa)  

- **Makine Öğrenmesi ve Modelleme**
  - `scikit-learn`:  
    - `train_test_split`: Eğitim/test bölme  
    - `LabelEncoder`, `StandardScaler`: Ön işleme  
    - `IsolationForest`: Anomali tespiti  
    - `KMeans`: Kullanıcı davranışı kümeleme  
    - `mean_squared_error`, `silhouette_score`, `davies_bouldin_score`, `r2_score`: Model değerlendirme metrikleri  
  - `lightgbm`: Gradient boosting tabanlı tahmin modelleri  
  - `xgboost`, `XGBRegressor`: Güçlü tahmin modelleri (boosting algoritmaları)  
  - `torch`: GPU desteği kontrolü  

- **Zaman Serisi Analizi**
  - `prophet`: Zaman serisi tahmini  
  - `statsmodels` (`SARIMAX`): Zaman serisi modelleme  

- **Sistem ve Yardımcı Araçlar**
  - `logging`: Uyarı ve hata mesajlarını bastırma  
  - `warnings`: Uyarı filtreleme  
  - `sys`, `pathlib.Path`: Sistem ve dosya işlemleri  
  - `datetime`, `timedelta`, `date`: Tarih/zaman işlemleri  
  - `itertools`: Kombinasyon ve iterasyon işlemleri  

- **Dış Kaynaklar ve API**
  - `requests`: HTTP istekleri  
  - `feedparser`: RSS/Atom veri çekme  

---


Login tahminleri haftalık olarak görselleştirilmiştir. Grafikler `matplotlib` ve `seaborn` ile oluşturulmuştur. Prophet ve cmdstanpy kütüphanelerinden gelen uyarılar bastırılarak terminal çıktısı sade tutulmuştur.



\## 📦 Veri Kaynağı



Bu projede kullanılan veri seti Kaggle üzerinden alınmıştır:  

\*\*Lisans\*\*: \[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  

Veri seti ilgili sahibine aittir ve bu kod yalnızca analiz amaçlı kullanılmıştır.


## 📦 Veri Dosyası

Bu projede kullanılan veri dosyası Kaggle üzerinden indirilmiştir ve yerel olarak şu dizinde tutulmaktadır:

`C:\Users\Aykut\.cache\kagglehub\datasets\dasgroup\rba-dataset\versions\1\rba-dataset.csv`

Veri dosyası GitHub’a yüklenmemiştir.  
Projeyi çalıştırmak için Kaggle hesabınızla [rba-dataset](https://www.kaggle.com/datasets/dasgroup/rba-dataset) sayfasından veriyi indirmeniz ve kodda `dosya_yolu` değişkenini kendi sisteminize göre güncellemeniz gerekmektedir.



\## ⚙️ Kurulum



```bash

pip install -r requirements.txt

python tahmin.py

