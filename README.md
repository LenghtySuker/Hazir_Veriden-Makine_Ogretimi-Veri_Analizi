### Hazir\_Veriden-Makine\_Ogretimi-Veri\_Analizi



\# Login Tahmin ve Davranış Analizi Projesi



Bu proje, kullanıcı login verileri üzerinden makine öğrenmesi modelleri ile tahminler üretmeyi ve geleceğe dair veri öngörüsü sunmayı amaçlamaktadır.



\## 🔍 Proje Özellikleri



Aşağıdaki 7 sabit özellik üzerine analiz ve tahminler yapılmıştır:



1\. \*\*OS/Device Bazlı Tahmin\*\*  

2\. \*\*Saat/Gün Bazlı Tahmin\*\*  

3\. \*\*Gelecek Hafta Tahmini\*\*  

4\. \*\*Kullanıcı Bazlı Saat/Gün Tahmini\*\*  

5\. \*\*OS/Device Bazlı Login Sayısı Tahmini (Zaman Serisi)\*\*  

6\. \*\*Anomali Tespiti\*\*  

7\. \*\*Benzer Login Davranışları\*\*



\## 🧪 Kullanılan Kütüphaneler



Bu projede aşağıdaki Python kütüphaneleri kullanılmıştır:



\- `pandas`, `numpy`: Veri işleme ve sayısal analiz  

\- `scikit-learn`:  

&nbsp; - `train\_test\_split`: Eğitim/test bölme  

&nbsp; - `LabelEncoder`, `StandardScaler`: Ön işleme  

&nbsp; - `RandomForestClassifier`, `RandomForestRegressor`, `IsolationForest`: Tahmin ve anomali tespiti  

&nbsp; - `KMeans`: Kullanıcı davranışı kümeleme  

\- `prophet`: Zaman serisi tahmini  

\- `matplotlib`, `seaborn`: Görselleştirme  

\- `logging`: Uyarı ve hata mesajlarını bastırma



\## 📊 Görselleştirme



Login tahminleri haftalık olarak görselleştirilmiştir. Grafikler `matplotlib` ve `seaborn` ile oluşturulmuştur. Prophet ve cmdstanpy kütüphanelerinden gelen uyarılar bastırılarak terminal çıktısı sade tutulmuştur.



\## 📦 Veri Kaynağı



Bu projede kullanılan veri seti Kaggle üzerinden alınmıştır:  

\*\*Lisans\*\*: \[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  

Veri seti ilgili sahibine aittir ve bu kod yalnızca analiz amaçlı kullanılmıştır.



\## ⚙️ Kurulum



```bash

pip install -r requirements.txt

python tahmin.py

