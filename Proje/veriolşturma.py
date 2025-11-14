import random
import uuid
import json
from datetime import datetime, timedelta

def generate_realistic_5year_data(total_logins=10_000_000, num_users=500_000, anomaly_rate=0.05):
    """
    5 YILLIK (2020-2025) GERÇEKÇİ LOGİN VERİSİ OLUŞTURUCU
    - Akademik takvime göre haftalık dağılım
    - Anomali özellikleri: Günlük login sayısı, saat, client, IP değişimi
    """
    
    print("🚀 5 YILLIK GERÇEKÇİ VERİ OLUŞTURUCU")
    print("="*70)
    
    # İSİM LİSTELERİ
    first_names = [
        "Ahmet","Mehmet","Ayşe","Fatma","Mahmut","Elif","Deniz","Ali","Zeynep","Eren",
        "Selin","Okan","Gizem","Burak","Emre","Cansu","Hakan","Betül","Tolga","Ece",
        "Can","Seda","Mert","Derya","Arda","Esra","Kaan","Leyla","Onur","Melisa",
        "Serkan","İpek","Baran","Berk","Naz","Sinem","Oğuz","Tuna","Cem","İrem",
        "Yağmur","Yusuf","İlker","Rabia","Gökhan","Pelin","Aslı","Volkan","Alper","Buse",
        "Rüya","Nihat","Oya","Tamer","Zehra","Hilal","Hasan","Tuğba","Enes","Çağlar",
        "Defne","Furkan","Ebru","Mustafa","Nazlı","Kerem","Duygu","Barış","Ceren","Umut"
    ]

    last_names = [
        "Yılmaz","Kaya","Demir","Çelik","Şahin","Rizelioğlu","Koç","Arslan","Öztürk","Acar",
        "Korkmaz","Yıldız","Taş","Bulut","Erdoğan","Kurt","Aksoy","Erol","Güneş","Şimşek",
        "Kaplan","Bozkurt","Polat","Avcı","Toprak","Özdemir","Baş","Karaca","Kocaman","Güner",
        "Altun","İnce","Köse","Yavuz","Kara","Doğan","Bilgin","Aydın","Ertaş","Çetin",
        "Durmaz","Tekin","Işık","Özer","Çakır","Turan","Aslan","Bayram","Çiftçi","Demirtaş"
    ]

    # TARAYICILAR
    browsers = [
        ("Chrome 120", 25), ("Chrome 125", 20), ("Chrome 115", 15),
        ("Edge 121", 10), ("Edge 123", 8),
        ("Safari 17", 6), ("Safari 18", 5),
        ("Firefox 118", 4), ("Firefox 119", 3),
        ("Opera 104", 1.5), ("Opera GX", 1),
        ("Brave 1.64", 0.5), ("Yandex 23.7", 0.3), ("Vivaldi 6.4", 0.2),
        ("Tor 12.0", 0.1), ("UC Browser 16", 0.1), ("Samsung Internet 23", 0.1),
        ("QQ Browser 15", 0.05), ("Chromium 110", 0.05), ("Epic 22", 0.05)
    ]

    # İŞLETİM SİSTEMLERİ
    os_list = [
        ("Windows 10", 30), ("Windows 11", 25), ("Windows 8.1", 5),
        ("Android 13", 12), ("Android 14", 10),
        ("iOS 16", 6), ("iOS 17", 5),
        ("macOS 14", 3), ("macOS 13", 2),
        ("Ubuntu 22.04", 0.8), ("Ubuntu 24.04", 0.5),
        ("Debian 12", 0.3), ("Fedora 39", 0.2), ("Kali Linux 2024.2", 0.1), ("Arch Linux", 0.1)
    ]

    # CLİENT'LAR
    clients = [
        ("KULLANICI PORTALI", 20), ("GÜVENLİ GİRİŞ SİSTEMİ", 18), 
        ("DERS KAYIT SİSTEMİ", 12), ("ÖĞRENCİ BİLGİ SİSTEMİ", 10),
        ("AKADEMİK PORTAL", 8), ("PERSONEL YÖNETİM SİSTEMİ", 7),
        ("LİSANSLI UYGULAMALAR YÖNETİM SİSTEMİ", 5), ("RAPORLAMA MODÜLÜ", 4),
        ("VERİ ANALİZ PANELİ", 3), ("FATURA YÖNETİMİ", 3),
        ("KULLANICI RAPORLARI", 2.5), ("İSTATİSTİK MERKEZİ", 2),
        ("DESTEK MODÜLÜ", 1.5), ("KULLANICI YÖNETİMİ", 1),
        ("E-POSTA SİSTEMİ", 1), ("BELGE PAYLAŞIM MODÜLÜ", 0.8),
        ("SUNUCU GÖZLEM PANELİ", 0.5), ("VERİ KORUMA MODÜLÜ", 0.4),
        ("API KONTROL PANELİ", 0.2), ("DOSYA YÜKLEME SERVİSİ", 0.1)
    ]

    browser_names, browser_weights = zip(*browsers)
    os_names, os_weights = zip(*os_list)
    client_names, client_weights = zip(*clients)

   # 5 YILLIK HAFTALIK AKTİVİTE PATTERNİ (Her 66 haftada bir tekrar eder)
    weekly_activity = [
        15,17,23,28,33,46,63,157,107,138,  # Ağu-Eyl dönemi (Hafta 1-10)
        103,97,101,86,99,84,74,96,102,108, # Eki-Ara dönemi (Hafta 11-20)
        115,117,135,133,84,60,64,148,106,135, # Ara-Şub dönemi (Hafta 21-30)
        104,98,102,99,92,79,101,74,91,40,  # Mar-Nis dönemi (Hafta 31-40)
        97,109,126,134,97,77,111,83,112,76, # May-Tem dönemi (Hafta 41-50)
        64,69,62,55,63,30,29,49,66,155,    # Tem-Eyl dönemi (Hafta 51-60)
        108,136,102,98,100,87              # Eyl-Eki dönemi (Hafta 61-66)
]


    # SAATLİK DAĞILIM (Normal dönem - hafta içi)
    hourly_weights_normal = [
        8,3,3,3,3,3,12,35,100,100,100,75,  # 00-11
        65,95,95,95,85,85,85,60,60,40,20,8 # 12-23
    ]

    # SAATLİK DAĞILIM (Kayıt/Sınav dönemi)
    hourly_weights_peak = [
        15,8,5,5,5,8,25,80,150,130,130,110, # 00-11
        100,120,120,120,110,100,100,90,80,60,40,25 # 12-23
    ]

    # HAFTALIK DAĞILIM (Pazartesi=0, Pazar=6)
    weekday_multipliers_normal = [1.0, 1.05, 1.03, 0.98, 0.85, 0.38, 0.33]  # Normal dönem
    weekday_multipliers_peak = [1.0, 1.0, 1.0, 1.0, 0.95, 0.70, 0.65]      # Kayıt/Sınav

    # KULLANICI HAVUZU OLUŞTUR
    print("🔨 1. Kullanıcı havuzu oluşturuluyor...")
    users = []
    
    for i in range(num_users):
        name = random.choice(first_names)
        surname = random.choice(last_names)
        user_id = i + 1
        email = f"{name.lower()}.{surname.lower()}{user_id}@bogazici.edu.tr"
        
        users.append({
            "email": email,
            "name": name,
            "surname": surname,
            "tckn": f"{random.randint(10,99)}x{random.randint(10,99)}x{random.randint(10,99)}x",
            "primary_os": random.choices(os_names, weights=os_weights)[0],
            "primary_browser": random.choices(browser_names, weights=browser_weights)[0],
            "primary_client": random.choices(client_names, weights=client_weights)[0],
            "primary_ip": f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}",
        })
        
        if (i + 1) % 100_000 == 0:
            print(f"   ✓ {i + 1:,} kullanıcı oluşturuldu")
    
    print(f"✅ {num_users:,} kullanıcı hazır!\n")

    # ANOMALİ KULLANICILARI BELİRLE
    num_anomalous = int(num_users * anomaly_rate)
    anomalous_emails = set(random.sample([u["email"] for u in users], num_anomalous))
    print(f"⚠️ {num_anomalous:,} anormal kullanıcı belirlendi\n")

    random.shuffle(users)

    # LOGİN KAYITLARI OLUŞTUR
    print("🔨 2. 5 yıllık login kayıtları oluşturuluyor...")
    output_file = "2login_data_5years_10M.jsonl"
    
    start_date = datetime(2020, 8, 1)   # 1 Ağustos 2020 (Akademik yıl başı)
    end_date = datetime(2025, 8, 1)    # 1 ağustos 2025 (TAM 5 YIL)
    
    total_weeks = int((end_date - start_date).days / 7) + 1
    
    with open(output_file, "w", encoding="utf-8") as f:
        login_count = 0
        anomaly_count = 0
        current_week_start = start_date
        user_index = 0
        
        for week_num in range(total_weeks):
            if login_count >= total_logins:
                break
            
            # Haftalık aktivite oranını al (66 haftalık döngü)
            activity_percent = weekly_activity[week_num % len(weekly_activity)]
            
            # Bu haftanın login hedefini hesapla
            base_weekly = (total_logins / total_weeks) * (activity_percent / 100)
            weekly_target = int(base_weekly * random.uniform(0.95, 1.05))
            weekly_target = min(weekly_target, total_logins - login_count)
            
            # Yüksek aktivite haftası mı?
            is_peak_week = activity_percent >= 120
            
            # Haftalık kullanıcı login sayaçları (anomali tespiti için)
            user_login_tracker = {}
            
            for day_offset in range(7):
                if login_count >= total_logins:
                    break
                
                current_date = current_week_start + timedelta(days=day_offset)
                weekday = current_date.weekday()
                
                # Günlük login sayısını hesapla
                weekday_mult = weekday_multipliers_peak[weekday] if is_peak_week else weekday_multipliers_normal[weekday]
                daily_target = int((weekly_target / 7) * weekday_mult)
                
                # Saatlik ağırlıklar
                hour_weights = hourly_weights_peak if is_peak_week else hourly_weights_normal
                
                for _ in range(daily_target):
                    if login_count >= total_logins:
                        break
                    
                    # Kullanıcı seç
                    user = users[user_index % len(users)]
                    user_index += 1
                    
                    is_anomalous_user = user["email"] in anomalous_emails
                    
                    # Kullanıcı günlük login sayacı
                    user_day_key = f"{user['email']}_{current_date.date()}"
                    if user_day_key not in user_login_tracker:
                        user_login_tracker[user_day_key] = {
                            "count": 0,
                            "ips": set(),
                            "clients": set(),
                            "browsers": set(),
                            "hours": []
                        }
                    
                    # ANOMALİ ENJEKSİYONU
                    if is_anomalous_user and random.random() < 0.15:  # %15 anomali şansı
                        anomaly_type = random.choice([
                            "daily_burst",      # Günde çok fazla login (8-15 kez)
                            "night_activity",   # Gece saatlerinde login (02-05)
                            "rapid_succession", # Çok hızlı ardışık login (dk içinde)
                            "client_hopping",   # Çok fazla farklı client (4-6 farklı)
                            "ip_hopping",       # Çok fazla farklı IP (3-5 farklı)
                            "browser_spam"      # Farklı browserlar (3-4 farklı)
                        ])
                        
                        if anomaly_type == "daily_burst":
                            burst_count = random.randint(8, 15)
                            for _ in range(burst_count):
                                hour = random.choices(range(24), weights=hour_weights)[0]
                                minute = random.randint(0, 59)
                                second = random.randint(0, 59)
                                microsecond = random.randint(0, 999000)
                                login_time = current_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
                                
                                os = user["primary_os"] if random.random() < 0.7 else random.choices(os_names, weights=os_weights)[0]
                                browser = user["primary_browser"] if random.random() < 0.7 else random.choices(browser_names, weights=browser_weights)[0]
                                client = user["primary_client"] if random.random() < 0.6 else random.choices(client_names, weights=client_weights)[0]
                                ip = user["primary_ip"] if random.random() < 0.5 else f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}"
                                
                                user_login_tracker[user_day_key]["count"] += 1
                                user_login_tracker[user_day_key]["ips"].add(ip)
                                user_login_tracker[user_day_key]["clients"].add(client)
                                user_login_tracker[user_day_key]["browsers"].add(browser)
                                user_login_tracker[user_day_key]["hours"].append(hour)
                                
                                record = {
                                    "key": user["email"], "email": user["email"], "name": user["name"],
                                    "surname": user["surname"], "tckn": user["tckn"], "browser": browser,
                                    "os": os, "ip": ip, 
                                    "loginTime": login_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    "registrationId": None, "clientId": uuid.uuid4().hex,
                                    "clientName": client, "sessionId": str(uuid.uuid4())
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                login_count += 1
                                anomaly_count += 1
                        
                        elif anomaly_type == "night_activity":
                            night_count = random.randint(4, 8)
                            for _ in range(night_count):
                                hour = random.randint(2, 5)  # Gece 02-05 arası
                                minute = random.randint(0, 59)
                                second = random.randint(0, 59)
                                login_time = current_date.replace(hour=hour, minute=minute, second=second)
                                
                                user_login_tracker[user_day_key]["count"] += 1
                                user_login_tracker[user_day_key]["hours"].append(hour)
                                
                                record = {
                                    "key": user["email"], "email": user["email"], "name": user["name"],
                                    "surname": user["surname"], "tckn": user["tckn"], 
                                    "browser": user["primary_browser"], "os": user["primary_os"],
                                    "ip": user["primary_ip"],
                                    "loginTime": login_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    "registrationId": None, "clientId": uuid.uuid4().hex,
                                    "clientName": user["primary_client"], "sessionId": str(uuid.uuid4())
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                login_count += 1
                                anomaly_count += 1
                        
                        elif anomaly_type == "rapid_succession":
                            rapid_count = random.randint(6, 10)
                            base_hour = random.randint(8, 20)
                            base_minute = random.randint(0, 50)
                            for i in range(rapid_count):
                                minute = base_minute + (i // 2)  # Her 2 loginde 1 dk artış
                                second = (i % 2) * 30 + random.randint(0, 20)
                                login_time = current_date.replace(hour=base_hour, minute=minute, second=second)
                                
                                user_login_tracker[user_day_key]["count"] += 1
                                user_login_tracker[user_day_key]["hours"].append(base_hour)
                                
                                record = {
                                    "key": user["email"], "email": user["email"], "name": user["name"],
                                    "surname": user["surname"], "tckn": user["tckn"], 
                                    "browser": user["primary_browser"], "os": user["primary_os"],
                                    "ip": user["primary_ip"],
                                    "loginTime": login_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    "registrationId": None, "clientId": uuid.uuid4().hex,
                                    "clientName": user["primary_client"], "sessionId": str(uuid.uuid4())
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                login_count += 1
                                anomaly_count += 1
                        
                        elif anomaly_type == "client_hopping":
                            hop_count = random.randint(5, 8)
                            hour = random.randint(9, 18)
                            for i in range(hop_count):
                                minute = random.randint(0, 59)
                                login_time = current_date.replace(hour=hour, minute=minute, second=random.randint(0,59))
                                client = random.choices(client_names, weights=client_weights)[0]
                                
                                user_login_tracker[user_day_key]["count"] += 1
                                user_login_tracker[user_day_key]["clients"].add(client)
                                user_login_tracker[user_day_key]["hours"].append(hour)
                                
                                record = {
                                    "key": user["email"], "email": user["email"], "name": user["name"],
                                    "surname": user["surname"], "tckn": user["tckn"], 
                                    "browser": user["primary_browser"], "os": user["primary_os"],
                                    "ip": user["primary_ip"],
                                    "loginTime": login_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    "registrationId": None, "clientId": uuid.uuid4().hex,
                                    "clientName": client, "sessionId": str(uuid.uuid4())
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                login_count += 1
                                anomaly_count += 1
                        
                        elif anomaly_type == "ip_hopping":
                            ip_count = random.randint(4, 6)
                            hour = random.randint(9, 18)
                            for i in range(ip_count):
                                minute = random.randint(0, 59)
                                login_time = current_date.replace(hour=hour, minute=minute, second=random.randint(0,59))
                                ip = f"{random.choice([172,192,10])}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
                                
                                user_login_tracker[user_day_key]["count"] += 1
                                user_login_tracker[user_day_key]["ips"].add(ip)
                                user_login_tracker[user_day_key]["hours"].append(hour)
                                
                                record = {
                                    "key": user["email"], "email": user["email"], "name": user["name"],
                                    "surname": user["surname"], "tckn": user["tckn"], 
                                    "browser": user["primary_browser"], "os": user["primary_os"],
                                    "ip": ip,
                                    "loginTime": login_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    "registrationId": None, "clientId": uuid.uuid4().hex,
                                    "clientName": user["primary_client"], "sessionId": str(uuid.uuid4())
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                login_count += 1
                                anomaly_count += 1
                        
                        elif anomaly_type == "browser_spam":
                            browser_count = random.randint(4, 6)
                            hour = random.randint(10, 17)
                            for i in range(browser_count):
                                minute = random.randint(0, 59)
                                login_time = current_date.replace(hour=hour, minute=minute, second=random.randint(0,59))
                                browser = random.choices(browser_names, weights=browser_weights)[0]
                                os = random.choices(os_names, weights=os_weights)[0]
                                
                                user_login_tracker[user_day_key]["count"] += 1
                                user_login_tracker[user_day_key]["browsers"].add(browser)
                                user_login_tracker[user_day_key]["hours"].append(hour)
                                
                                record = {
                                    "key": user["email"], "email": user["email"], "name": user["name"],
                                    "surname": user["surname"], "tckn": user["tckn"], 
                                    "browser": browser, "os": os,
                                    "ip": user["primary_ip"],
                                    "loginTime": login_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    "registrationId": None, "clientId": uuid.uuid4().hex,
                                    "clientName": user["primary_client"], "sessionId": str(uuid.uuid4())
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                login_count += 1
                                anomaly_count += 1
                    
                    else:
                        # NORMAL LOGIN
                        hour = random.choices(range(24), weights=hour_weights)[0]
                        minute = random.randint(0, 59)
                        second = random.randint(0, 59)
                        microsecond = random.randint(0, 999000)
                        login_time = current_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
                        
                        # %80 primary, %20 farklı
                        if random.random() < 0.80:
                            os = user["primary_os"]
                            browser = user["primary_browser"]
                            client = user["primary_client"]
                            ip = user["primary_ip"]
                        else:
                            os = random.choices(os_names, weights=os_weights)[0]
                            browser = random.choices(browser_names, weights=browser_weights)[0]
                            client = random.choices(client_names, weights=client_weights)[0]
                            ip = f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}"
                        
                        user_login_tracker[user_day_key]["count"] += 1
                        user_login_tracker[user_day_key]["ips"].add(ip)
                        user_login_tracker[user_day_key]["clients"].add(client)
                        user_login_tracker[user_day_key]["browsers"].add(browser)
                        user_login_tracker[user_day_key]["hours"].append(hour)
                        
                        record = {
                            "key": user["email"], "email": user["email"], "name": user["name"],
                            "surname": user["surname"], "tckn": user["tckn"], "browser": browser,
                            "os": os, "ip": ip, 
                            "loginTime": login_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            "registrationId": None, "clientId": uuid.uuid4().hex,
                            "clientName": client, "sessionId": str(uuid.uuid4())
                        }
                        
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        login_count += 1
            
            current_week_start += timedelta(days=7)
            
            if (week_num + 1) % 10 == 0:
                print(f"   ✓ Hafta {week_num+1}/{total_weeks} | Toplam: {login_count:,} login | Anormal: {anomaly_count:,}")
    
    print(f"\n✅ {login_count:,} login kaydı oluşturuldu!")
    print(f"⚠️ Anormal login: {anomaly_count:,} (%{anomaly_count/login_count*100:.2f})")
    print(f"📁 Dosya: {output_file}")
    print("="*70)
    print("\n📊 ANOMALİ ÖZELLİKLERİ:")
    print("   • daily_burst: Günde 8-15 kez login")
    print("   • night_activity: Gece 02-05 arası login")
    print("   • rapid_succession: 10 dk içinde 6-10 login")
    print("   • client_hopping: 5-8 farklı client")
    print("   • ip_hopping: 4-6 farklı IP")
    print("   • browser_spam: 4-6 farklı browser")
    print("="*70)

if __name__ == "__main__":
    generate_realistic_5year_data(
        total_logins=10_000_000,
        num_users=500_000,
        anomaly_rate=0.08  # %8 anormal kullanıcı
    )