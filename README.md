# Trafo v3: filo izleme, açıklamalı alarm ve model sağlığı

Dağıtım trafolarında yağ sıcaklığını 1, 6 ve 24 saat ileri, aralıklı olarak tahmin eden bir sistem. Beklenenden sapan ısınmayı iki seviyede uyarır, uyarının nedenine dair sayıları gösterir ve modelin hâlâ güvenilir olup olmadığını izler. Uçtan uca çalışır: eğitim, kalibrasyon, alarm seçimi, ONNX/int8 küçültme, servis, panel ve telefon ölçüm uygulaması.

> **Simülasyon.** Filo ekranındaki 8 trafo, iki gerçek açık veri serisinden ([ETDataset](https://github.com/zhouhaoyi/ETDataset), ETTh1 ve ETTh2, CC BY-ND 4.0) zaman kaydırılarak üretilmiştir. Arızalar yapay olarak eklenir. Yerleşim ve ekipler temsilîdir. Sistem gerçek bir şebekeye bağlı değildir.

v2'nin devamıdır. v2 tek trafo için tahmin ve alarm kuruyordu; v3 bunu ikinci bir trafoya taşır, filo ölçeğindeki sorunları (yanlış alarm bütçesi, açıklama, model sağlığı) çözer ve modeli edge'e hazırlar. Her kararın gerekçesi [KARARLAR.md](KARARLAR.md) dosyasında, model ayrıntıları [MODEL_KARTI.md](MODEL_KARTI.md) dosyasında.

## Sonuçlar (test dönemi: son 4 ay, eğitimde ve seçimde hiç görülmedi)

### Tahmin: ortalama mutlak hata, °C

| | 1 saat | 6 saat | 24 saat |
|---|---|---|---|
| ETTh1 · LSTM | **0,43** | **1,15** | 1,52 |
| ETTh1 · son değer | 0,42 | 1,07 | 1,53 |
| ETTh1 · dünün aynı saati | 1,53 | 1,53 | 1,53 |
| ETTh2 · LSTM | **0,60** | **2,63** | 2,73 |
| ETTh2 · son değer | 0,84 | 4,14 | 2,68 |
| ETTh2 · dünün aynı saati | 2,68 | 2,68 | 2,68 |

ETTh1'de nokta tahmin "son değer"le aynı düzeyde. Modelin katkısı, aşağıdaki kalibre aralıklar ve bu aralıkların alarmda kullanılması. ETTh2'de model 1 ve 6 saatte çıtaları açıkça geçiyor, 24 saatte biraz geride kalıyor. 3 seed'de ETTh2'nin 24 saat MAE'si 2,73 ile 3,67 arasında oynuyor (ETTh1'de 1,51 ile 1,55 arasında), bu yüzden o sayı vitrin sayısı değil.

### %90 aralık kapsaması (val'de CQR ile kalibre)

| | 1 saat | 6 saat | 24 saat | 6 saatte genişlik |
|---|---|---|---|---|
| ETTh1 | %96 | %95 | %97 | 5,6 °C |
| ETTh2 | %93 | %91 | %96 | 11,7 °C |

### Arıza tespiti (yapay arıza, her durum 40 deneme)

| | Yanlış uyarı / hafta | 3 °C kademeli | 5 °C kademeli | Ani 5 °C |
|---|---|---|---|---|
| ETTh1 · izlemede | 0,99 | %75 · 18 sa | %98 · 13 sa | %100 · 3,5 sa |
| ETTh1 · alarm | 0,00 | %10 | %65 · 26 sa | %70 · 12 sa |
| ETTh2 · izlemede | 0,76 | %33 · 16 sa | %45 · 29 sa | %70 · 19 sa |
| ETTh2 · alarm | 0,29 | %20 | %35 · 30 sa | %55 · 15 sa |

Gerçek arıza etiketi yok; arızalar yük değişmeden yağ sıcaklığına rampa eklenerek üretiliyor. Bu ölçüm, "beklenmeyen ısınmayı ne kadar sürede görür" sorusunu cevaplar; gerçek arızalardaki başarıyı kanıtlamaz.

**Neden iki seviye:** Tek eşikle ya 8 trafolu filo haftada ~8 yanlış alarm üretiyor (temiz veride trafo-saatlerinin %23'ü kırmızı) ya da 3 °C'lik arızaların %80-90'ı kaçıyor. Hassas kural sarı ("izlemede"), sıkı kural kırmızı ("alarm"). Temiz filoda zamanın %85'i normal, %13'ü izlemede, %1,7'si alarm.

**ETTh2'de tespit zayıf:** 6 saatlik %90 aralık 11,7 °C genişliğinde, 3 °C'lik ısınma bu belirsizliğin içinde kalıyor. Yöntem, tahmini belirsiz olan trafoda küçük arızayı göremiyor.

### Açıklama ve negatif kontrol

| ETTh1 | Hassas seviyede 72 saatte uyarı | İlk uyarıda "soğutma şüphesi" / "yük kaynaklı" |
|---|---|---|
| Soğutma arızası 3 °C | %68 | 26 / 1 |
| Ani 5 °C | %95 | 32 / 6 |
| Yük artışı %50 (arıza değil) | %43 | 10 / 7 |
| Senaryo yok (taban oran) | %38 | 15 / 0 |

Yük artışı, senaryosuz taban orana göre ek uyarı üretmiyor; ürettiği uyarıların yaklaşık yarısı da "yük kaynaklı" etiketleniyor. Sınır: yük senaryosundaki sıcaklık tepkisi modelin kendi beklentisinden üretiliyor, gerçek bir trafo farklı tepki verebilir. ETTh2'nin alarm seviyesi yük artışında ek alarm üretiyor (%17,5'ten %32,5'e); bu bir zayıflık.

### Model sağlığı izleme

| | Bayrak kalkan test haftası |
|---|---|
| ETTh1 modeli, ETTh1 verisi | 0 / 17 |
| ETTh2 modeli, ETTh2 verisi | 0 / 17 |
| ETTh1 modeli, ETTh2 verisi (bilinen kayma) | **17 / 17** |

Göstergeler: haftalık aralık kapsaması, artıkların ortalama z'si, yük ortalamasının train'den uzaklığı. Eşikler val'den belirlendi.

### Edge (dizüstü CPU, AMD Zen 3, tek iş parçacığı, batch 1, 1000 çalıştırma)

| | Boyut | Gecikme ortanca / p95 | ETTh1 MAE 6 sa | 3 °C tespit · yanlış/hafta |
|---|---|---|---|---|
| PyTorch | 64 KB | 0,57 / 0,62 ms | 1,149 | %75 · 0,99 |
| ONNX fp32 | 65 KB | 0,20 / 0,22 ms | 1,149 | %75 · 0,99 |
| ONNX int8 (fp32 kuralı) | 26 KB | 0,23 / 0,25 ms | 1,092 | %50 · 0,35 |
| ONNX int8 (kural yeniden seçildi) | 26 KB | 0,23 / 0,25 ms | 1,092 | %75 · 0,47, gecikme 25 sa |

int8 dosyayı 2,5 kat küçültüyor ama bu boyuttaki modelde dizüstünde hızlandırmıyor. Küçültülmüş modelin alarm kuralı yeniden seçilmezse tespit düşüyor. Telefon ölçümü için `mobile_bench/` uygulaması hazır; sonuç henüz yok.

## Mimari

```
ETT CSV ──► data.py ──► train.py (PyTorch, MLflow) ──► calibrate.py (CQR) ──► evaluate_alarm.py
                                    │                                            (izlemede + filo kuralı)
                                    ▼
                              edge.py ──► ONNX fp32 / int8 ──► mobile_bench (Flutter, telefon)
                                    │
                                    ▼
   fleet.py (8 sanal trafo, senaryolar) ◄── explain.py (açıklama) ◄── monitoring.py (sağlık)
                                    │
                                    ▼
                       api/main.py (FastAPI, ONNX) ──► panel/ (Next.js)
```

## Çalıştırma

Gereken: Python 3.12, Node 24. Veri ilk çalıştırmada kaynağından indirilir (depoda tutulmaz).

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
export PYTHONPATH=src

# Modeller depoda var; sıfırdan üretmek için:
python -m trafo.train --dataset ETTh2 --seed 0 --save
python -m trafo.calibrate --dataset ETTh2
python -m trafo.evaluate_alarm --dataset ETTh2            # hassas kural
python -m trafo.evaluate_alarm --dataset ETTh2 --fleet    # filo kuralı
python -m trafo.edge --dataset ETTh2
python -m trafo.monitoring
python -m trafo.explain --dataset ETTh2
python -m trafo.baselines && python -m trafo.seeds

python -m pytest -q                                       # 12 test

# Servis ve panel
uvicorn api.main:app --port 8000
cd panel && npm install && npm run dev                    # http://localhost:3000
```

Docker ile: `docker compose up --build` (panel :3000, API :8000/docs). Servis imajında PyTorch yok.

## Klasörler

| | |
|---|---|
| `src/trafo/` | veri, model, eğitim, kalibrasyon, alarm, senaryolar, açıklama, izleme, filo, edge |
| `api/` | FastAPI servisi |
| `panel/` | Next.js panel |
| `mobile_bench/` | Flutter telefon ölçüm uygulaması |
| `models/` | eğitilmiş modeller (.pt, .onnx) ve meta (`latest.json`) |
| `reports/` | tüm ölçümlerin JSON'ları |
| `tests/` | pytest |

## Sınırlar

- İki gerçek trafo var. Filo, bunlardan zaman kaydırılarak üretildi; aynı seriden gelen trafolar bağımsız değil.
- Gerçek arıza etiketi yok. Tüm tespit oranları yapay arızalarla ölçüldü.
- Ortam sıcaklığı verisi yok. Yaz-kış farkı modelin göremediği en büyük etken.
- Yük senaryosunun sıcaklık tepkisi modelin kendisinden üretiliyor; negatif kontrol bu yüzden kısmen döngüsel.
- Telefon gecikmesi henüz ölçülmedi.

## Veri ve lisans

ETDataset: Zhou vd., "Informer", AAAI 2021. CC BY-ND 4.0. Veri değiştirilmeden kullanılır; sanal trafolar ve yapay arızalar çalışma anında bellekte üretilir, dosyaya yazılmaz.
