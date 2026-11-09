# OASIS sitesi · proje sayfası metni (taslak)

Bu dosyadaki sayıların hepsi `reports/` ve README'den alındı. Değişirse oradan güncelleyin.

## Kart (proje listesinde)

**Trafo Filo İzleme**
Yağ sıcaklığını 1-24 saat ileri tahmin eden, beklenmeyen ısınmayı iki seviyede uyaran ve uyarının nedenini sayılarla gösteren uçtan uca sistem. Model telefonda çalışacak boyutta: 26 KB.

Etiketler: Zaman serisi · Aralık tahmini · Anomali tespiti · MLOps · Edge

## Sayfa

### Sorun

Bir dağıtım trafosunun yağ sıcaklığı yükle birlikte iner çıkar. Bu yüzden "sıcaklık 60 °C'yi geçti mi" gibi sabit bir eşik geç kalır.
- Soğutma bozulduğunda sıcaklık, o anki yükte olması gerekenden yüksek seyreder.
- Bunu görmek için önce "şu an kaç derece olmalı" sorusuna güvenilir bir cevap gerekir.

### Ne yaptım

- **Tahmin:** LSTM modeli son 7 günün yük ve sıcaklığından 1, 6 ve 24 saat sonrasını tahmin ediyor.
  - Tek sayı değil, %90 aralık veriyor.
  - Aralıklar ayrı bir doğrulama döneminde kalibre edildi (conformal yöntem). Testte kapsama %91-97.
- **İki seviyeli uyarı:** Tek eşikle 8 trafolu filo haftada ~8 yanlış alarm üretiyordu. Ya da küçük arızaların %80-90'ı kaçıyordu.
  - Hassas kural sarı "izlemede" verir.
  - Sıkı kural kırmızı "alarm" verir ve en yakın ekibi önerir.
- **Açıklama:** Her uyarıda beklenen ve ölçülen sıcaklık, aralığın üstünde geçen saat ve yükün olağan olup olmadığı gösterilir.
  - Yük olağansa "soğutma şüphesi", yük yüksekse "yük kaynaklı" denir.
- **Model sağlığı:** Haftalık olarak aralık kapsaması, sistematik sapma ve girdi kayması izlenir.
  - Model, başka bir trafonun verisine uygulandığında 17 haftanın 17'sinde uyarı verdi.
  - Kendi verisinde hiç uyarı vermedi.
- **Edge:** Model ONNX'e aktarıldı ve int8'e nicemlendi.
  - Boyut 65 KB'tan 26 KB'a indi, hata değişmedi.
  - Dizüstü CPU'da tek iş parçacığıyla 0,2 ms.
  - Küçültülmüş modelin alarm kuralı yeniden seçilmezse tespitin %75'ten %50'ye düştüğünü ölçtüm ve düzelttim.
- **Servis ve panel:** FastAPI servisi (PyTorch'suz, ONNX ile), Next.js filo paneli, Docker Compose.
  - Telefon ölçümü için ayrıca bir Flutter uygulaması var.

### Sonuçlar

Test dönemi: son 4 ay, eğitimde hiç görülmedi.

| | Trafo 1 | Trafo 2 |
|---|---|---|
| 6 saat sonrası ortalama hata | 1,15 °C | 2,63 °C |
| %90 aralığın gerçek kapsaması (6 sa) | %95 | %91 |
| 3 °C'lik kademeli ısınmayı yakalama (sarı) | %75, ortanca 18 saatte | %33 |
| Yanlış uyarı (sarı) | haftada 0,99 | haftada 0,76 |
| 5 °C ani ısınmayı yakalama (sarı) | %100, ortanca 3,5 saatte | %70 |

**Trafo 2'de tespit neden zayıf?** O trafonun sıcaklığı daha az tahmin edilebilir: 6 saatlik aralık 11,7 °C genişliğinde. 3 °C'lik ısınma bu belirsizliğin içinde kalıyor. Bunu gizlemek yerine raporladım.

### Dürüstlük notu

- Veri açık veri setidir (ETDataset, iki gerçek trafo).
- Filodaki 8 trafo bu iki seriden zaman kaydırılarak üretilmiş bir simülasyondur.
- Arızalar yapay olarak eklenir. Gerçek arıza etiketiyle doğrulanmamıştır.
- Ortam sıcaklığı verisi yoktur.

### Teknolojiler

Python · PyTorch · ONNX Runtime · MLflow · FastAPI · Next.js · Recharts · Flutter · Docker · pytest

### Görseller

- `media/01_filo_alarm.jpg`: filo görünümü; bir trafo alarmda ve en yakın ekip çizgiyle gösteriliyor.
- `media/02_detay_grafik.jpg`: son 72 saat ölçülen/beklenen sıcaklık, %90 aralık ve 24 saat ileri tahmin.
- `media/03_aciklama_saglik.jpg`: açıklama kartı, model sağlığı ve senaryo paneli.
