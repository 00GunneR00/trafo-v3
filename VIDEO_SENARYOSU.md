# Video senaryosu (~90 sn)

Ekran kaydı: panel (`docker compose up` ya da yerel), 1920x1080. Seslendirme Türkçe. Altyazı önerilir.

| Süre | Ekran | Ses / yazı |
|---|---|---|
| 0-8 | Panel açılış, 8 trafo yeşil | "Trafo yağ sıcaklığı yükle iner çıkar. Sabit eşik geç kalır. Bu sistem her trafo için 'şu an kaç derece olmalı' sorusunu cevaplıyor." |
| 8-15 | Üstteki SİMÜLASYON etiketine yakınlaş | "Veri iki gerçek trafonun açık verisi. Filodaki sekiz trafo bu iki seriden üretilmiş bir simülasyon." |
| 15-30 | TR-03'e tıkla, detay grafiği | "Kırmızı ölçülen, mavi bant modelin %90 aralığı. Testte bu aralık gerçeği %95 oranında kapsıyor. Sağda 24 saat ileri tahmin." |
| 30-45 | Senaryo: Ani ısınma 6 °C, başlat, +1 saat adımlarla ilerle | "Şimdi bu trafoya yapay bir soğutma arızası ekliyorum. Yük aynı, sıcaklık beklenenin üstüne çıkıyor." |
| 45-60 | Durum sarı, sonra kırmızı; ekip çizgisi | "Önce sarı uyarı, sonra kırmızı alarm. Neden kartı yükün olağan olduğunu söylüyor, yani 'soğutma şüphesi'. En yakın ekip öneriliyor." |
| 60-70 | Başka trafoda yük artışı senaryosu | "Aynı ısınmayı bu kez yük artışı yapıyor. Model bunu bekliyor, alarm yok." |
| 70-80 | Model sağlığı kartı ve alttaki tablo | "Model kendini de izliyor: aralık kapsaması, sapma ve girdi kayması. Başka bir trafonun verisine uygulandığında 17 haftanın 17'sinde uyarı verdi." |
| 80-90 | Edge tablosu / telefon uygulaması | "Model ONNX ve int8 ile 26 kilobayta indi, işlemcide 0,2 milisaniye. Kod, ölçümler ve tüm kararlar depoda." |

Notlar:
- Sarı ve kırmızının ne zaman geldiği senaryonun başlangıç saatine bağlı. Çekimden önce aynı saatte deneyin. Örneğin TR-03'te başlangıç 285, izleme saati 300.
- Yanlış alarm ve %33 gibi zayıf sonuçları videoda saklamayın. Sayfada "dürüstlük notu" var.
