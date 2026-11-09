# Kararlar

Bu dosyada projede verilen her kararın gerekçesi yer alıyor. trafo-v2'deki `KARARLAR.md` dosyasının devamı; v2'nin veri, bölme, model ve alarm kararları aynen geçerli.

## 1. Filo: gerçek iki seriden üretilen sanal trafolar (seçenek B)

*9 Kasım 2026*

- Gerçek veride sadece iki trafo var: ETTh1 ve ETTh2. ETTm1 ve ETTm2 aynı trafoların 15 dakikalık sürümleri.
- Filo ekranında bu iki gerçek seriden **zaman kaydırılarak üretilmiş sanal trafolar** gösterilecek (örneğin 8 adet).
  - Her sanal trafo, gerçek serinin farklı bir başlangıç anından oynatılır. Değerler değiştirilmez.
- Panelde, videoda ve README'de bu trafoların sanal olduğu **açıkça yazılacak**. Örnek etiket: "Simülasyon: 2 gerçek trafonun ölçümlerinden türetildi".
- **Lisans (ETT, CC BY-ND 4.0):**
  - Orijinal veri değiştirilmeden ve kaynak gösterilerek kullanılır.
  - Sanal trafolar ve yapay arızalar çalışma anında üretilir. Değiştirilmiş seri dosya olarak kaydedilmez ve paylaşılmaz.
  - Bu bir yorumdur, hukuki görüş değildir.
- **Elenen seçenekler:**
  - **A, yalnız 2 gerçek trafo:** Dürüst bir seçenek ama filo ekranı boş görünüyor.
  - **C, v1'in sentetik filosu:** Sentetik veri, portföy değerini düşürür.

## 2. Staj kurumunun adı geçmez

*9 Kasım 2026*

- Sitede, videoda, depoda ve belgelerde staj yapılan dağıtım şirketinin adı ve logosu **geçmez**. Gerekçe: müşteri ilişkisi ya da şirket verisi kullanılmış gibi bir izlenim oluşmasın.
- **Harita:**
  - Konumlar **temsilidir** ve öyle etiketlenir.
  - Belirli bir şehrin dağıtım şebekesi gibi görünen bir yerleşim kullanılmaz.
- **v1'den kod alınırken temizlenecekler:** v1'in (Trafo_ariza) kodunda şehir ve ilçe adları geçiyor. Dosyalar: `backend/teams.py`, `backend/main.py`, `ml/generate_data.py`, `frontend/src/app/page.tsx`, `frontend/src/app/layout.tsx`. Bu dosyalardan alınan kod v3'e taşınırken bu adlar nötr adlarla değiştirilecek.

## 3. İkinci trafo (ETTh2): aynı tarif, ayrı model

*9 Kasım 2026*

- ETTh2 için v2'nin tarifi değiştirilmeden kullanıldı (LSTM 2x32, girdi 168 saat, delta hedef, göreli OT, aynı bölme). Ayrı ayar yapılmadı; amaç "tarif başka trafoya taşınıyor mu" sorusunu dürüstçe cevaplamak.
- **Sonuç karışık:**
  - 1 ve 6 saatte "son değer" çıtasını geçiyor: test MAE 0,60 / 0,84 ve 2,63 / 4,14 °C.
  - 24 saatte çıtaların biraz gerisinde: 2,73 / 2,68 °C.
  - 3 seed'de 24 saat MAE 2,73 ile 3,67 arasında oynuyor. ETTh1'de bu aralık 1,51-1,55.
  - Bu yüzden ETTh2'de 24 saatlik tahmin vitrin sayısı olarak kullanılmaz; alarmda kullanılan ufuk 6 saattir.
- **ETTh2'de alarm daha zayıf:**
  - 3 °C kademeli arızada tespit %33, ETTh1'de %75.
  - Sebep: ETTh2'nin %90 aralığı 6 saatte 11,7 °C genişliğinde (ETTh1'de 5,6 °C). 3 °C'lik ısınma bu belirsizliğin içinde kalıyor.
  - Bu, yöntemin sınırıdır ve README'de böyle yazılır.

## 4. Edge: servis fp32 ONNX, telefon int8 adayı

*9 Kasım 2026*

- **ONNX fp32:**
  - PyTorch ile farkı ölçekli birimde 1e-6 mertebesinde. MAE, kapsama ve alarm sonuçları birebir aynı.
  - Dizüstü CPU'da tek iş parçacığıyla 0,20 ms. PyTorch CPU'da 0,57 ms.
  - Servis bunu kullanır; konteynerde PyTorch yoktur.
- **Dinamik int8:**
  - Dosya 65 KB'tan 26 KB'a iniyor. MAE değişmiyor, hatta biraz iyileşiyor.
  - Dizüstünde fp32'den **hızlı değil** (0,23 ms). Bu kadar küçük modelde nicemleme/geri çevirme ek yükü kazancı yiyor. Telefon ölçümü ayrıca yapılacak (mobile_bench).
- **int8 ile alarm:** fp32'nin kalibrasyonu ve alarm kuralı int8'e aynen uygulanınca ETTh1'de 3 °C tespit 0,75'ten 0,55'e düştü.
  - Val'de yeniden kalibrasyon tek başına yetmedi: 0,50.
  - Alarm kuralı da val'de yeniden seçilince tespit 0,75'e döndü (CUSUM k 0,25, H 16, ufuk 24). Yanlış alarm haftada 0,47 oldu, ortanca gecikme 25 saat (fp32'de 18).
  - Çıkarım: küçültülmüş model yeni bir modeldir; kalibrasyonu ve alarm eşiği onun için yeniden seçilmelidir.

## 5. İki seviyeli uyarı: "izlemede" ve "alarm"

*9 Kasım 2026*

- **Sorun:** v2'nin alarm bütçesi trafo başına haftada en fazla 1 yanlış alarmdı. 8 trafolu filoda bu, haftada ~8 yanlış alarm demek.
  - Panelde 48 saatlik alarm tutmayla temiz veride trafo-saatlerinin %23'ü kırmızı göründü.
- **Denenen:** Trafo başına 0,25/hafta bütçesiyle kural yeniden seçildi.
  - Yanlış alarm 0 ile 0,29 arasına indi.
  - Ama 3 °C kademeli arızada tespit %10 (ETTh1) ve %20 (ETTh2) oldu.
  - 5 °C'de %35 ile %65 arasında, ani 5 °C'de %55 ile %70 arasında.
- **Karar: iki seviye.** Bunlar tek bir eşikle aynı anda sağlanamıyor; endüstrideki uyarı/alarm ayrımı da buna karşılık geliyor.
  - **İzlemede (sarı):** v2 bütçesindeki hassas kural.
  - **Alarm (kırmızı):** filo bütçesindeki kural; ekip önerisi yalnız bu seviyede yapılır.
- **Sonuç:** Temiz filoda zamanın %85'i normal, %13'ü izlemede, %1,7'si alarm.
- **Tutma süresi:** Uyarılar panelde 24 saat tutulur.
  - CUSUM alarmdan sonra sıfırlanır. Model de yeni sıcaklık seviyesine yaklaşık bir günde uyum sağlar (girdi göreli).
  - Bu yüzden arıza sürse de sinyal kaybolur. Uyarı, operatör görene kadar tutulmalıdır.

## 6. Açıklama: kural tabanlı, yük aynı saatlerle kıyaslanır

*9 Kasım 2026*

- Model nedeni bilmez. Açıklama, operatörün ilk soracağı sayıları verir: beklenen/ölçülen, aralık, aralık üstünde geçen saat, yük durumu, CUSUM birikimi.
- **Etiket:**
  - Uyarı anında yük olağansa "soğutma şüphesi".
  - Yük z >= 2 ise "yük kaynaklı".
- **Yük z'si:** son 6 saat, önceki 7 günün **aynı saatleriyle** kıyaslanır.
  - İlk sürüm 7 günün tamamıyla kıyaslıyordu. Günlük döngü sigmayı şişirdiği için yük senaryolarında "yük kaynaklı" etiketi uyarıların yalnız ~%6'sında çıktı.
  - Aynı-saat kıyasıyla bu oran %33-61'e çıktı.
  - Bedeli: ETTh1'de ani 5 °C arızaların %16'sı "yük kaynaklı" etiketlendi.
- **Negatif kontrol:** Yük artışı senaryosunda hassas seviyenin uyarı oranı senaryosuz taban orana eşit (ETTh1: %37,5-42,5'e karşı %37,5). Yani yükle açıklanan ısınma ek uyarı üretmiyor.
  - Sınır: Senaryodaki sıcaklık tepkisi modelin kendisinden üretiliyor, gerçek bir trafo farklı tepki verebilir.
  - ETTh2'nin filo kuralı (eşik, 24 saat ufuk) yük senaryosunda ek alarm üretiyor: %17,5'ten %32,5'e. Bu bir zayıflık olarak raporlanır.

## 7. İzleme: PSI yerine standartlaştırılmış ortalama kayması

*9 Kasım 2026*

- **Göstergeler (haftalık):** %90 aralık kapsaması (6 saat ufuk), artıkların ortalama z'si, yük kayması.
- **Eşikler:** val'in en kötü haftası + pay. Test eşik seçiminde kullanılmadı.
- **PSI denendi, bırakıldı:** 7 günlük pencere, bir yıllık train dağılımına göre zaten dar. Bu yüzden kayma olmayan haftalarda bile 2-9 çıktı ve ayırt edici değildi.
- **Yerine:** yük sütunlarının haftalık ortalamasının train ortalamasından uzaklığı, train std biriminde, sütunlar üzerinden ortalama.
  - Aynı trafo: test haftalarında en fazla 1,8.
  - Başka trafo (ETTh1 modeli ETTh2 verisinde): en az 3,6.
- **Sonuç:**
  - Her modelin kendi test döneminde 17 haftanın hiçbirinde bayrak kalkmadı.
  - Bilinen kaymada 17 haftanın hepsinde kalktı (kapsama %15-66'ya düştü).

## 8. Panel v1'den kopyalanmadı

*9 Kasım 2026*

- v1'in sayfa kodunda kurum adı ve şehir adları geçiyordu. Panel yeniden yazıldı; v1'den yalnız yapılandırma dosyaları (tsconfig, eslint, postcss) ve koyu tema yaklaşımı alındı.
- Leaflet haritası yerine şematik SVG yerleşim kullanıldı. Böylece gerçek bir şehir izlenimi oluşmuyor.
