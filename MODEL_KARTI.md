# Model kartı: Trafo v3 yağ sıcaklığı tahmincisi

## Ne yapar

Bir dağıtım trafosunun son 7 günlük saatlik ölçümünden (6 yük sütunu + yağ sıcaklığı, saat ve gün kodlaması) yağ sıcaklığının 1, 6 ve 24 saat sonraki değerini tahmin eder. Her ufuk için %5, %50 ve %95 yüzdeliklerini verir.

**Amaçlanan kullanım:** Beklenmeyen ısınmayı erken fark etmek için "beklenen" sıcaklık ve aralığını vermek.

**Amaçlanmayan kullanım:**
- Koruma rölesi yerine geçmez.
- Gerçek arıza verisiyle doğrulanmadı.
- Ortam sıcaklığı olmadan mevsim geçişlerinde sapabilir.

## Modeller

| | ETTh1 | ETTh2 |
|---|---|---|
| Sürüm | 20261026-164645 (v2'den) | 20261109-231131 |
| Mimari | LSTM 2 katman x 32, dropout 0,3 | aynı |
| Hedef | OT(t+h) - OT(t), train std ile ölçekli | aynı |
| Girdi | 168 saat x 11 özellik, OT pencerenin son değerine göre göreli | aynı |
| Kayıp | pinball (yüzdelik kaybı), %5 / %50 / %95 | aynı |
| Eğitim | 12 ay train, 4 ay val (erken durdurma, kalibrasyon, kural seçimi), 4 ay test | aynı |
| Parametre | 15.561 | aynı |
| Boyut | 64 KB (pt), 65 KB (ONNX fp32), 26 KB (ONNX int8) | aynı |

## Kalibrasyon

Kalibrasyon için CQR (conformalized quantile regression) kullanıldı.

- **%90 iki yönlü aralık:** Val'deki skor `max(alt - y, y - üst)`. Bu skorun sonlu örnek düzeltmeli yüzdeliği kadar kaydırılır.
- **Alarm sınırı:** Tek yönlü %99 üst sınır. Skor `y - üst`.

Val kapsaması her ufukta %90,1. Test kapsaması tablolarda.

## Ölçümler

Test dönemi, PyTorch CPU ile ölçüldü. Ayrıntılı tablolar README'de.

| | ETTh1 | ETTh2 |
|---|---|---|
| MAE 1 / 6 / 24 sa (°C) | 0,43 / 1,15 / 1,52 | 0,60 / 2,63 / 2,73 |
| %90 kapsama 1 / 6 / 24 sa | %96 / %95 / %97 | %93 / %91 / %96 |
| 3 seed MAE 24 sa aralığı | 1,51 - 1,55 | 2,73 - 3,67 |
| İzlemede kuralı | CUSUM k 0,5, H 8, 6 sa | CUSUM k 0,25, H 24, 6 sa |
| Alarm kuralı | CUSUM k 1, H 12, 24 sa | 3 saat art arda %99 sınır aşımı, 24 sa |

## Bilinen zayıflıklar

- **ETTh1'de nokta tahmin "son değer"le aynı düzeyde.** Katkı, aralıklarda ve alarmda.
- **ETTh2'de 24 saatlik tahmin seed'e duyarlı.** Ortalaması da çıtanın gerisinde.
- **ETTh2'de aralıklar geniş.** 6 saatte 11,7 °C. Bu yüzden 3 °C'lik arıza tespiti %33'te kalıyor.
- **Model yeni sıcaklık seviyesine bir günde uyum sağlıyor.** Girdi göreli olduğu için uzun süren bir arıza bir süre sonra "normal" görünür. Panel bu yüzden uyarıyı 24 saat tutar.
- **int8 nicemleme sonrası alarm kuralı yeniden seçilmeli.** Seçilmezse 3 °C tespit %75'ten %50'ye düşüyor.

## İzleme

Haftalık olarak üç gösterge izlenir: aralık kapsaması, artık ortalaması ve yük kayması. Eşikler val'den belirlendi (bkz. `models/*/latest.json` → `monitoring`).

- Kendi test dönemlerinde bayrak kalkmadı.
- Başka trafonun verisinde 17 haftanın 17'sinde bayrak kalktı.

## Veri

ETDataset ETTh1 ve ETTh2: Çin'de iki ilçedeki iki trafonun Temmuz 2016 - Haziran 2018 dönemine ait saatlik ölçümleri. Lisans CC BY-ND 4.0.
