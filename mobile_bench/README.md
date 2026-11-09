# mobile_bench: telefonda ölçüm

ETTh1 modelinin ONNX fp32 ve int8 kopyalarını telefonda çalıştırır.

**Doğruluk kontrolü:** Sabit bir girdinin (sin dizisi, gerçek veri değil) çıktısı, dizüstü onnxruntime çıktısıyla kıyaslanır.

**Gecikme ölçümü:** Tek iş parçacığı, batch 1, 50 ısınma + 1000 çalıştırma.

## Kurulum

USB hata ayıklama açık bir Android telefonla:

```bash
flutter run --release
# ya da hazır APK (çoğu telefon arm64):
adb install build/app/outputs/flutter-apk/app-arm64-v8a-release.apk
```

1. "Ölçümü başlat"a basın.
2. Bitince "Sonucu JSON olarak kopyala"ya basın.
3. JSON'daki `device` alanına telefon modelini yazın.
4. JSON'u `reports/edge_phone.json` olarak kaydedin.

## Bilinen risk

- Eklenti (`onnxruntime` 1.4.1) ONNX Runtime 1.15.1 içeriyor.
- fp32 modeli (opset 17, IR 8) bu sürümle uyumlu.
- int8 modeli `DynamicQuantizeLSTM` (com.microsoft) işlemini kullanıyor. Eklentinin derlemesinde bu işlem yoksa int8 ölçümü hata verir. Hata mesajı ekranda görünür.
- Telefonda henüz denenmedi.
- Eklenti compileSdk 33 ile derleniyor. `android/build.gradle.kts` dosyasında 36'ya yükseltildi.
