// Trafo v3 · telefon ölçümü
//
// ETTh1 modelinin ONNX fp32 ve int8 kopyalarını telefonda çalıştırır:
//  - doğruluk kontrolü: sabit bir girdi için çıktı, dizüstündeki onnxruntime çıktısıyla kıyaslanır
//  - gecikme: tek iş parçacığı, batch 1, 50 ısınma + 1000 çalıştırma, ortanca ve p95
// Girdi gerçek veri değildir (sin dizisi); LSTM'in süresi girdinin değerine bağlı değildir.
// Sonuç JSON olarak panoya kopyalanır -> reports/edge_phone.json
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

const window = 168;
const features = 11;
const warmup = 50;
const runs = 1000;

// python: np.sin(np.arange(168*11)*0.37) girdisi için dizüstü onnxruntime 1.31 çıktısı (ölçekli, 3 ufuk x 3 yüzdelik)
const reference = {
  'fp32': [-0.156373, 0.009314, 0.155313, -0.391556, -0.03436, 0.293986, -0.619391, 0.00846, 0.438341],
  'int8': [-0.161138, 0.010506, 0.150422, -0.378357, -0.027191, 0.287432, -0.522682, 0.01185, 0.426887],
};

void main() => runApp(const BenchApp());

class BenchApp extends StatelessWidget {
  const BenchApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Trafo v3 ölçüm',
        theme: ThemeData(colorSchemeSeed: Colors.blue, brightness: Brightness.dark, useMaterial3: true),
        home: const BenchPage(),
      );
}

class BenchResult {
  BenchResult(this.name, this.sizeKb, this.medianMs, this.p95Ms, this.maxAbsDiff);
  final String name;
  final double sizeKb, medianMs, p95Ms, maxAbsDiff;

  Map<String, dynamic> toJson() => {
        'model': name,
        'size_kb': double.parse(sizeKb.toStringAsFixed(1)),
        'median_ms': double.parse(medianMs.toStringAsFixed(3)),
        'p95_ms': double.parse(p95Ms.toStringAsFixed(3)),
        'max_abs_diff_vs_laptop': maxAbsDiff,
      };
}

class BenchPage extends StatefulWidget {
  const BenchPage({super.key});

  @override
  State<BenchPage> createState() => _BenchPageState();
}

class _BenchPageState extends State<BenchPage> {
  final _results = <BenchResult>[];
  String _status = 'Hazır';
  bool _running = false;

  Float32List _input() {
    final x = Float32List(window * features);
    for (var i = 0; i < x.length; i++) {
      x[i] = math.sin(i * 0.37);
    }
    return x;
  }

  Future<BenchResult> _bench(String name) async {
    final data = await rootBundle.load('assets/models/etth1_$name.onnx');
    final bytes = data.buffer.asUint8List();
    final opts = OrtSessionOptions()..setIntraOpNumThreads(1)..setInterOpNumThreads(1);
    final session = OrtSession.fromBuffer(bytes, opts);
    final runOpts = OrtRunOptions();
    final input = OrtValueTensor.createTensorWithDataList(_input(), [1, window, features]);

    List<double> once() {
      final out = session.run(runOpts, {'x': input});
      final v = (out.first!.value as List).expand((h) => (h as List).expand((q) => (q as List).cast<double>())).toList();
      for (final o in out) {
        o?.release();
      }
      return v;
    }

    final first = once();
    final ref = reference[name]!;
    var diff = 0.0;
    for (var i = 0; i < ref.length; i++) {
      diff = math.max(diff, (first[i] - ref[i]).abs());
    }
    for (var i = 0; i < warmup; i++) {
      once();
    }
    final times = <double>[];
    final sw = Stopwatch();
    for (var i = 0; i < runs; i++) {
      sw
        ..reset()
        ..start();
      once();
      sw.stop();
      times.add(sw.elapsedMicroseconds / 1000.0);
      if (i % 100 == 0) {
        setState(() => _status = '$name: $i / $runs');
        await Future<void>.delayed(Duration.zero);
      }
    }
    times.sort();
    input.release();
    runOpts.release();
    session.release();
    opts.release();
    return BenchResult(name, bytes.length / 1024, times[times.length ~/ 2], times[(times.length * 0.95).floor()], diff);
  }

  Future<void> _run() async {
    setState(() {
      _running = true;
      _results.clear();
    });
    try {
      OrtEnv.instance.init();
      for (final name in ['fp32', 'int8']) {
        final r = await _bench(name);
        setState(() => _results.add(r));
      }
      setState(() => _status = 'Bitti. JSON panoya kopyalanabilir.');
    } catch (e) {
      setState(() => _status = 'Hata: $e');
    } finally {
      OrtEnv.instance.release();
      setState(() => _running = false);
    }
  }

  String get _json => const JsonEncoder.withIndent('  ').convert({
        'device': 'ELLE YAZIN (ör. model, işlemci)',
        'threads': 1,
        'batch': 1,
        'warmup': warmup,
        'runs': runs,
        'results': _results.map((r) => r.toJson()).toList(),
      });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Trafo v3 · telefon ölçümü')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text('ETTh1 modeli, ONNX Runtime, tek iş parçacığı, batch 1. '
              'Doğruluk: sabit girdide dizüstü çıktısına göre en büyük fark (ölçekli birim).'),
          const SizedBox(height: 16),
          FilledButton(onPressed: _running ? null : _run, child: Text(_running ? 'Ölçülüyor…' : 'Ölçümü başlat')),
          const SizedBox(height: 8),
          Text(_status),
          const SizedBox(height: 16),
          for (final r in _results)
            Card(
              child: ListTile(
                title: Text('${r.name} · ${r.sizeKb.toStringAsFixed(1)} KB'),
                subtitle: Text('ortanca ${r.medianMs.toStringAsFixed(3)} ms · p95 ${r.p95Ms.toStringAsFixed(3)} ms\n'
                    'dizüstüne göre fark ${r.maxAbsDiff.toStringAsExponential(2)}'),
                isThreeLine: true,
              ),
            ),
          if (_results.length == 2)
            OutlinedButton(
              onPressed: () async {
                await Clipboard.setData(ClipboardData(text: _json));
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('JSON panoya kopyalandı')));
                }
              },
              child: const Text('Sonucu JSON olarak kopyala'),
            ),
        ],
      ),
    );
  }
}
