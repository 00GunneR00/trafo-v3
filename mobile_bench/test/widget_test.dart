import 'package:flutter_test/flutter_test.dart';
import 'package:trafo_bench/main.dart';

void main() {
  testWidgets('başlangıç ekranı', (tester) async {
    await tester.pumpWidget(const BenchApp());
    expect(find.text('Ölçümü başlat'), findsOneWidget);
  });
}
