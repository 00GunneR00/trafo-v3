"""Edge: ONNX fp32 dışa aktarma, dinamik int8 nicemleme, doğruluk/aralık/alarm kıyası ve gecikme ölçümü.

Sonuç meta["edge"] altına yazılır; API ve panel bu bilgiyi gösterir. Gecikme bu dizüstünün
CPU'sunda, tek iş parçacığıyla, batch=1 için ölçülür (telefon ölçümü Flutter uygulamasıyla ayrıca).

  python -m trafo.edge --dataset ETTh1
"""
import argparse
import json
import platform
import time

import numpy as np

from .calibrate import evaluate, fit_calibration
from .data import DATASETS, HORIZONS, REPORTS, TARGET, TEST_END, VAL_END, WINDOW, load_raw
from .forecaster import Forecaster, model_dir, save_meta

N_RUNS = 1000
N_WARMUP = 50


def export_fp32(fc: Forecaster, path) -> None:
    import torch
    model = fc.model.cpu().eval()
    dummy = torch.zeros(1, WINDOW, fc.meta["n_features"], dtype=torch.float32)
    torch.onnx.export(model, dummy, str(path), input_names=["x"], output_names=["q"],
                      dynamic_axes={"x": {0: "batch"}, "q": {0: "batch"}}, opset_version=17, dynamo=False)


def quantize_int8(src, dst) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def latency_ms(fc: Forecaster, X: np.ndarray) -> dict:
    x = X[:1]
    for _ in range(N_WARMUP):
        fc.raw(x)
    times = []
    for i in range(N_RUNS):
        xi = X[i % len(X):i % len(X) + 1]
        t0 = time.perf_counter()
        fc.raw(xi)
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def test_mae(fc: Forecaster, df) -> dict:
    ot = df[TARGET].to_numpy(np.float32)
    out = {}
    for i, h in enumerate(HORIZONS):
        t = np.arange(VAL_END - h, TEST_END - h)
        out[f"h{h}"] = float(np.abs(fc.predict(df, t)[:, i, 1] - ot[t + h]).mean())
    return out


def int8_with_fp32_calibration(dataset, df, meta, rule) -> dict:
    """Yeniden kalibrasyonun etkisini göstermek için: int8 model + fp32'nin kaydırmaları."""
    from .evaluate_alarm import evaluate_split, score
    fc = Forecaster(dataset, "onnx-int8", meta=meta)
    fc.calibration = meta["calibration"]
    test = evaluate_split(fc, df, VAL_END, TEST_END, np.random.default_rng(1))
    cov = evaluate(fc, df, VAL_END, TEST_END)
    return {"test_coverage_90": {h: v["coverage_90"] for h, v in cov.items()},
            "alarm_test": score(test, HORIZONS.index(rule["horizon_h"]), rule)}


def run(dataset: str) -> dict:
    from .evaluate_alarm import run as alarm_run
    df = load_raw(dataset)
    d = model_dir(dataset)
    torch_fc = Forecaster(dataset, "torch", device="cpu")
    meta = torch_fc.meta
    fp32, int8 = d / f"lstm_{meta['version']}.onnx", d / f"lstm_{meta['version']}.int8.onnx"
    export_fp32(torch_fc, fp32)
    quantize_int8(fp32, int8)
    meta["edge"] = {"onnx_fp32": {"file": fp32.name}, "onnx_int8": {"file": int8.name}}
    # int8 çıktısı fp32'den biraz farklı; fp32'nin kalibrasyonu ona aynen uymaz. Aynı val verisiyle
    # yeniden kalibre edilir (alarm kuralı değişmez). Karşılaştırma için fp32 kalibrasyonlu sonuç da raporlanır.
    q = Forecaster(dataset, "onnx-int8", meta=meta)
    meta["edge"]["onnx_int8"]["calibration"] = fit_calibration(q, df)
    save_meta(dataset, meta)   # alarm değerlendirmesi meta'yı diskten okur

    rule = {k: v for k, v in meta["alarm_rule"].items() if k != "test"}
    X = torch_fc.windows(df, np.arange(VAL_END, VAL_END + 512))
    torch_fc._torch.set_num_threads(1)
    results = {"onnx-int8 (fp32 kalibrasyonu)": int8_with_fp32_calibration(dataset, df, meta, rule)}
    for backend in ["torch", "onnx", "onnx-int8"]:
        fc = torch_fc if backend == "torch" else Forecaster(dataset, backend, meta=meta)
        cov = evaluate(fc, df, VAL_END, TEST_END)
        results[backend] = {
            "size_kb": round((d / meta["file"]).stat().st_size / 1024, 1) if backend == "torch"
            else round((fp32 if backend == "onnx" else int8).stat().st_size / 1024, 1),
            "test_mae": test_mae(fc, df),
            "test_coverage_90": {h: v["coverage_90"] for h, v in cov.items()},
            "latency_cpu_1thread": latency_ms(fc, X),
            "alarm_test": alarm_run(dataset, backend, rule=rule, save=False),
        }
        if backend != "torch":
            ref = torch_fc.raw(X)
            results[backend]["max_abs_diff_vs_torch_scaled"] = float(np.abs(fc.raw(X) - ref).max())
        r = results[backend]
        print(f"{dataset} {backend:9s} {r['size_kb']:7.1f} KB  MAE {r['test_mae']}  "
              f"gecikme {r['latency_cpu_1thread']['median_ms']:.3f} ms (p95 {r['latency_cpu_1thread']['p95_ms']:.3f})  "
              f"3°C kademeli {r['alarm_test']['kademeli_3']['detection_rate']:.2f}  yanlış/hafta {r['alarm_test']['false_per_week']:.2f}")

    machine = {"cpu": platform.processor(), "system": platform.platform(), "threads": 1, "batch": 1,
               "runs": N_RUNS, "warmup": N_WARMUP}
    meta["edge"]["onnx_fp32"].update(results["onnx"])
    meta["edge"]["onnx_int8"].update(results["onnx-int8"])
    meta["edge"]["onnx_int8"]["with_fp32_calibration"] = results["onnx-int8 (fp32 kalibrasyonu)"]
    meta["edge"]["torch_cpu"] = results["torch"]
    meta["edge"]["machine"] = machine
    save_meta(dataset, meta)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"edge_{dataset}.json").write_text(json.dumps({"machine": machine, "results": results}, indent=2,
                                                             ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    run(p.parse_args().dataset)
