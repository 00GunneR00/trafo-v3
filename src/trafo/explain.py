"""İki seviyeli uyarı ve kural tabanlı açıklama, ve değerlendirmesi.

Seviyeler (KARARLAR.md, 5):
  izlemede : hassas kural (meta["alarm_rule"], trafo başına haftada <= 1 yanlış alarm; v2 ile aynı bütçe)
  alarm    : filo kuralı (meta["alarm_rule_fleet"], trafo başına haftada <= 0,25; 8 trafoda ~2/hafta)
Neden iki seviye: tek seviyede ya filo sürekli kırmızıdır (8 trafo x 1/hafta) ya da küçük arızalar
kaçar (filo bütçesiyle 3 °C kademeli arızada tespit %10-20). Ölçümler reports/alarm_*.json'da.

Model "neden" bilmez; açıklama operatörün ilk soracağı sorulara sayıyla cevap verir:
  - Beklenen ne, ölçülen ne? (ortanca tahmin, %90 aralık, fark)
  - Son 24 saatte kaç saat aralığın üstündeydi?
  - Yük olağan mı? (son 6 saatin toplam yükü, önceki 7 günün dağılımına göre z)
  - Hassas kuralın CUSUM birikimi
Uyarı varken etiket: yük olağansa "sogutma_suphesi", yük yüksekse (z >= 2) "yuk_kaynakli".

Değerlendirme (python -m trafo.explain --dataset ETTh1): test döneminde yapay soğutma arızaları ve
yük artışı (negatif kontrol) uygulanır; her seviyenin uyarı oranı ve ilk uyarıdaki etiket sayılır.
Sınır: yük senaryosundaki sıcaklık tepkisi modelden üretilir (scenarios.inject_load), gerçek bir
trafonun tepkisi farklı olabilir.
"""
import argparse
import json

import numpy as np

from .alarm import alarm_fn, cusum_path, signals
from .data import DATASETS, HORIZONS, REPORTS, TARGET, TEST_END, VAL_END
from .evaluate_alarm import WARMUP
from .forecaster import Forecaster
from .scenarios import FAULT_HOURS, Scenario, apply

USEFUL_LOAD = ["HUFL", "MUFL", "LUFL"]
LOAD_Z_HIGH = 2.0
LOAD_RECENT_H = 6
LOAD_REF_H = 24 * 7
LABELS = {"sogutma_suphesi": "Yükle açıklanamayan ısınma: soğutma sorunu olabilir",
          "yuk_kaynakli": "Isınma yük artışıyla birlikte: önce yükü kontrol et",
          "normal": "Beklenen aralıkta"}


def rules(meta: dict) -> dict:
    strip = lambda r: {k: v for k, v in r.items() if k != "test"}
    out = {"izlemede": strip(meta["alarm_rule"])}
    if "alarm_rule_fleet" in meta:
        out["alarm"] = strip(meta["alarm_rule_fleet"])
    return out


def load_z(df, s: int) -> float:
    """s saatinde son 6 saatin ortalama toplam yükü, önceki 7 günün AYNI saatlerine göre kaç sigma.

    Aynı saatlerle kıyas günlük döngüyü çıkarır (ilk sürüm 7 günün tamamıyla kıyaslıyordu; günlük
    döngü sigmayı şişirdiği için yük senaryolarında z nadiren 2'yi geçti, bkz. KARARLAR.md).
    """
    total = df[USEFUL_LOAD].sum(axis=1).to_numpy(np.float32)
    recent = total[s - LOAD_RECENT_H + 1:s + 1].mean()
    ref = np.array([total[s - 24 * d - LOAD_RECENT_H + 1:s - 24 * d + 1].mean() for d in range(1, 8)])
    return float((recent - ref.mean()) / (ref.std() + 1e-6))


def fired(rule: dict, sig: dict, k: int) -> np.ndarray:
    """[0, k] saatlerinde kuralın alarm verdiği saatler."""
    part = {key: v[:k + 1] for key, v in sig.items()}
    return alarm_fn(rule)(part, HORIZONS.index(rule["horizon_h"]))


def explain_at(fc: Forecaster, df, s: int, levels: dict, sig=None, s0: int | None = None, hold_h: int = 6) -> dict:
    """s saatinin durumu ve açıklaması. levels: rules(meta). sig verilirse (s0'dan başlayan) yeniden hesaplanmaz.

    hold_h: son kaç saatteki alarm "aktif" sayılır. Değerlendirmede 6; panelde 24 (CUSUM alarmdan sonra
    sıfırlanır ve model yeni seviyeye bir günde uyum sağlar; uyarı operatör görene kadar tutulmalı).
    """
    if sig is None:
        s0 = s - 168
        sig = signals(fc, df, s0, s + 1)
    k = s - s0
    lo = max(0, k - hold_h + 1)
    active = {}
    for name, rule in levels.items():
        hits = np.flatnonzero(fired(rule, sig, k)[lo:])
        if len(hits):
            active[name] = lo + int(hits[0])
    status = "alarm" if "alarm" in active else ("izlemede" if "izlemede" in active else "normal")

    warn = levels["izlemede"]
    i = HORIZONS.index(warn["horizon_h"])
    if status != "normal":
        lz = load_z(df, active[status] + s0)          # uyarının başladığı andaki yük
        lab = "yuk_kaynakli" if lz >= LOAD_Z_HIGH else "sogutma_suphesi"
    else:
        lz, lab = load_z(df, s), "normal"
    cusum = float(cusum_path(sig["z"][:k + 1, i], warn["k"], warn["H"])[0][k]) if warn["kind"] == "cusum" else None
    ot = df[TARGET].to_numpy(np.float32)
    recent = slice(max(0, k - 23), k + 1)
    return {
        "status": status,
        "label": lab,
        "text": LABELS[lab],
        "measured_c": round(float(ot[s]), 2),
        "expected_c": round(float(sig["median"][k, i]), 2),
        "interval_c": [round(float(sig["lower"][k, i]), 2), round(float(sig["upper"][k, i]), 2)],
        "diff_c": round(float(ot[s] - sig["median"][k, i]), 2),
        "hours_above_interval_24h": int((ot[s0:s + 1][recent] > sig["upper"][recent, i]).sum()),
        "load_z": round(lz, 2),
        "cusum": None if cusum is None else round(cusum, 2),
        "cusum_threshold": warn.get("H"),
        "horizon_h": warn["horizon_h"],
        "alarm": status == "alarm",
        "warning_age_h": {n: int(k - v) for n, v in active.items()},
    }


def scenario_outcome(fc, df, sc: Scenario, levels: dict) -> dict:
    """Senaryo boyunca her seviyenin ilk uyarısı ve hassas seviyedeki etiket."""
    out = apply(df, sc, fc)
    s0 = sc.start - WARMUP
    sig = signals(fc, out, s0, sc.start + sc.hours)
    res = {}
    for name, rule in levels.items():
        hits = np.flatnonzero(fired(rule, sig, len(sig["z"]) - 1)[WARMUP:])
        res[name] = int(hits[0]) if len(hits) else None
    lab = None
    if res["izlemede"] is not None:
        lab = "yuk_kaynakli" if load_z(out, sc.start + res["izlemede"]) >= LOAD_Z_HIGH else "sogutma_suphesi"
    return {"delay": res, "label": lab}


CASES = [("sogutma", 3.0), ("sogutma", 5.0), ("ani", 5.0), ("yuk", 0.3), ("yuk", 0.5), ("temiz", 0.0)]


def evaluate(dataset: str, n: int = 40) -> dict:
    from .data import load_raw
    df = load_raw(dataset)
    fc = Forecaster(dataset)
    levels = rules(fc.meta)
    rng = np.random.default_rng(2)
    starts = rng.integers(VAL_END + WARMUP + LOAD_REF_H, TEST_END - FAULT_HOURS, n)
    report = {}
    for kind, m in CASES:
        sc_kind = "sogutma" if kind == "temiz" else kind     # temiz: aynı pencerelerde 0 büyüklük
        res = [scenario_outcome(fc, df, Scenario(sc_kind, int(t0), m), levels) for t0 in starts]
        key = f"{kind}_{m:g}"
        entry = {"n": n}
        for name in levels:
            d = [r["delay"][name] for r in res if r["delay"][name] is not None]
            entry[name] = {"rate_72h": len(d) / n, "median_delay_h": float(np.median(d)) if d else None}
        entry["labels_at_first_warning"] = {lab: sum(r["label"] == lab for r in res)
                                            for lab in ["sogutma_suphesi", "yuk_kaynakli"]}
        report[key] = entry
        print(dataset, key, {k: v for k, v in entry.items() if k != "n"})
    return {"dataset": dataset, "levels": levels, "load_z_high": LOAD_Z_HIGH, "cases": report,
            "note": "yuk_* negatif kontroldür (arıza yok); temiz_0 aynı pencerelerde senaryosuz taban orandır. "
                    "Oranlar 72 saatte en az bir uyarıyı sayar, taban oran dahil."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    d = p.parse_args().dataset
    rep = evaluate(d)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"explain_{d}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
