"""Model sağlığı izleme: modelin hâlâ güvenilir olup olmadığını haftalık pencerelerle ölçer.

Üç gösterge (her biri son 7 gün):
  coverage : %90 aralığın (6 saat ufuk) gerçek kapsama oranı
  bias_z   : ölçekli artıkların ortalaması (sürekli üstte/altta kalma)
  load_shift: yük sütunlarının haftalık ortalamasının train ortalamasından uzaklığı (train std
             biriminde, sütunlar üzerinden ortalama). Girdi kayması göstergesi.
             Not: PSI denendi; 7 günlük pencere bir yıllık train dağılımına göre zaten dar olduğu için
             kayma olmayan haftalarda bile 2-9 çıktı, ayırt edici değildi (KARARLAR.md).
Eşikler val'deki haftalık değerlerden belirlenir (val'in en kötü haftası + pay); test hiç görülmez.

Deney (python -m trafo.monitoring):
  1. Her modelin kendi test döneminde kaç hafta bayrak kalkıyor.
  2. Bilinen bir kayma: ETTh1 modeli ETTh2 verisine uygulanır (başka trafo, başka dağılım).
     İzleme bunu yakalamalı.
"""
import json

import numpy as np

from .alarm import signals
from .data import DATASETS, LOAD_COLS, REPORTS, TEST_END, TRAIN_END, VAL_END, load_raw
from .forecaster import Forecaster, save_meta

WEEK = 24 * 7
H_IDX = 1          # 6 saat ufuk
MARGIN = {"coverage": 0.05, "bias_z": 0.25, "load_shift": 0.25}


def load_shift(train_df, cur) -> float:
    mu, sd = train_df[LOAD_COLS].mean(), train_df[LOAD_COLS].std()
    return float(((cur[LOAD_COLS].mean() - mu).abs() / sd).mean())


def weekly_stats(fc: Forecaster, df, start: int, end: int, train_df) -> list[dict]:
    sig = signals(fc, df, start, end)
    ot = df["OT"].to_numpy(np.float32)[start:end]
    inside = (ot >= sig["lower"][:, H_IDX]) & (ot <= sig["upper"][:, H_IDX])
    out = []
    for w0 in range(0, end - start - WEEK + 1, WEEK):
        sl = slice(w0, w0 + WEEK)
        cur = df.iloc[start + w0:start + w0 + WEEK]
        out.append({
            "start": str(df.index[start + w0]),
            "coverage": float(inside[sl].mean()),
            "bias_z": float(sig["z"][sl, H_IDX].mean()),
            "load_shift": load_shift(train_df, cur),
        })
    return out


def thresholds_from_val(val_weeks: list[dict]) -> dict:
    cov = [w["coverage"] for w in val_weeks]
    bias = [abs(w["bias_z"]) for w in val_weeks]
    ls = [w["load_shift"] for w in val_weeks]
    return {"coverage_min": round(min(cov) - MARGIN["coverage"], 3),
            "abs_bias_z_max": round(max(bias) + MARGIN["bias_z"], 3),
            "load_shift_max": round(max(ls) + MARGIN["load_shift"], 3)}


def flags(week: dict, thr: dict) -> list[str]:
    f = []
    if week["coverage"] < thr["coverage_min"]:
        f.append("kapsama düştü")
    if abs(week["bias_z"]) > thr["abs_bias_z_max"]:
        f.append("sistematik sapma")
    if week["load_shift"] > thr["load_shift_max"]:
        f.append("yük dağılımı kaydı")
    return f


def status(fc: Forecaster, df, s: int, train_df) -> dict:
    """s saatinde biten son 7 günün sağlık durumu (API kullanır)."""
    w = weekly_stats(fc, df, s - WEEK + 1, s + 1, train_df)[0]
    thr = fc.meta["monitoring"]["thresholds"]
    return {**w, "thresholds": thr, "flags": flags(w, thr), "healthy": not flags(w, thr)}


def run_dataset(model_ds: str, data_ds: str, thr: dict | None = None) -> dict:
    fc = Forecaster(model_ds)
    df = load_raw(data_ds)
    train_df = load_raw(model_ds).iloc[:TRAIN_END]
    val = weekly_stats(fc, df, TRAIN_END, VAL_END, train_df)
    test = weekly_stats(fc, df, VAL_END, TEST_END, train_df)
    if thr is None:
        thr = thresholds_from_val(val)
    rate = lambda weeks: sum(bool(flags(w, thr)) for w in weeks) / len(weeks)
    return {"model": model_ds, "data": data_ds, "thresholds": thr,
            "flag_rate_val": rate(val), "flag_rate_test": rate(test),
            "test_weeks": [{**w, "flags": flags(w, thr)} for w in test]}


def main():
    report = {}
    for d in DATASETS:
        r = run_dataset(d, d)
        report[f"{d} modeli / {d} verisi"] = r
        fc = Forecaster(d)
        fc.meta["monitoring"] = {"thresholds": r["thresholds"], "window_h": WEEK, "horizon_h": 6,
                                 "test_flag_rate": r["flag_rate_test"]}
        save_meta(d, fc.meta)
        print(d, "eşikler", r["thresholds"], "bayrak oranı val", round(r["flag_rate_val"], 2),
              "test", round(r["flag_rate_test"], 2))
    thr = report["ETTh1 modeli / ETTh1 verisi"]["thresholds"]
    shift = run_dataset("ETTh1", "ETTh2", thr)
    report["ETTh1 modeli / ETTh2 verisi (bilinen kayma)"] = shift
    print("kayma deneyi: ETTh1 modeli ETTh2 verisinde, test bayrak oranı", round(shift["flag_rate_test"], 2),
          "| bayraklar:", {f: sum(f in w["flags"] for w in shift["test_weeks"])
                           for f in ["kapsama düştü", "sistematik sapma", "yük dağılımı kaydı"]})
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "monitoring.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
