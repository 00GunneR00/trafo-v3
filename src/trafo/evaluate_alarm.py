"""Alarm kuralının yapay arızalarla seçimi ve değerlendirmesi (v2 anomaly.py ile aynı yöntem).

1. val'de eşik ve CUSUM kurallarının parametre ızgarası denenir; haftada en fazla 1 yanlış alarm
   şartıyla 3 °C kademeli arızayı en çok yakalayan seçilir.
2. Seçilen kurallar test'te raporlanır. Kazanan kural meta["alarm_rule"] olarak kaydedilir.

  python -m trafo.evaluate_alarm --dataset ETTh2
"""
import argparse
import json

import numpy as np

from .alarm import alarm_fn, signals
from .data import DATASETS, HORIZONS, REPORTS, TEST_END, TRAIN_END, VAL_END, load_raw
from .forecaster import Forecaster, save_meta
from .scenarios import FAULT_HOURS, RAMPS, Scenario, inject_fault

MAGNITUDES = [1.0, 2.0, 3.0, 5.0]
N_INJECTIONS = 40
N_GRID = [1, 2, 3, 4, 6, 8]
CUSUM_K = [0.25, 0.5, 1.0]
CUSUM_H = [2, 4, 6, 8, 12, 16, 24]
MAX_FALSE_PER_WEEK = 1.0
FLEET_MAX_FALSE = 0.25        # 8 trafoda haftada ~2 yanlış alarm
WARMUP = 48
TYPE_NAMES = {"sogutma": "kademeli", "ani": "ani"}


def false_alarms_per_week(alarm: np.ndarray) -> float:
    onsets = alarm & ~np.concatenate([[False], alarm[:-1]])
    return float(onsets.sum() / (len(alarm) / (24 * 7)))


def evaluate_split(fc, df, start, end, rng):
    clean = signals(fc, df, start, end)
    injections = {}
    for kind in RAMPS:
        for m in MAGNITUDES:
            runs = []
            for t0 in rng.integers(start + WARMUP, end - FAULT_HOURS, N_INJECTIONS):
                faulty = inject_fault(df, Scenario(kind, int(t0), m))
                runs.append(signals(fc, faulty, int(t0) - WARMUP, int(t0) + FAULT_HOURS))
            injections[(TYPE_NAMES[kind], m)] = runs
    return {"clean": clean, "injections": injections}


def score(ev, h_idx, rule):
    fn = alarm_fn(rule)
    out = {"false_per_week": false_alarms_per_week(fn(ev["clean"], h_idx))}
    for (ftype, m), runs in ev["injections"].items():
        delays = []
        for sig in runs:
            hits = np.flatnonzero(fn(sig, h_idx)[WARMUP:])
            delays.append(int(hits[0]) if len(hits) else None)
        found = [d for d in delays if d is not None]
        out[f"{ftype}_{m:g}"] = {"detection_rate": len(found) / len(delays),
                                 "median_delay_h": float(np.median(found)) if found else None}
    return out


def rule_grid():
    for n in N_GRID:
        yield {"kind": "esik", "n": n}
    for k in CUSUM_K:
        for h_thr in CUSUM_H:
            yield {"kind": "cusum", "k": k, "H": h_thr}


def run(dataset: str, backend: str = "torch", rule: dict | None = None, save: bool = True,
        max_false: float = MAX_FALSE_PER_WEEK, meta_key: str = "alarm_rule") -> dict:
    """rule verilirse ızgara atlanır, yalnız o kural test'te ölçülür (ör. küçültülmüş model için)."""
    df = load_raw(dataset)
    fc = Forecaster(dataset, backend)
    test = evaluate_split(fc, df, VAL_END, TEST_END, np.random.default_rng(1))
    if rule is not None:
        return score(test, HORIZONS.index(rule["horizon_h"]), rule)

    val = evaluate_split(fc, df, TRAIN_END, VAL_END, np.random.default_rng(0))
    grid = [{"h": h, "rule": r, **score(val, i, r)} for i, h in enumerate(HORIZONS) for r in rule_grid()]

    def pick(kind):
        ok = [g for g in grid if g["rule"]["kind"] == kind and g["false_per_week"] <= max_false]
        return max(ok, key=lambda g: (g["kademeli_3"]["detection_rate"], -(g["kademeli_3"]["median_delay_h"] or 1e9)))

    chosen = {}
    for kind in ["esik", "cusum"]:
        best = pick(kind)
        chosen[kind] = {"horizon_h": best["h"], "rule": best["rule"],
                        "val": {k: v for k, v in best.items() if k not in ("h", "rule")},
                        "test": score(test, HORIZONS.index(best["h"]), best["rule"])}
        t = chosen[kind]["test"]
        print(f"{dataset} [{kind}] ufuk {best['h']} {best['rule']}: test yanlış/hafta {t['false_per_week']:.2f}, "
              f"3°C kademeli {t['kademeli_3']['detection_rate']:.2f}")
    winner = max(chosen, key=lambda k: chosen[k]["val"]["kademeli_3"]["detection_rate"])
    report = {"dataset": dataset, "model_version": fc.version, "chosen": chosen, "production_rule": winner,
              "selection": {"max_false_per_week": max_false, "target": "kademeli_3",
                            "note": "seçim val'de, rapor test'te"}}
    if save:
        REPORTS.mkdir(exist_ok=True)
        suffix = "" if meta_key == "alarm_rule" else "_" + meta_key.removeprefix("alarm_rule_")
        (REPORTS / f"alarm_{dataset}{suffix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        w = chosen[winner]
        fc.meta[meta_key] = {**w["rule"], "horizon_h": w["horizon_h"], "test": w["test"]}
        save_meta(dataset, fc.meta)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    p.add_argument("--fleet", action="store_true",
                   help="filo bütçesi: trafo başına haftada en fazla FLEET_MAX_FALSE yanlış alarm")
    a = p.parse_args()
    if a.fleet:
        run(a.dataset, max_false=FLEET_MAX_FALSE, meta_key="alarm_rule_fleet")
    else:
        run(a.dataset)
