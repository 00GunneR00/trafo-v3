"""Filo simülatörü: 2 gerçek seriden (ETTh1, ETTh2) 8 sanal trafo (KARARLAR.md, 1).

Her sanal trafo, kendi serisinin test döneminden farklı bir başlangıç noktasıyla oynatılır
(kaydırmalar 168 saatin katıdır, böylece filodaki tüm trafolarda saat ve haftanın günü aynıdır).
Aynı seriden gelen trafolar bağımsız değildir; panelde "Simülasyon" etiketi bu yüzden vardır.
Veri dosyaya yazılmaz: senaryolar bellekteki kopyaya uygulanır.

Konumlar şematiktir (0-100 ızgara), gerçek bir şehir ya da şebeke değildir. Ekipler de temsilîdir.
"""
from dataclasses import dataclass, field

import numpy as np

from .alarm import signals
from .data import HORIZONS, TARGET, TEST_END, TRAIN_END, VAL_END, WINDOW, load_raw
from .explain import explain_at, rules
from .forecaster import Forecaster
from .monitoring import WEEK, flags, load_shift
from .scenarios import Scenario, apply

OFFSETS = [0, 336, 672, 1008]
SIM_HOURS = TEST_END - VAL_END - max(OFFSETS) - max(HORIZONS)
HISTORY_H = 72
ALARM_HOLD_H = 24
KM_PER_UNIT = 0.15

TRANSFORMERS = [
    {"id": "TR-01", "series": "ETTh1", "offset": 0, "x": 18, "y": 22, "zone": "Kuzey"},
    {"id": "TR-02", "series": "ETTh1", "offset": 336, "x": 42, "y": 15, "zone": "Kuzey"},
    {"id": "TR-03", "series": "ETTh1", "offset": 672, "x": 75, "y": 28, "zone": "Doğu"},
    {"id": "TR-04", "series": "ETTh1", "offset": 1008, "x": 85, "y": 60, "zone": "Doğu"},
    {"id": "TR-05", "series": "ETTh2", "offset": 0, "x": 60, "y": 82, "zone": "Güney"},
    {"id": "TR-06", "series": "ETTh2", "offset": 336, "x": 30, "y": 75, "zone": "Güney"},
    {"id": "TR-07", "series": "ETTh2", "offset": 672, "x": 12, "y": 50, "zone": "Batı"},
    {"id": "TR-08", "series": "ETTh2", "offset": 1008, "x": 50, "y": 48, "zone": "Merkez"},
]
TEAMS = [
    {"id": "E1", "name": "Ekip 1", "x": 25, "y": 35},
    {"id": "E2", "name": "Ekip 2", "x": 70, "y": 45},
    {"id": "E3", "name": "Ekip 3", "x": 45, "y": 88},
]


@dataclass
class Fleet:
    backend: str = "onnx"
    scenarios: dict = field(default_factory=dict)          # tid -> Scenario (satır konumu sim saatine göre)
    _fc: dict = field(default_factory=dict)
    _df: dict = field(default_factory=dict)
    _train: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    def __post_init__(self):
        for ds in {t["series"] for t in TRANSFORMERS}:
            self._fc[ds] = Forecaster(ds, self.backend)
            self._df[ds] = load_raw(ds)
            self._train[ds] = self._df[ds].iloc[:TRAIN_END]

    # --- yardımcılar ---
    @staticmethod
    def get(tid: str) -> dict:
        for t in TRANSFORMERS:
            if t["id"] == tid:
                return t
        raise KeyError(tid)

    @staticmethod
    def row(t: dict, hour: int) -> int:
        return VAL_END + t["offset"] + int(np.clip(hour, 0, SIM_HOURS - 1))

    def frame(self, tid: str):
        t = self.get(tid)
        df = self._df[t["series"]]
        sc = self.scenarios.get(tid)
        if sc is None:
            return df
        key = ("df", tid, sc)
        if key not in self._cache:
            real = Scenario(sc.kind, self.row(t, sc.start), sc.magnitude, sc.hours)
            self._cache = {k: v for k, v in self._cache.items() if k[1] != tid}
            self._cache[key] = apply(df, real, self._fc[t["series"]])
        return self._cache[key]

    def set_scenario(self, tid: str, kind: str, magnitude: float, start_hour: int, hours: int = 72):
        self.get(tid)
        self.scenarios[tid] = Scenario(kind, int(start_hour), float(magnitude), int(hours))
        self._cache = {k: v for k, v in self._cache.items() if k[1] != tid}

    def clear_scenario(self, tid: str | None = None):
        tids = [tid] if tid else list(self.scenarios)
        for i in tids:
            self.scenarios.pop(i, None)
        self._cache = {k: v for k, v in self._cache.items() if k[1] not in tids}

    # --- durum ---
    def state(self, tid: str, hour: int, detail: bool = False) -> dict:
        key = ("state", tid, hour, self.scenarios.get(tid), detail)
        if key in self._cache:
            return self._cache[key]
        t = self.get(tid)
        fc, df = self._fc[t["series"]], self.frame(tid)
        levels = rules(fc.meta)
        s = self.row(t, hour)
        s0 = s - WEEK + 1                                    # CUSUM ve izleme için 1 hafta
        sig = signals(fc, df, s0, s + 1)
        ex = explain_at(fc, df, s, levels, sig, s0, hold_h=ALARM_HOLD_H)

        i6 = HORIZONS.index(6)
        ot = df[TARGET].to_numpy(np.float32)[s0:s + 1]
        week = {"coverage": float(((ot >= sig["lower"][:, i6]) & (ot <= sig["upper"][:, i6])).mean()),
                "bias_z": float(sig["z"][:, i6].mean()),
                "load_shift": load_shift(self._train[t["series"]], df.iloc[s0:s + 1])}
        thr = fc.meta["monitoring"]["thresholds"]
        health = {**{k: round(v, 3) for k, v in week.items()}, "flags": flags(week, thr), "thresholds": thr}
        health["healthy"] = not health["flags"]

        status = ex["status"]
        out = {**{k: t[k] for k in ["id", "series", "x", "y", "zone"]}, "hour": int(hour), "status": status,
               "measured_c": ex["measured_c"], "expected_c": ex["expected_c"], "explanation": ex,
               "health": health, "scenario": self._scenario_info(tid), "simulation": True}
        if status == "alarm":
            out["team"] = self.nearest_team(t)
        if detail:
            out["history"] = self.history(fc, df, s, sig, s0, levels["izlemede"])
            out["forecast"] = self.forecast(fc, df, s)
        self._cache[key] = out
        return out

    def _scenario_info(self, tid):
        sc = self.scenarios.get(tid)
        return None if sc is None else {"kind": sc.kind, "magnitude": sc.magnitude, "start_hour": sc.start,
                                        "hours": sc.hours}

    def history(self, fc, df, s, sig, s0, rule) -> list[dict]:
        i = HORIZONS.index(rule["horizon_h"])
        k0 = max(0, (s - s0) - HISTORY_H + 1)
        ot = df[TARGET].to_numpy(np.float32)
        return [{"h": int(k - (s - s0)), "measured": round(float(ot[s0 + k]), 2),
                 "expected": round(float(sig["median"][k, i]), 2), "lower": round(float(sig["lower"][k, i]), 2),
                 "upper": round(float(sig["upper"][k, i]), 2)} for k in range(k0, s - s0 + 1)]

    @staticmethod
    def forecast(fc, df, s) -> list[dict]:
        pred = fc.predict(df, np.array([s]))
        up, lo = fc.upper_bound(pred, "interval")[0], fc.lower_bound(pred)[0]
        return [{"h": h, "expected": round(float(pred[0, i, 1]), 2), "lower": round(float(lo[i]), 2),
                 "upper": round(float(up[i]), 2)} for i, h in enumerate(HORIZONS)]

    @staticmethod
    def nearest_team(t: dict) -> dict:
        d = [(np.hypot(e["x"] - t["x"], e["y"] - t["y"]), e) for e in TEAMS]
        dist, e = min(d, key=lambda p: p[0])
        return {**e, "distance_km": round(float(dist * KM_PER_UNIT), 1), "representative": True}

    def overview(self, hour: int) -> dict:
        items = [self.state(t["id"], hour) for t in TRANSFORMERS]
        return {"hour": int(hour), "sim_hours": SIM_HOURS, "day": int(hour // 24) + 1, "clock": f"{int(hour % 24):02d}:00",
                "transformers": items, "teams": TEAMS, "simulation": True,
                "counts": {s: sum(i["status"] == s for i in items) for s in ["normal", "izlemede", "alarm"]}}


assert WINDOW <= VAL_END - TRAIN_END
