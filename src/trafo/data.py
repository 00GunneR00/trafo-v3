"""Veri yükleme, zamana göre bölme, ölçekleme ve pencereleme.

v2'deki data.py'nin devamı. Fark: veri seti adı parametre (ETTh1, ETTh2). Bölme, ölçekleme
ve pencereleme kuralları aynen korunur (bkz. trafo-v2/KARARLAR.md, 2-3).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

DATASETS = ["ETTh1", "ETTh2"]
SOURCE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET = "OT"
FEATURE_COLS = LOAD_COLS + [TARGET]
TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

WINDOW = 168            # girdi: son 7 gün
HORIZONS = [1, 6, 24]   # kaç saat ileri
QUANTILES = [0.05, 0.5, 0.95]

# ETT standardı: 12 / 4 / 4 ay (ay = 30 gün)
TRAIN_END = 12 * 30 * 24
VAL_END = TRAIN_END + 4 * 30 * 24
TEST_END = VAL_END + 4 * 30 * 24


def raw_path(name: str) -> Path:
    if name not in DATASETS:
        raise ValueError(f"Bilinmeyen veri seti: {name}")
    return DATA / f"{name}.csv"


def download(name: str) -> Path:
    """Veriyi kaynağından indirir. Veri depoya konmaz (lisans: CC BY-ND 4.0, kaynakta kalır)."""
    import urllib.request
    path = raw_path(name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL.format(name=name), path)
    return path


def load_raw(name: str = "ETTh1") -> pd.DataFrame:
    df = pd.read_csv(download(name), parse_dates=["date"]).set_index("date")
    return add_time_features(df)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Saat ve gün döngüseldir (23 ile 0 komşudur), bu yüzden sin/cos ile kodlanır.
    df = df.copy()
    hour = df.index.hour
    dow = df.index.day_of_week
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


@dataclass
class Scaler:
    """Train'in ortalama ve standart sapmasıyla ölçekler. Asla val/test'e fit edilmez."""
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "Scaler":
        return cls(train[FEATURE_COLS].mean(), train[FEATURE_COLS].std())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[FEATURE_COLS] = (df[FEATURE_COLS] - self.mean) / self.std
        return out

    @property
    def ot_std(self) -> float:
        return float(self.std[TARGET])

    def to_dict(self) -> dict:
        return {"mean": self.mean.to_dict(), "std": self.std.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(pd.Series(d["mean"]), pd.Series(d["std"]))


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Zamana göre bölme. val ve test, ilk tahmin için kendinden önceki WINDOW saati de içerir."""
    return {
        "train": df.iloc[:TRAIN_END],
        "val": df.iloc[TRAIN_END - WINDOW:VAL_END],
        "test": df.iloc[VAL_END - WINDOW:TEST_END],
    }


def make_windows(scaled: pd.DataFrame, raw_ot: pd.Series, target: str = "delta"):
    """Kayan pencereler.

    X: (n, WINDOW, 11), y: (n, len(HORIZONS)) ölçekli hedef, base: (n,) t anındaki gerçek OT,
    t_index: tahminin yapıldığı an. target="delta": y = (OT(t+h) - OT(t)) / ot_std.
    """
    values = scaled[FEATURE_COLS + TIME_COLS].to_numpy(np.float32)
    ot_scaled = scaled[TARGET].to_numpy(np.float32)
    ot = raw_ot.to_numpy(np.float32)
    max_h = max(HORIZONS)
    n = len(scaled) - WINDOW - max_h + 1
    if n <= 0:
        raise ValueError("Pencere için veri yetersiz")

    idx = np.arange(n)[:, None] + np.arange(WINDOW)[None, :]
    X = values[idx]
    t = np.arange(n) + WINDOW - 1
    if target == "delta":
        y = np.stack([ot_scaled[t + h] - ot_scaled[t] for h in HORIZONS], axis=1)
    elif target == "level":
        y = np.stack([ot_scaled[t + h] for h in HORIZONS], axis=1)
    else:
        raise ValueError(target)
    return X, y.astype(np.float32), ot[t], scaled.index[t]


def to_celsius(pred_scaled: np.ndarray, base: np.ndarray, scaler: Scaler, target: str) -> np.ndarray:
    """Ölçekli tahmini °C'ye çevirir. pred_scaled: (n, H) ya da (n, H, Q)."""
    std, mean = scaler.ot_std, float(scaler.mean[TARGET])
    if target == "delta":
        b = base.reshape(-1, *([1] * (pred_scaled.ndim - 1)))
        return pred_scaled * std + b
    return pred_scaled * std + mean


def prepare(name: str = "ETTh1", target: str = "delta"):
    """Eğitimin tek çağrısı: ölçekli pencereler + scaler."""
    df = load_raw(name)
    parts = split(df)
    scaler = Scaler.fit(parts["train"])
    out = {}
    for part_name, part in parts.items():
        out[part_name] = make_windows(scaler.transform(part), part[TARGET], target)
    return out, scaler
