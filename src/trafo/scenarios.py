"""Yapay senaryolar. Hepsi çalışma anında, bellekteki kopyaya uygulanır; değiştirilmiş seri
dosyaya yazılmaz (ETT lisansı CC BY-ND 4.0, bkz. KARARLAR.md, 1).

  sogutma  : yük aynı, yağ sıcaklığı rampayla yükselir (soğutma arızası benzeri). v2'deki yapay arıza.
  ani      : aynı, ama 3 saatte tam büyüklüğe çıkar.
  yuk      : yük artışı ve buna "beklenen" sıcaklık tepkisi. Arıza DEĞİL; negatif kontrol.
             Sıcaklık tepkisi, modelin kendi beklentisinden üretilir (bkz. inject_load).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import LOAD_COLS, TARGET

FAULT_HOURS = 72
RAMPS = {"sogutma": 24, "ani": 3}


@dataclass(frozen=True)
class Scenario:
    kind: str            # "sogutma" | "ani" | "yuk"
    start: int           # satır konumu
    magnitude: float     # °C (arıza) ya da oran (yük, ör. 0,3 = %30 artış)
    hours: int = FAULT_HOURS


def ramp(magnitude: float, ramp_hours: int, hours: int = FAULT_HOURS) -> np.ndarray:
    k = np.arange(hours)
    return magnitude * np.minimum(1.0, (k + 1) / ramp_hours)


def inject_fault(df: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    """Yük sütunlarına dokunmadan OT'ye ısınma ekler."""
    out = df.copy()
    end = min(sc.start + sc.hours, len(out))
    col = out.columns.get_loc(TARGET)
    out.iloc[sc.start:end, col] += ramp(sc.magnitude, RAMPS[sc.kind], sc.hours)[:end - sc.start]
    return out


def inject_load(df: pd.DataFrame, sc: Scenario, fc) -> pd.DataFrame:
    """Yük artışı + beklenen sıcaklık tepkisi (negatif kontrol).

    Yük sütunları 6 saatte (1 + magnitude) katına çıkar ve öyle kalır. Sıcaklığın buna nasıl
    tepki vereceğini fizik modeli olmadan bilemeyiz; bu yüzden tepki modelin kendisinden alınır:
    her saat, artırılmış yükle 1 saat önceki ortanca tahmin uygulanır. Böylece senaryo "model
    için açıklanabilir" bir ısınma üretir. Amaç: alarmın ve açıklamanın, yükle açıklanan ısınmada
    "soğutma arızası" dememesini test etmek. Sınır: gerçek bir trafo farklı tepki verebilir.
    """
    out = df.copy()
    end = min(sc.start + sc.hours, len(out))
    factor = 1.0 + ramp(sc.magnitude, 6, sc.hours)[:end - sc.start]
    for c in LOAD_COLS:
        j = out.columns.get_loc(c)
        out.iloc[sc.start:end, j] = out.iloc[sc.start:end, j].to_numpy() * factor
    col = out.columns.get_loc(TARGET)
    clean_delta = np.diff(df[TARGET].to_numpy(np.float32)[sc.start - 1:end])
    ot_clean = df[TARGET].to_numpy(np.float32)
    for s in range(sc.start, end):
        # 1 saatlik değişim beklentisi: artırılmış yükle ve orijinal yükle
        d_new = fc.predict(out, np.array([s - 1]))[0, 0, 1] - out.iloc[s - 1, col]
        d_old = fc.predict(df, np.array([s - 1]))[0, 0, 1] - ot_clean[s - 1]
        # Orijinal seyir + yük artışının modele göre ek etkisi
        out.iloc[s, col] = out.iloc[s - 1, col] + clean_delta[s - sc.start] + (d_new - d_old)
    return out


def apply(df: pd.DataFrame, sc: Scenario, fc=None) -> pd.DataFrame:
    if sc.kind in RAMPS:
        return inject_fault(df, sc)
    if sc.kind == "yuk":
        if fc is None:
            raise ValueError("yük senaryosu tahminciye ihtiyaç duyar")
        return inject_load(df, sc, fc)
    raise ValueError(sc.kind)
