"""Basit çıtalar: "son değer" ve "dünün aynı saati". Her veri setinde LSTM bunlarla kıyaslanır.

  python -m trafo.baselines
"""
import json

import numpy as np

from .data import DATASETS, HORIZONS, REPORTS, TARGET, TEST_END, TRAIN_END, VAL_END, load_raw


def baseline_mae(dataset: str) -> dict:
    ot = load_raw(dataset)[TARGET].to_numpy(np.float32)
    out = {}
    for split_name, (start, end) in {"val": (TRAIN_END, VAL_END), "test": (VAL_END, TEST_END)}.items():
        for i, h in enumerate(HORIZONS):
            s = np.arange(start, end)            # hedef saatler
            out[f"{split_name}_son_deger_h{h}"] = float(np.abs(ot[s] - ot[s - h]).mean())
            # Dünün aynı saati: s için bilinen en yakın "24 saat önceki" değer, s - h anında bilinir
            lag = 24 * int(np.ceil(h / 24))
            out[f"{split_name}_dun_h{h}"] = float(np.abs(ot[s] - ot[s - lag]).mean())
    return out


if __name__ == "__main__":
    res = {d: baseline_mae(d) for d in DATASETS}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "baselines.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for d, r in res.items():
        print(d, {k: round(v, 3) for k, v in r.items() if k.startswith("test")})
