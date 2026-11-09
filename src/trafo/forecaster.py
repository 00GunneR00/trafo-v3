"""Kayıtlı modeli yükleyip ham veriden °C tahmin üreten ortak katman.

Değerlendirme, alarm, izleme ve API aynı kodu kullanır. Üç arka uç var:
  torch      : eğitimdeki PyTorch modeli (GPU varsa GPU)
  onnx       : aynı modelin ONNX fp32 kopyası (servis bunu kullanır; konteynerde PyTorch gerekmez)
  onnx-int8  : dinamik int8 nicemlenmiş ONNX kopyası (bkz. edge.py)
Ön işleme (ölçekleme, göreli girdi) ve son işleme (°C'ye çevirme, kalibrasyon) arka uçtan bağımsızdır.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURE_COLS, HORIZONS, MODELS, TARGET, TIME_COLS, WINDOW, Scaler, add_time_features, to_celsius

OT_IDX = FEATURE_COLS.index(TARGET)


def model_dir(dataset: str) -> Path:
    return MODELS / dataset


def load_meta(dataset: str) -> dict:
    return json.loads((model_dir(dataset) / "latest.json").read_text(encoding="utf-8"))


def save_meta(dataset: str, meta: dict) -> None:
    text = json.dumps(meta, indent=2, ensure_ascii=False)
    (model_dir(dataset) / "latest.json").write_text(text, encoding="utf-8")
    (model_dir(dataset) / f"lstm_{meta['version']}.json").write_text(text, encoding="utf-8")


class Forecaster:
    def __init__(self, dataset: str = "ETTh1", backend: str = "torch", device: str | None = None,
                 meta: dict | None = None):
        self.dataset = dataset
        self.backend = backend
        self.meta = meta or load_meta(dataset)
        self.scaler = Scaler.from_dict(self.meta["scaler"])
        self.calibration = self.meta.get("calibration", {})
        d = model_dir(dataset)
        if backend == "torch":
            import torch
            from .model import QuantileLSTM
            self._torch = torch
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = QuantileLSTM(self.meta["n_features"], self.meta["hidden"],
                                      self.meta["layers"], self.meta["dropout"]).to(self.device)
            state = torch.load(d / self.meta["file"], map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
        elif backend in ("onnx", "onnx-int8"):
            import onnxruntime as ort
            key = "onnx_fp32" if backend == "onnx" else "onnx_int8"
            if key not in self.meta.get("edge", {}):
                raise FileNotFoundError(f"{dataset}: {backend} modeli yok, önce edge.py çalıştırılmalı")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self.session = ort.InferenceSession(str(d / self.meta["edge"][key]["file"]), opts,
                                                providers=["CPUExecutionProvider"])
            self.device = "cpu"
            # Nicemlenmiş model kendi val kalibrasyonunu taşıyabilir (bkz. edge.py)
            if "calibration" in self.meta["edge"][key]:
                self.calibration = self.meta["edge"][key]["calibration"]
        else:
            raise ValueError(backend)

    @property
    def version(self) -> str:
        return self.meta["version"]

    def windows(self, df: pd.DataFrame, t_positions: np.ndarray) -> np.ndarray:
        """t_positions: tahminin yapıldığı satırların konumu (her biri >= WINDOW-1)."""
        if "hour_sin" not in df.columns:
            df = add_time_features(df)
        values = self.scaler.transform(df)[FEATURE_COLS + TIME_COLS].to_numpy(np.float32)
        idx = t_positions[:, None] - WINDOW + 1 + np.arange(WINDOW)[None, :]
        X = values[idx]
        if self.meta["relative_input"]:
            X[:, :, OT_IDX] -= X[:, -1:, OT_IDX]
        return X

    def raw(self, X: np.ndarray, batch: int = 2048) -> np.ndarray:
        """Ölçekli model çıktısı (n, H, Q), sıralanmamış."""
        outs = []
        if self.backend == "torch":
            with self._torch.no_grad():
                for i in range(0, len(X), batch):
                    xb = self._torch.from_numpy(X[i:i + batch]).to(self.device)
                    outs.append(self.model(xb).float().cpu().numpy())
        else:
            name = self.session.get_inputs()[0].name
            for i in range(0, len(X), batch):
                outs.append(self.session.run(None, {name: X[i:i + batch]})[0])
        return np.concatenate(outs)

    def predict(self, df: pd.DataFrame, t_positions: np.ndarray | None = None) -> np.ndarray:
        """°C cinsinden (n, ufuk, yüzdelik) tahmin. Varsayılan: verinin son saati için tek tahmin."""
        if t_positions is None:
            t_positions = np.array([len(df) - 1])
        pred = np.sort(self.raw(self.windows(df, t_positions)), axis=-1)
        base = df[TARGET].to_numpy(np.float32)[t_positions]
        return to_celsius(pred, base, self.scaler, self.meta["target"])

    def upper_bound(self, pred_c: np.ndarray, level: str = "alarm") -> np.ndarray:
        """Kalibre edilmiş üst sınır (n, ufuk). level: "interval" (%90 aralığın üstü) ya da "alarm" (%99)."""
        shift = np.array([self.calibration.get(level, {}).get(f"h{h}", 0.0) for h in HORIZONS], np.float32)
        return pred_c[:, :, 2] + shift

    def lower_bound(self, pred_c: np.ndarray) -> np.ndarray:
        shift = np.array([self.calibration.get("interval", {}).get(f"h{h}", 0.0) for h in HORIZONS], np.float32)
        return pred_c[:, :, 0] - shift
