"""LSTM tabanlı aralık (quantile) tahmin modeli. v2 ile aynı mimari."""
import numpy as np
import torch
from torch import nn

from .data import FEATURE_COLS, HORIZONS, QUANTILES, TARGET

OT_IDX = FEATURE_COLS.index(TARGET)


def make_relative(X):
    """OT kanalını pencerenin son değerine göre farka çevirir (v2 KARARLAR.md, 3).

    Model böylece "şu an kaç derece" yerine "son 7 günde sıcaklık nasıl değişti"yi görür.
    """
    X = X.copy() if isinstance(X, np.ndarray) else X.clone()
    X[:, :, OT_IDX] = X[:, :, OT_IDX] - X[:, -1:, OT_IDX]
    return X


class QuantileLSTM(nn.Module):
    """Girdi (batch, 168, 11) -> çıktı (batch, ufuk sayısı, yüzdelik sayısı)."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, len(HORIZONS) * len(QUANTILES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).view(-1, len(HORIZONS), len(QUANTILES))


def pinball_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Quantile kaybı. pred (B, H, Q), y (B, H)."""
    q = torch.tensor(QUANTILES, device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    err = y.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def sort_quantiles(pred: np.ndarray) -> np.ndarray:
    """Yüzdelik kesişmesini (quantile crossing) sıralayarak düzeltir."""
    return np.sort(pred, axis=-1)
