"""LSTM eğitimi (v2 tarifinin aynısı), MLflow kaydı ve model dosyası.

  python -m trafo.train --dataset ETTh2 --seed 0 --save      # ikinci gerçek trafo için model
  python -m trafo.train --dataset ETTh1 --seed 1             # seed oynaklığı (kaydetmeden)

Varsayılanlar v2'de seçilen yapılandırma: hedef değişim, göreli girdi, LSTM 2x32, dropout 0,3, lr 5e-4.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import DATASETS, HORIZONS, QUANTILES, WINDOW, prepare, to_celsius
from .forecaster import model_dir, save_meta
from .metrics import interval_metrics, mlflow_setup, point_metrics
from .model import QuantileLSTM, make_relative, pinball_loss, sort_quantiles


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default="ETTh1")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    model.eval()
    outs = [model(torch.from_numpy(X[i:i + batch]).to(device)).float().cpu().numpy()
            for i in range(0, len(X), batch)]
    return sort_quantiles(np.concatenate(outs))


def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    target = "delta"

    data, scaler = prepare(args.dataset, target)
    data = {k: (make_relative(X), y, b, t) for k, (X, y, b, t) in data.items()}
    Xtr, ytr, _, _ = data["train"]
    Xva, yva, _, _ = data["val"]

    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=args.batch, shuffle=True, drop_last=True, generator=gen)
    model = QuantileLSTM(Xtr.shape[2], args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)

    mlflow = mlflow_setup()
    run_name = f"{args.dataset}_lstm_s{args.seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"dataset": args.dataset, "model_type": "lstm"})
        mlflow.log_params({**vars(args), "target": target, "relative_input": True, "amp": use_amp, "window": WINDOW})
        best_val, best_state, bad, times = float("inf"), None, 0, []
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred = model(xb)
                loss = pinball_loss(pred.float(), yb)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(opt)
                grad_scaler.update()
            times.append(time.time() - t0)
            val_loss = pinball_loss(torch.from_numpy(predict(model, Xva, device)), torch.from_numpy(yva)).item()
            mlflow.log_metric("val_pinball", val_loss, step=epoch)
            if val_loss < best_val - 1e-4:
                best_val, bad = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break

        model.load_state_dict(best_state)
        metrics = {"best_val_pinball": best_val, "epochs_run": epoch + 1,
                   "mean_epoch_seconds": float(np.mean(times[1:] or times))}
        for split_name in ["val", "test"]:
            X, y, base, _ = data[split_name]
            pq = to_celsius(predict(model, X, device), base, scaler, target)
            yc = to_celsius(y, base, scaler, target)
            metrics.update(point_metrics(yc, pq[:, :, 1], split_name))
            metrics.update(interval_metrics(yc, pq[:, :, 0], pq[:, :, 2], split_name))
        mlflow.log_metrics(metrics)
        print(f"{run_name}: " + "  ".join(f"h{h} test MAE {metrics[f'test_mae_h{h}']:.3f}" for h in HORIZONS))

        if args.save:
            d = model_dir(args.dataset)
            d.mkdir(parents=True, exist_ok=True)
            version = time.strftime("%Y%m%d-%H%M%S")
            path = d / f"lstm_{version}.pt"
            torch.save(model.state_dict(), path)
            meta = {
                "dataset": args.dataset, "version": version, "file": path.name, "run_name": run_name,
                "mlflow_run_id": mlflow.active_run().info.run_id, "seed": args.seed,
                "target": target, "relative_input": True,
                "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                "n_features": int(Xtr.shape[2]), "window": WINDOW,
                "horizons": HORIZONS, "quantiles": QUANTILES, "scaler": scaler.to_dict(),
                "test_metrics": {k: v for k, v in metrics.items() if k.startswith("test_")},
            }
            save_meta(args.dataset, meta)
            print(f"kaydedildi: {path}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
