"""Seed oynaklığı: aynı tarifle 3 seed eğitilir (kaydedilmez), test ölçümlerinin ortalaması ve aralığı raporlanır.

Servisteki modeller tek seed'dir (ETTh1: v2'nin modeli, ETTh2: seed 0). Bu rapor, manşet sayıların
seed'e ne kadar bağlı olduğunu gösterir.

  python -m trafo.seeds
"""
import json

import numpy as np

from .data import DATASETS, HORIZONS, REPORTS
from .train import parse_args, train

SEEDS = [0, 1, 2]


def main():
    report = {}
    for d in DATASETS:
        runs = [train(parse_args(["--dataset", d, "--seed", str(s)])) for s in SEEDS]
        keys = [f"test_mae_h{h}" for h in HORIZONS] + [f"test_coverage_h{h}" for h in HORIZONS]
        report[d] = {k: {"mean": float(np.mean([r[k] for r in runs])), "min": float(np.min([r[k] for r in runs])),
                         "max": float(np.max([r[k] for r in runs]))} for k in keys}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "seeds.json").write_text(json.dumps({"seeds": SEEDS, "report": report}, indent=2), encoding="utf-8")
    for d, r in report.items():
        print(d, {k: f"{v['mean']:.3f} [{v['min']:.3f}-{v['max']:.3f}]" for k, v in r.items() if "mae" in k})


if __name__ == "__main__":
    main()
