"""Trafo v3 servisi: filo simülasyonu, açıklamalı alarm, model sağlığı.

Çıkarım ONNX fp32 ile yapılır (PyTorch'tan farkı 1e-6 mertebesinde, bkz. reports/edge_*.json);
konteynerde PyTorch yoktur.

Uç noktalar:
  GET    /health                     servis durumu
  GET    /model                      iki modelin kartı: ölçümler, çıtalar, kalibrasyon, alarm, edge, izleme
  GET    /fleet?hour=H               8 sanal trafonun H saatindeki durumu (simülasyon)
  GET    /transformers/{id}?hour=H   tek trafo: son 72 saat, 1/6/24 saat tahmin, açıklama, sağlık
  POST   /scenario                   bir trafoya senaryo uygula (sogutma | ani | yuk)
  DELETE /scenario[/{id}]            senaryoyu kaldır

Çalıştırma: PYTHONPATH=src uvicorn api.main:app --port 8000
"""
import json
import sys
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafo.data import DATASETS, REPORTS  # noqa: E402
from trafo.fleet import SIM_HOURS, TEAMS, TRANSFORMERS, Fleet  # noqa: E402
from trafo.forecaster import load_meta  # noqa: E402

app = FastAPI(title="Trafo v3: filo izleme servisi", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

fleet = Fleet(backend="onnx")
STARTED = time.time()


def _report(name: str):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _model_card(ds: str) -> dict:
    m = load_meta(ds)
    rep = m["calibration_report"]["after"]["test"]
    edge = m.get("edge", {})
    return {
        "series": ds,
        "version": m["version"],
        "architecture": f"LSTM {m['layers']}x{m['hidden']}, girdi {m['window']} saat, hedef: değişim (delta)",
        # Tüm raporlarla aynı ölçüm: PyTorch CPU (eğitimdeki GPU/AMP ölçümü 3. basamakta farklı olabilir)
        "test_mae_c": {k: round(v, 3) for k, v in edge["torch_cpu"]["test_mae"].items()} if "torch_cpu" in edge
        else {f"h{h}": round(m["test_metrics"][f"test_mae_h{h}"], 3) for h in m["horizons"]},
        "test_coverage_90": {k: round(v["coverage_90"], 3) for k, v in rep.items()},
        "test_width_90_c": {k: round(v["width_90"], 2) for k, v in rep.items()},
        "alarm_rule": m["alarm_rule"],
        "alarm_rule_fleet": m.get("alarm_rule_fleet"),
        "monitoring": m.get("monitoring"),
        "edge": {k: {kk: v.get(kk) for kk in ["file", "size_kb", "test_mae", "latency_cpu_1thread"]}
                 for k, v in edge.items() if k.startswith("onnx")},
        "edge_machine": edge.get("machine"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "backend": fleet.backend, "uptime_s": round(time.time() - STARTED),
            "models": {ds: load_meta(ds)["version"] for ds in DATASETS}, "simulation": True}


@app.get("/model")
def model():
    return {"models": [_model_card(ds) for ds in DATASETS], "baselines": _report("baselines.json"),
            "seeds": _report("seeds.json"), "explain": {ds: _report(f"explain_{ds}.json") for ds in DATASETS}}


def _hour(hour: int) -> int:
    if not 0 <= hour < SIM_HOURS:
        raise HTTPException(422, f"hour 0 ile {SIM_HOURS - 1} arasında olmalı")
    return hour


@app.get("/fleet")
def get_fleet(hour: int = Query(0)):
    return fleet.overview(_hour(hour))


@app.get("/transformers/{tid}")
def transformer(tid: str, hour: int = Query(0)):
    if tid not in {t["id"] for t in TRANSFORMERS}:
        raise HTTPException(404, "böyle bir trafo yok")
    return fleet.state(tid, _hour(hour), detail=True)


class ScenarioIn(BaseModel):
    id: str
    kind: Literal["sogutma", "ani", "yuk"]
    magnitude: float = Field(..., gt=0, le=10, description="°C (arıza) ya da oran (yük, 0,3 = %30)")
    start_hour: int = Field(..., ge=0)
    hours: int = Field(72, ge=6, le=168)


@app.post("/scenario")
def scenario(sc: ScenarioIn):
    if sc.id not in {t["id"] for t in TRANSFORMERS}:
        raise HTTPException(404, "böyle bir trafo yok")
    if sc.kind == "yuk" and sc.magnitude > 1:
        raise HTTPException(422, "yük oranı en fazla 1 (%100 artış)")
    fleet.set_scenario(sc.id, sc.kind, sc.magnitude, _hour(sc.start_hour), sc.hours)
    return {"ok": True, "scenarios": {k: v.__dict__ for k, v in fleet.scenarios.items()}}


@app.delete("/scenario")
def clear_all():
    fleet.clear_scenario()
    return {"ok": True}


@app.delete("/scenario/{tid}")
def clear_one(tid: str):
    fleet.clear_scenario(tid)
    return {"ok": True}


@app.get("/teams")
def teams():
    return {"teams": TEAMS, "representative": True}
