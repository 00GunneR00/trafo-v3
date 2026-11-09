"""Hızlı testler: bölme, CUSUM, senaryolar, kalibrasyon, edge eşdeğerliği, filo.

  PYTHONPATH=src python -m pytest -q
"""
import json

import numpy as np
import pytest

from trafo.alarm import alarms_threshold, cusum_path
from trafo.calibrate import conformal_quantile
from trafo.data import DATASETS, HORIZONS, LOAD_COLS, REPORTS, TARGET, TEST_END, TRAIN_END, VAL_END, WINDOW, load_raw, split
from trafo.forecaster import Forecaster, load_meta
from trafo.scenarios import Scenario, inject_fault, ramp


@pytest.fixture(scope="module")
def df():
    return load_raw("ETTh1")


def test_split_is_chronological_and_disjoint(df):
    parts = split(df)
    assert parts["train"].index.max() < parts["val"].index[WINDOW]
    assert parts["val"].index.max() < parts["test"].index[WINDOW]
    assert len(parts["test"]) == TEST_END - VAL_END + WINDOW


def test_cusum_resets_and_fires():
    z = np.array([0, 0, 3, 3, 3, 3, 3, 0], dtype=np.float32)
    path, fired = cusum_path(z, k=0.5, h_thr=8)  # birikim: 2,5 5 7,5 10 -> alarm, sıfırla
    assert fired.tolist() == [False, False, False, False, False, True, False, False]
    assert path[6] == pytest.approx(2.5)          # alarmdan sonra sıfırdan başladı


def test_threshold_rule_needs_consecutive_hours():
    e = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    assert alarms_threshold(e, 3).tolist() == [False] * 5 + [True]


def test_conformal_quantile_is_finite_sample_corrected():
    s = np.arange(1, 101, dtype=float)
    # düzeltilmiş seviye ceil(101*0,9)/100 = 0,91 (düz 0,9 değil); "higher" ile 92
    assert conformal_quantile(s, 0.9) == 92.0
    assert conformal_quantile(s, 0.9) > np.quantile(s, 0.9, method="higher")


def test_fault_injection_only_touches_ot(df):
    sc = Scenario("sogutma", 10_000, 3.0)
    out = inject_fault(df, sc)
    assert (out[LOAD_COLS].values == df[LOAD_COLS].values).all()
    diff = (out[TARGET] - df[TARGET]).to_numpy()
    assert diff[:10_000].max() == 0
    assert diff[10_000 + 23] == pytest.approx(3.0, abs=1e-4)   # 24 saatlik rampanın sonu
    assert ramp(5.0, 3)[2] == pytest.approx(5.0)
    assert (df[TARGET] == load_raw("ETTh1")[TARGET]).all()      # orijinal kopya değişmedi


@pytest.mark.parametrize("ds", DATASETS)
def test_meta_has_all_parts(ds):
    m = load_meta(ds)
    for key in ["calibration", "alarm_rule", "alarm_rule_fleet", "monitoring", "edge"]:
        assert key in m, key
    assert m["window"] == WINDOW and m["horizons"] == HORIZONS


@pytest.mark.parametrize("ds", DATASETS)
def test_onnx_matches_torch(ds):
    torch_fc = Forecaster(ds, "torch", device="cpu")
    onnx_fc = Forecaster(ds, "onnx")
    d = load_raw(ds)
    t = np.arange(VAL_END, VAL_END + 64)
    np.testing.assert_allclose(torch_fc.predict(d, t), onnx_fc.predict(d, t), atol=1e-3)


def test_reproduces_v2_on_etth1():
    """v2'nin ETTh1 modeli v3 koduyla aynı test sonuçlarını vermeli (bkz. README)."""
    m = load_meta("ETTh1")
    cov = m["calibration_report"]["after"]["test"]
    assert cov["h1"]["coverage_90"] == pytest.approx(0.9622, abs=1e-3)
    rule = m["alarm_rule"]
    assert (rule["kind"], rule["k"], rule["H"], rule["horizon_h"]) == ("cusum", 0.5, 8, 6)
    assert rule["test"]["kademeli_3"]["detection_rate"] == pytest.approx(0.75)


def test_reports_exist():
    for name in ["baselines.json", "seeds.json", "monitoring.json", "edge_ETTh1.json", "explain_ETTh1.json"]:
        assert (REPORTS / name).exists(), name
    mon = json.loads((REPORTS / "monitoring.json").read_text(encoding="utf-8"))
    assert mon["ETTh1 modeli / ETTh2 verisi (bilinen kayma)"]["flag_rate_test"] > 0.9


def test_fleet_scenario_raises_warning():
    from trafo.fleet import Fleet
    f = Fleet(backend="onnx")
    assert f.state("TR-03", 300)["status"] == "normal"
    f.set_scenario("TR-03", "ani", 6.0, 285)
    s = f.state("TR-03", 300)
    assert s["status"] in ("izlemede", "alarm") and s["explanation"]["label"] == "sogutma_suphesi"
    f.clear_scenario()
    assert f.state("TR-03", 300)["status"] == "normal"
