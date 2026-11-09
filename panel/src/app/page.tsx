"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import DetailChart from "@/components/DetailChart";
import FleetMap, { STATUS_COLOR, STATUS_LABEL } from "@/components/FleetMap";
import { ExplanationCard, HealthCard, ModelSummary } from "@/components/Cards";
import {
  clearScenario,
  fetchFleet,
  fetchModel,
  fetchTransformer,
  postScenario,
  type FleetOverview,
  type ModelCard,
  type TransformerState,
} from "@/lib/api";

const START_HOUR = 300;
const SCENARIOS = [
  { kind: "sogutma", label: "Soğutma arızası (24 sa rampa)", unit: "°C", def: 3, min: 1, max: 8, step: 0.5 },
  { kind: "ani", label: "Ani ısınma (3 sa)", unit: "°C", def: 5, min: 1, max: 8, step: 0.5 },
  { kind: "yuk", label: "Yük artışı (arıza değil)", unit: "%", def: 40, min: 10, max: 100, step: 10 },
] as const;

export default function Page() {
  const [hour, setHour] = useState(START_HOUR);
  const [playing, setPlaying] = useState(false);
  const [fleet, setFleet] = useState<FleetOverview | null>(null);
  const [selected, setSelected] = useState<string | null>("TR-03");
  const [detail, setDetail] = useState<TransformerState | null>(null);
  const [models, setModels] = useState<ModelCard[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [scKind, setScKind] = useState<(typeof SCENARIOS)[number]["kind"]>("sogutma");
  const [scMag, setScMag] = useState<number>(3);
  const [version, setVersion] = useState(0); // senaryo değişince yeniden çek
  const inflight = useRef<AbortController | null>(null);

  const load = useCallback(async (h: number, id: string | null) => {
    inflight.current?.abort();
    const ac = new AbortController();
    inflight.current = ac;
    try {
      const [f, d] = await Promise.all([fetchFleet(h, ac.signal), id ? fetchTransformer(id, h, ac.signal) : Promise.resolve(null)]);
      setFleet(f);
      setDetail(d);
      setError(null);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError("Servise bağlanılamadı. API'nin çalıştığından emin olun (varsayılan http://localhost:8000).");
    }
  }, []);

  useEffect(() => {
    const id = setTimeout(() => load(hour, selected), 0);
    return () => clearTimeout(id);
  }, [hour, selected, version, load]);

  useEffect(() => {
    fetchModel().then((m) => setModels(m.models)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setHour((h) => Math.min(h + 1, (fleet?.sim_hours ?? 1848) - 1)), 700);
    return () => clearInterval(id);
  }, [playing, fleet?.sim_hours]);

  const sc = SCENARIOS.find((s) => s.kind === scKind)!;
  const applyScenario = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      await postScenario({ id: selected, kind: scKind, magnitude: scKind === "yuk" ? scMag / 100 : scMag, start_hour: hour });
      setVersion((v) => v + 1);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const removeScenario = async (all = false) => {
    await clearScenario(all ? undefined : (selected ?? undefined));
    setVersion((v) => v + 1);
  };

  const sorted = [...(fleet?.transformers ?? [])].sort((a, b) => {
    const rank = { alarm: 0, izlemede: 1, normal: 2 };
    return rank[a.status] - rank[b.status] || a.id.localeCompare(b.id);
  });

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-5 p-4 sm:p-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Trafo Filo İzleme</h1>
          <p className="text-sm text-neutral-400">
            Yağ sıcaklığı tahmini · açıklamalı alarm · model sağlığı.{" "}
            <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-xs text-amber-300">SİMÜLASYON</span>
          </p>
          <p className="mt-1 text-xs text-neutral-500">
            8 sanal trafo, 2 gerçek açık veri serisinden (ETTh1, ETTh2; CC BY-ND 4.0) zaman kaydırılarak üretildi. Yerleşim ve ekipler temsilîdir.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="font-mono text-neutral-300">
            Gün {fleet?.day ?? "–"} · {fleet?.clock ?? "--:--"}
          </span>
          <button onClick={() => setHour((h) => Math.max(0, h - 24))} className="rounded-md bg-neutral-800 px-2 py-1.5 hover:bg-neutral-700">
            −24 sa
          </button>
          <button onClick={() => setHour((h) => Math.max(0, h - 1))} className="rounded-md bg-neutral-800 px-2 py-1.5 hover:bg-neutral-700">
            −1
          </button>
          <button onClick={() => setPlaying((p) => !p)} className="w-20 rounded-md bg-blue-600 px-3 py-1.5 font-medium hover:bg-blue-500">
            {playing ? "Durdur" : "Oynat"}
          </button>
          <button onClick={() => setHour((h) => h + 1)} className="rounded-md bg-neutral-800 px-2 py-1.5 hover:bg-neutral-700">
            +1
          </button>
          <button onClick={() => setHour((h) => h + 24)} className="rounded-md bg-neutral-800 px-2 py-1.5 hover:bg-neutral-700">
            +24 sa
          </button>
        </div>
      </header>

      <input
        type="range"
        min={0}
        max={(fleet?.sim_hours ?? 1848) - 1}
        value={hour}
        onChange={(e) => setHour(Number(e.target.value))}
        className="w-full accent-blue-500"
        aria-label="Simülasyon saati"
      />

      {error && <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">{error}</div>}

      <div className="grid grid-cols-3 gap-3">
        {(["alarm", "izlemede", "normal"] as const).map((s) => (
          <div key={s} className="rounded-lg border p-3" style={{ borderColor: `${STATUS_COLOR[s]}55`, background: `${STATUS_COLOR[s]}12` }}>
            <div className="text-sm" style={{ color: STATUS_COLOR[s] }}>
              {STATUS_LABEL[s]}
            </div>
            <div className="text-3xl font-semibold">{fleet?.counts[s] ?? "–"}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <div className="aspect-square max-h-[460px] w-full overflow-hidden rounded-lg border border-neutral-800 lg:aspect-auto lg:h-[460px]">
          {fleet && <FleetMap transformers={fleet.transformers} teams={fleet.teams} selected={selected} onSelect={setSelected} />}
        </div>
        <div className="max-h-[460px] overflow-auto rounded-lg border border-neutral-800">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-neutral-900 text-left text-neutral-400">
              <tr>
                <th className="p-2.5 font-normal">Trafo</th>
                <th className="p-2.5 font-normal">Bölge</th>
                <th className="p-2.5 font-normal">Ölçülen / beklenen</th>
                <th className="p-2.5 font-normal">Durum</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((t) => (
                <tr
                  key={t.id}
                  onClick={() => setSelected(t.id)}
                  className={`cursor-pointer border-t border-neutral-800 hover:bg-neutral-900 ${t.id === selected ? "bg-neutral-900" : ""}`}
                >
                  <td className="p-2.5 font-medium">
                    {t.id}
                    {t.scenario && <span className="ml-1.5 rounded bg-violet-500/20 px-1 text-[10px] text-violet-300">senaryo</span>}
                    {!t.health.healthy && <span className="ml-1.5 rounded bg-amber-500/20 px-1 text-[10px] text-amber-300">model?</span>}
                  </td>
                  <td className="p-2.5 text-neutral-400">{t.zone}</td>
                  <td className="p-2.5 font-mono">
                    {t.measured_c.toFixed(1)} / {t.expected_c.toFixed(1)} °C
                  </td>
                  <td className="p-2.5">
                    <span className="rounded-full px-2 py-0.5 text-xs" style={{ color: STATUS_COLOR[t.status], background: `${STATUS_COLOR[t.status]}1f` }}>
                      {STATUS_LABEL[t.status]}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {detail && (
        <section className="flex flex-col gap-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-xl font-semibold">
              {detail.id} <span className="text-sm font-normal text-neutral-400">· {detail.zone} · kaynak seri {detail.series}</span>
            </h2>
          </div>
          <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
            <DetailChart t={detail} />
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <ExplanationCard t={detail} />
            <HealthCard t={detail} />
            <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
              <h3 className="mb-2 font-semibold">Senaryo dene</h3>
              <div className="flex flex-col gap-2 text-sm">
                {SCENARIOS.map((s) => (
                  <label key={s.kind} className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="sc"
                      checked={scKind === s.kind}
                      onChange={() => {
                        setScKind(s.kind);
                        setScMag(s.def);
                      }}
                    />
                    {s.label}
                  </label>
                ))}
                <label className="mt-1 flex items-center gap-2">
                  <span className="w-20 text-neutral-400">Büyüklük</span>
                  <input type="range" min={sc.min} max={sc.max} step={sc.step} value={scMag} onChange={(e) => setScMag(Number(e.target.value))} className="flex-1 accent-violet-500" />
                  <span className="w-14 text-right font-mono">
                    {scKind === "yuk" ? `%${scMag}` : `${scMag} °C`}
                  </span>
                </label>
                <button disabled={busy} onClick={applyScenario} className="mt-1 rounded-md bg-violet-600 px-3 py-1.5 font-medium hover:bg-violet-500 disabled:opacity-50">
                  {busy ? "Uygulanıyor…" : `${detail.id} için şimdi başlat`}
                </button>
                {detail.scenario && (
                  <button onClick={() => removeScenario()} className="rounded-md bg-neutral-800 px-3 py-1.5 hover:bg-neutral-700">
                    Senaryoyu kaldır
                  </button>
                )}
                {fleet?.transformers.some((t) => t.scenario) && (
                  <button onClick={() => removeScenario(true)} className="text-xs text-neutral-400 underline">
                    Tüm senaryoları kaldır
                  </button>
                )}
                <p className="text-xs text-neutral-500">
                  Senaryo 72 saat sürer, sonra “Oynat” ile izleyin. Yük senaryosunda sıcaklık tepkisi modelin kendi beklentisinden üretilir; alarm vermemesi beklenir.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {models.length > 0 && <ModelSummary models={models} />}

      <footer className="pb-4 text-xs text-neutral-500">
        Trafo v3 · veri: ETDataset (Zhou vd., 2021), CC BY-ND 4.0 · Bu panel bir portföy projesidir; gerçek bir şebekeye bağlı değildir.
      </footer>
    </div>
  );
}
