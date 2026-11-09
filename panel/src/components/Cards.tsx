"use client";

import type { ModelCard, TransformerState } from "@/lib/api";
import { STATUS_COLOR, STATUS_LABEL } from "./FleetMap";

const box = "rounded-lg border border-neutral-800 bg-neutral-950 p-4";

function Row({ k, v, warn }: { k: string; v: string; warn?: boolean }) {
  return (
    <div className="flex justify-between gap-3 py-0.5 text-sm">
      <span className="text-neutral-400">{k}</span>
      <span className={`font-mono ${warn ? "text-amber-300" : "text-neutral-100"}`}>{v}</span>
    </div>
  );
}

export function ExplanationCard({ t }: { t: TransformerState }) {
  const e = t.explanation;
  const c = STATUS_COLOR[t.status];
  return (
    <div className={box}>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-semibold">Neden?</h3>
        <span className="rounded-full border px-2 py-0.5 text-xs" style={{ color: c, borderColor: `${c}66`, background: `${c}1a` }}>
          {STATUS_LABEL[t.status]}
          {t.status !== "normal" && e.warning_age_h[t.status] !== undefined && ` · ${e.warning_age_h[t.status]} sa önce`}
        </span>
      </div>
      <p className="mb-3 text-sm" style={{ color: t.status === "normal" ? "#d4d4d4" : c }}>
        {e.text}
      </p>
      <Row k="Ölçülen / beklenen" v={`${e.measured_c.toFixed(1)} / ${e.expected_c.toFixed(1)} °C`} />
      <Row k="Fark" v={`${e.diff_c > 0 ? "+" : ""}${e.diff_c.toFixed(1)} °C`} warn={e.diff_c > e.interval_c[1] - e.expected_c} />
      <Row k="%90 aralık" v={`${e.interval_c[0].toFixed(1)} – ${e.interval_c[1].toFixed(1)} °C`} />
      <Row k="Son 24 sa aralık üstü" v={`${e.hours_above_interval_24h} saat`} warn={e.hours_above_interval_24h >= 6} />
      <Row k="Yük (son 6 sa, önceki 7 günün aynı saatleri)" v={`${e.load_z > 0 ? "+" : ""}${e.load_z.toFixed(1)} σ`} warn={e.load_z >= 2} />
      {e.cusum !== null && e.cusum_threshold !== null && (
        <Row k="CUSUM birikimi (hassas kural)" v={`${e.cusum.toFixed(1)} / ${e.cusum_threshold}`} warn={e.cusum >= e.cusum_threshold / 2} />
      )}
      {t.team && (
        <div className="mt-3 rounded-md border border-blue-500/30 bg-blue-500/10 p-2 text-sm text-blue-200">
          Önerilen ekip: <b>{t.team.name}</b> · ~{t.team.distance_km} km <span className="text-xs text-blue-300/70">(temsilî)</span>
        </div>
      )}
      <p className="mt-3 text-xs text-neutral-500">
        İki seviye: “İzlemede” hassas kuraldır (trafo başına haftada ≤1 yanlış uyarı), “Alarm” filo kuralıdır (≤0,25). Uyarı varken yük, önceki 7 günün aynı saatlerine göre olağansa (&lt; 2σ) “soğutma şüphesi”, yüksekse “yük kaynaklı” denir. Model nedeni bilmez; bu sayılar operatörün ilk bakacağı yerlerdir.
      </p>
    </div>
  );
}

export function HealthCard({ t }: { t: TransformerState }) {
  const h = t.health;
  return (
    <div className={box}>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-semibold">Model sağlığı (son 7 gün)</h3>
        <span className={`rounded-full px-2 py-0.5 text-xs ${h.healthy ? "bg-emerald-500/15 text-emerald-300" : "bg-amber-500/15 text-amber-300"}`}>
          {h.healthy ? "Güvenilir" : "Dikkat"}
        </span>
      </div>
      <Row k="%90 aralık kapsaması" v={`%${(h.coverage * 100).toFixed(0)} (alt sınır %${(h.thresholds.coverage_min * 100).toFixed(0)})`} warn={h.coverage < h.thresholds.coverage_min} />
      <Row k="Sistematik sapma |z|" v={`${Math.abs(h.bias_z).toFixed(2)} (≤ ${h.thresholds.abs_bias_z_max})`} warn={Math.abs(h.bias_z) > h.thresholds.abs_bias_z_max} />
      <Row k="Yük kayması" v={`${h.load_shift.toFixed(2)} σ (≤ ${h.thresholds.load_shift_max})`} warn={h.load_shift > h.thresholds.load_shift_max} />
      {h.flags.length > 0 && <p className="mt-2 text-sm text-amber-300">{h.flags.join(" · ")}</p>}
      <p className="mt-3 text-xs text-neutral-500">Eşikler doğrulama döneminin haftalık değerlerinden belirlendi; test verisi eşik seçiminde kullanılmadı.</p>
    </div>
  );
}

export function ModelSummary({ models }: { models: ModelCard[] }) {
  return (
    <div className={`${box} overflow-x-auto`}>
      <h3 className="mb-2 font-semibold">Modeller (test dönemi)</h3>
      <table className="w-full text-sm">
        <thead className="text-left text-neutral-400">
          <tr>
            <th className="py-1 pr-3 font-normal">Seri</th>
            <th className="py-1 pr-3 font-normal">MAE 1/6/24 sa (°C)</th>
            <th className="py-1 pr-3 font-normal">%90 kapsama</th>
            <th className="py-1 pr-3 font-normal">3 °C kademeli · izlemede / alarm</th>
            <th className="py-1 pr-3 font-normal">Yanlış uyarı/hafta · izl. / alarm</th>
            <th className="py-1 font-normal">int8 / gecikme</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {models.map((m) => {
            type Det = { detection_rate: number; median_delay_h: number | null };
            const k3 = m.alarm_rule.test["kademeli_3"] as Det;
            const k3f = m.alarm_rule_fleet?.test["kademeli_3"] as Det | undefined;
            const fpw = m.alarm_rule.test["false_per_week"] as number;
            const fpwf = m.alarm_rule_fleet?.test["false_per_week"] as number | undefined;
            const i8 = m.edge["onnx_int8"];
            return (
              <tr key={m.series} className="border-t border-neutral-800">
                <td className="py-1 pr-3">{m.series}</td>
                <td className="py-1 pr-3">{["h1", "h6", "h24"].map((h) => m.test_mae_c[h].toFixed(2)).join(" / ")}</td>
                <td className="py-1 pr-3">{["h1", "h6", "h24"].map((h) => `%${Math.round(m.test_coverage_90[h] * 100)}`).join(" ")}</td>
                <td className="py-1 pr-3">
                  %{Math.round(k3.detection_rate * 100)} ({k3.median_delay_h} sa) / {k3f ? `%${Math.round(k3f.detection_rate * 100)}` : "–"}
                </td>
                <td className="py-1 pr-3">
                  {fpw.toFixed(2)} / {fpwf !== undefined ? fpwf.toFixed(2) : "–"}
                </td>
                <td className="py-1">
                  {i8?.size_kb} KB · {i8?.latency_cpu_1thread.median_ms.toFixed(2)} ms
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-neutral-500">Arıza tespit oranları yapay olarak eklenen arızalarla ölçülmüştür (gerçek arıza etiketi yok). Gecikme: dizüstü CPU, tek iş parçacığı, batch 1.</p>
    </div>
  );
}
