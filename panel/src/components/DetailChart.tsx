"use client";

import { Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TransformerState } from "@/lib/api";

/** Son 72 saat: ölçülen, beklenen (ortanca) ve %90 aralık; sağda 1/6/24 saat ileri tahmin. */
export default function DetailChart({ t }: { t: TransformerState }) {
  const past = (t.history ?? []).map((p) => ({ h: p.h, measured: p.measured, expected: p.expected, band: [p.lower, p.upper] as [number, number] }));
  const future = (t.forecast ?? []).map((p) => ({ h: p.h, forecast: p.expected, fband: [p.lower, p.upper] as [number, number] }));
  const now = past.at(-1);
  const data = [...past, ...(now ? [{ h: 0, forecast: now.measured, fband: [now.measured, now.measured] as [number, number] }] : []), ...future].sort(
    (a, b) => a.h - b.h,
  );
  const tick = { fontSize: 11, fill: "#a3a3a3" };
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#3f3f4640" />
        <XAxis dataKey="h" type="number" domain={[-71, 24]} ticks={[-72, -48, -24, 0, 6, 24]} tick={tick} tickFormatter={(v) => (v === 0 ? "şimdi" : `${v > 0 ? "+" : ""}${v}s`)} />
        <YAxis tick={tick} domain={["auto", "auto"]} unit="°" />
        <Tooltip
          contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }}
          labelFormatter={(v) => (v === 0 ? "şimdi" : `${v} saat`)}
          formatter={(v) => (Array.isArray(v) ? `${Number(v[0]).toFixed(1)} – ${Number(v[1]).toFixed(1)} °C` : `${Number(v).toFixed(2)} °C`)}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area dataKey="band" name={`%90 aralık (${t.explanation.horizon_h} sa önceki tahmin)`} stroke="none" fill="#60a5fa" fillOpacity={0.15} isAnimationActive={false} />
        <Area dataKey="fband" name="İleri tahmin aralığı" stroke="none" fill="#a78bfa" fillOpacity={0.2} isAnimationActive={false} />
        <Line dataKey="expected" name="Beklenen" stroke="#60a5fa" dot={false} strokeDasharray="4 3" isAnimationActive={false} />
        <Line dataKey="measured" name="Ölçülen" stroke="#f87171" dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line dataKey="forecast" name="İleri tahmin" stroke="#a78bfa" strokeWidth={1.5} dot={{ r: 2 }} isAnimationActive={false} connectNulls />
        <ReferenceLine x={0} stroke="#737373" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
