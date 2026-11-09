"use client";

import type { Status, Team, TransformerState } from "@/lib/api";

export const STATUS_COLOR: Record<Status, string> = {
  normal: "#34d399",
  izlemede: "#fbbf24",
  alarm: "#f87171",
};

export const STATUS_LABEL: Record<Status, string> = {
  normal: "Normal",
  izlemede: "İzlemede",
  alarm: "Alarm",
};

/** Şematik yerleşim (0-100 ızgara). Gerçek bir şehir ya da şebeke değildir. */
export default function FleetMap({
  transformers,
  teams,
  selected,
  onSelect,
}: {
  transformers: TransformerState[];
  teams: Team[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const sel = transformers.find((t) => t.id === selected);
  return (
    <svg viewBox="0 0 100 100" className="h-full w-full" role="img" aria-label="Şematik filo yerleşimi">
      <defs>
        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#262626" strokeWidth="0.2" />
        </pattern>
      </defs>
      <rect width="100" height="100" fill="#0f0f10" />
      <rect width="100" height="100" fill="url(#grid)" />

      {/* Seçili trafodan önerilen ekibe çizgi */}
      {sel?.team && (
        <line x1={sel.x} y1={sel.y} x2={sel.team.x} y2={sel.team.y} stroke="#f87171" strokeWidth="0.4" strokeDasharray="1.2 1" />
      )}

      {teams.map((e) => (
        <g key={e.id}>
          <rect x={e.x - 2} y={e.y - 2} width="4" height="4" rx="0.8" fill="#1e3a8a" stroke="#60a5fa" strokeWidth="0.3" />
          <text x={e.x} y={e.y + 5.2} textAnchor="middle" fontSize="2.6" fill="#93c5fd">
            {e.name}
          </text>
        </g>
      ))}

      {transformers.map((t) => {
        const c = STATUS_COLOR[t.status];
        const isSel = t.id === selected;
        return (
          <g key={t.id} onClick={() => onSelect(t.id)} className="cursor-pointer">
            {t.status === "alarm" && <circle cx={t.x} cy={t.y} r="3.2" fill="none" stroke={c} strokeWidth="0.5" className="pulse-ring" />}
            <circle cx={t.x} cy={t.y} r={isSel ? 3 : 2.4} fill={c} fillOpacity={0.9} stroke={isSel ? "#fff" : "#0a0a0a"} strokeWidth={isSel ? 0.6 : 0.3} />
            <text x={t.x} y={t.y - 4} textAnchor="middle" fontSize="2.8" fill="#e5e5e5" fontWeight={isSel ? 700 : 400}>
              {t.id}
            </text>
            <text x={t.x} y={t.y + 5.5} textAnchor="middle" fontSize="2.3" fill="#a3a3a3">
              {t.measured_c.toFixed(1)} °C
            </text>
          </g>
        );
      })}
    </svg>
  );
}
