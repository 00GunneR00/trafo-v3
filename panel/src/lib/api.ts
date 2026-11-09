export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Status = "normal" | "izlemede" | "alarm";
export type Label = "sogutma_suphesi" | "yuk_kaynakli" | "normal";

export interface Explanation {
  status: Status;
  label: Label;
  text: string;
  measured_c: number;
  expected_c: number;
  interval_c: [number, number];
  diff_c: number;
  hours_above_interval_24h: number;
  load_z: number;
  cusum: number | null;
  cusum_threshold: number | null;
  horizon_h: number;
  alarm: boolean;
  warning_age_h: Partial<Record<"izlemede" | "alarm", number>>;
}

export interface Health {
  coverage: number;
  bias_z: number;
  load_shift: number;
  flags: string[];
  healthy: boolean;
  thresholds: { coverage_min: number; abs_bias_z_max: number; load_shift_max: number };
}

export interface Team {
  id: string;
  name: string;
  x: number;
  y: number;
  distance_km?: number;
}

export interface ScenarioInfo {
  kind: "sogutma" | "ani" | "yuk";
  magnitude: number;
  start_hour: number;
  hours: number;
}

export interface TransformerState {
  id: string;
  series: string;
  x: number;
  y: number;
  zone: string;
  hour: number;
  status: Status;
  measured_c: number;
  expected_c: number;
  explanation: Explanation;
  health: Health;
  scenario: ScenarioInfo | null;
  team?: Team;
  history?: { h: number; measured: number; expected: number; lower: number; upper: number }[];
  forecast?: { h: number; expected: number; lower: number; upper: number }[];
}

export interface FleetOverview {
  hour: number;
  sim_hours: number;
  day: number;
  clock: string;
  transformers: TransformerState[];
  teams: Team[];
  counts: Record<Status, number>;
}

export interface AlarmRule {
  kind: string;
  k?: number;
  H?: number;
  n?: number;
  horizon_h: number;
  test: Record<string, { detection_rate: number; median_delay_h: number | null } | number>;
}

export interface ModelCard {
  series: string;
  version: string;
  architecture: string;
  test_mae_c: Record<string, number>;
  test_coverage_90: Record<string, number>;
  test_width_90_c: Record<string, number>;
  alarm_rule: AlarmRule;
  alarm_rule_fleet: AlarmRule | null;
  edge: Record<string, { size_kb: number; latency_cpu_1thread: { median_ms: number; p95_ms: number } }>;
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${API_URL}${path}`, { signal, cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export const fetchFleet = (hour: number, signal?: AbortSignal) => get<FleetOverview>(`/fleet?hour=${hour}`, signal);
export const fetchTransformer = (id: string, hour: number, signal?: AbortSignal) =>
  get<TransformerState>(`/transformers/${id}?hour=${hour}`, signal);
export const fetchModel = () => get<{ models: ModelCard[] }>(`/model`);

export async function postScenario(body: { id: string; kind: string; magnitude: number; start_hour: number }) {
  const r = await fetch(`${API_URL}/scenario`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
}

export async function clearScenario(id?: string) {
  await fetch(`${API_URL}/scenario${id ? `/${id}` : ""}`, { method: "DELETE" });
}
