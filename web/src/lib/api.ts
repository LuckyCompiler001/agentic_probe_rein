// Tiny fetch wrapper around the FastAPI backend.
// Single-user, localhost — no auth, no error envelope.

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8765";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

// ── Types matching the backend shapes ────────────────────────────────────────

export interface RunSummary {
  run_id: string;
  steps_done: number;
  current_step: string | null;
  mode?: "normal" | "auto_research" | "unknown";
  project?: string;
  last_activity?: number;
}

export interface ProgressbarStep {
  name: string;
  done: boolean;
  answer?: unknown;
}

export interface Progressbar {
  steps: ProgressbarStep[];
}

export interface ProbeDesign {
  probe_type: string;
  probe_name: string;
  content: string;
  possible_sources: string[];
  confidence: number;
}

export interface DevPlan {
  content: string;
  metric: string;
  threshold: string;
  confidence: number;
}

export interface RunDetail {
  run_id: string;
  progressbar: Progressbar;
  files: {
    probe_designs: { probe_designs: ProbeDesign[] } | null;
    probe_confidenced: { probe_designs: ProbeDesign[] } | null;
    dev_doc: { dev_plans: DevPlan[] } | null;
    dev_doc_confidenced: { dev_plans: DevPlan[] } | null;
  };
}

export interface ProbeResult {
  n: number;
  metric_name: string;
  threshold: number | string;
  values: { epoch: number; value: number }[];
  stats?: { min: number; max: number; mean: number; std: number };
  delta?: number;
  status: "PASS" | "FAIL";
  conclusion?: string;
}

export interface PendingQuestion {
  kind: "yn" | "int_range" | "pos_int" | "text" | "select_run" | "select_project";
  question: string;
  default?: boolean | number | string;
  lo?: number;
  hi?: number;
  existing?: string[];
  available?: string[];
}

export interface SessionState {
  active: boolean;
  run_id: string | null;
  pending: PendingQuestion | null;
  finished: boolean;
  error: string | null;
}

// ── Endpoints ────────────────────────────────────────────────────────────────

export const api = {
  listRuns: () => request<{ runs: RunSummary[] }>("/api/runs"),
  getRun: (id: string) => request<RunDetail>(`/api/runs/${id}`),

  patchFile: (runId: string, name: string, content: unknown) =>
    request<{ saved: string; size: number }>(
      `/api/runs/${runId}/files/${name}`,
      { method: "PATCH", body: JSON.stringify({ content }) }
    ),

  listProjects: () => request<{ projects: string[] }>("/api/projects"),

  listProbeResults: (runId: string) =>
    request<{ results: ProbeResult[] }>(`/api/runs/${runId}/probe-results`),

  plotUrl: (runId: string, n: number) =>
    `${API_BASE}/api/runs/${runId}/probe-plot/${n}`,

  getSession: () => request<SessionState>("/api/session"),
  startSession: (run_id: string | null) =>
    request<{ started: string }>("/api/session/start", {
      method: "POST",
      body: JSON.stringify({ run_id }),
    }),
  answerSession: (value: unknown) =>
    request<{ ok: boolean }>("/api/session/answer", {
      method: "POST",
      body: JSON.stringify({ value }),
    }),
  abortSession: () =>
    request<{ ok: boolean }>("/api/session/abort", { method: "POST" }),

  thresholdOverride: (
    run_id: string,
    plan_idx: number | null,
    new_threshold: string
  ) =>
    request<{ updated_dev_doc: boolean; agent_dispatched: boolean }>(
      "/api/threshold-override",
      {
        method: "POST",
        body: JSON.stringify({ run_id, plan_idx, new_threshold }),
      }
    ),

  logStreamUrl: (run_id: string) =>
    `${API_BASE}/api/runs/${run_id}/log/stream`,

  listSnapshots: (runId: string) =>
    request<{ snapshots: { n: number; size: number }[] }>(
      `/api/runs/${runId}/snapshots`
    ),
  getSnapshot: (runId: string, n: number) =>
    request<{ n: number; content: string }>(`/api/runs/${runId}/snapshots/${n}`),
};
