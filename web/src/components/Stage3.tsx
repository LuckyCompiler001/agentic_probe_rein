"use client";

import { useState } from "react";
import { api, RunRecord } from "@/lib/api";
import { Button, Pill, SectionLabel, Spinner } from "./ui";
import { Header } from "./Stage1";
import { LogSection } from "./LogPanel";

export function Stage3({
  run,
  onUpdate,
}: {
  run: RunRecord;
  onUpdate: () => void;
}) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleImplement() {
    setRunning(true);
    setError(null);
    try {
      await api.implement(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setRunning(false);
    }
  }

  const hasFirstRun = run.iterations.length > 0;
  const firstRun = hasFirstRun ? run.iterations[0] : null;
  const stage3Done = run.stage > 3 || (run.stage === 3 && run.phase === "done");

  return (
    <div className="space-y-6">
      <Header
        n={3}
        title="Implementation"
        subtitle="The code agent writes prober.py and integrates it into train.py, then runs training once to validate the integration."
      />

      <section className="space-y-2">
        <SectionLabel>Selected dev plan</SectionLabel>
        <div className="rounded-md border border-ink-200 bg-ink-50/50 px-3 py-2 text-[12.5px] text-ink-700">
          plan #{run.plan_index} from stage 2 (locked)
        </div>
      </section>

      <section className="flex items-center gap-3">
        {!stage3Done && (
          <Button onClick={handleImplement} disabled={running || run.busy}>
            {running ? (
              <>
                <Spinner /> Running agent…
              </>
            ) : (
              "Implement & Run"
            )}
          </Button>
        )}
        {stage3Done && <Pill tone="pass">implementation complete</Pill>}
      </section>

      {firstRun && (
        <section className="space-y-2">
          <SectionLabel>First training result</SectionLabel>
          <div className="rounded-md border border-ink-200 bg-white p-4">
            <div className="flex items-center gap-3">
              <Pill tone={firstRun.status === "PASS" ? "pass" : "fail"}>
                {firstRun.status ?? "—"}
              </Pill>
              <span className="font-mono text-[12px] text-ink-700">
                {firstRun.metric_name}
              </span>
              <span className="ml-auto font-mono tabular-nums text-[14px] text-ink-950">
                {firstRun.metric_value !== null ? firstRun.metric_value.toFixed(4) : "—"}
              </span>
            </div>
            <div className="mt-2 text-[11px] text-ink-500">
              threshold: <span className="font-mono">{firstRun.threshold ?? "—"}</span>
            </div>
          </div>
        </section>
      )}

      {error && (
        <div className="px-3 py-2 rounded-md text-[12px] text-red-600 bg-red-50 border border-red-100">
          {error}
        </div>
      )}

      <LogSection runId={run.run_id} live={running || run.busy} />
    </div>
  );
}
