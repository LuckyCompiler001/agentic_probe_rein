"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ChevronDownIcon, ChevronRightIcon } from "@/components/icons";
import { api, type DevPlan } from "@/lib/api";

interface Props {
  runId: string;
  plans: DevPlan[] | undefined;
  selectedPlanIdx: number | null;
}

export function ThresholdWidget({ runId, plans, selectedPlanIdx }: Props) {
  const qc = useQueryClient();
  const initialPlanIdx =
    selectedPlanIdx !== null && selectedPlanIdx >= 0 ? selectedPlanIdx : 0;
  const [planIdx, setPlanIdx] = useState<number | null>(
    plans && plans.length > 0 ? initialPlanIdx : null
  );
  const [value, setValue] = useState("");
  const [open, setOpen] = useState(false);

  const apply = useMutation({
    mutationFn: () => api.thresholdOverride(runId, planIdx, value),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["run", runId] });
      qc.invalidateQueries({ queryKey: ["probe-results", runId] });
      setValue("");
    },
  });

  const currentThreshold =
    planIdx !== null && plans?.[planIdx]
      ? plans[planIdx].threshold
      : null;

  return (
    <div className="card overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-3 px-5 py-3 hover:bg-surface-2 transition-colors"
      >
        <div className="flex items-center gap-2">
          {open ? (
            <ChevronDownIcon className="w-4 h-4 text-fg-muted" />
          ) : (
            <ChevronRightIcon className="w-4 h-4 text-fg-muted" />
          )}
          <span className="font-medium text-[0.95rem]">Threshold override</span>
        </div>
        {currentThreshold !== null && (
          <span className="text-xs text-fg-muted mono">
            current: {currentThreshold}
          </span>
        )}
      </button>
      {open && (
        <div className="px-5 py-4 space-y-3 border-t border-border">
          {plans && plans.length > 0 && (
            <div>
              <label className="block text-xs text-fg-muted mb-1.5">Plan</label>
              <select
                value={planIdx ?? ""}
                onChange={(e) =>
                  setPlanIdx(
                    e.target.value === "" ? null : parseInt(e.target.value, 10)
                  )
                }
                className="input"
              >
                <option value="">(none — auto-research mode)</option>
                {plans.map((p, i) => (
                  <option key={i} value={i}>
                    Plan {i + 1} — {p.metric.slice(0, 60)}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div>
            <label className="block text-xs text-fg-muted mb-1.5">
              New threshold
            </label>
            <input
              type="text"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="0.013"
              className="input mono"
            />
            <div className="text-xs text-fg-muted mt-1.5">
              Accepts a number (0.013), comparison (&lt; 0.03), or full clause
              (AUROC degradation &lt; 0.05). Updates dev plan JSON; if
              prober.py exists, dispatches an agent to propagate.
            </div>
          </div>

          <div className="flex items-center gap-3 pt-1">
            <button
              onClick={() => apply.mutate()}
              disabled={apply.isPending || !value.trim()}
              className="btn btn-primary btn-sm"
            >
              {apply.isPending ? "Applying…" : "Apply"}
            </button>
            {apply.isSuccess && apply.data && (
              <span className="text-xs text-fg-muted">
                {apply.data.agent_dispatched
                  ? "Agent dispatched."
                  : "JSON updated. (prober.py not yet present)"}
              </span>
            )}
            {apply.isError && (
              <span className="text-xs text-fail mono">
                {(apply.error as Error).message}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
