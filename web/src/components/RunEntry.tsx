"use client";

import Link from "next/link";
import { ArrowRightIcon } from "@/components/icons";
import type { RunSummary } from "@/lib/api";
import { formatRunIdDate, relativeTime } from "@/lib/format";

export function RunEntry({ run }: { run: RunSummary; index: number }) {
  const datestr = formatRunIdDate(run.run_id);
  const ago = relativeTime(run.last_activity);
  const modeLabel =
    run.mode === "auto_research"
      ? "Auto-research"
      : run.mode === "normal"
      ? "Normal"
      : "—";

  return (
    <Link
      href={`/runs/${run.run_id}`}
      className="group block px-5 py-4 row-hover border border-transparent hover:border-border"
    >
      <div className="flex items-center justify-between gap-6">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-3">
            <span className="font-medium tracking-tight">{datestr}</span>
            <span className="text-xs text-fg-subtle">·</span>
            <span className="text-xs text-fg-muted mono">{run.run_id}</span>
          </div>
          <div className="mt-1 flex items-center gap-3 text-sm text-fg-muted">
            {run.project && (
              <>
                <span className="mono">{run.project}</span>
                <span className="text-fg-subtle">·</span>
              </>
            )}
            <span>{modeLabel}</span>
            <span className="text-fg-subtle">·</span>
            <span>{run.steps_done} steps complete</span>
            {ago && (
              <>
                <span className="text-fg-subtle">·</span>
                <span>{ago}</span>
              </>
            )}
          </div>
        </div>
        <ArrowRightIcon className="w-4 h-4 text-fg-subtle group-hover:text-fg group-hover:translate-x-0.5 transition-all" />
      </div>
    </Link>
  );
}

export function NewRunBlock({ onNew }: { onNew: () => void }) {
  return (
    <button
      onClick={onNew}
      className="w-full mt-6 px-5 py-4 rounded-[10px] border border-dashed border-border-strong hover:border-fg hover:bg-surface-2 transition-all text-left group"
    >
      <div className="flex items-center justify-between">
        <div>
          <div className="font-medium tracking-tight">Begin a new run</div>
          <div className="text-sm text-fg-muted mt-0.5">
            Opens a fresh entry, asks for project context, generates probe designs.
          </div>
        </div>
        <span className="text-fg-muted group-hover:text-fg transition-colors text-xl leading-none">
          +
        </span>
      </div>
    </button>
  );
}

export function NewRunPlaceholder() {
  return (
    <div className="w-full mt-6 px-5 py-6 rounded-[10px] border border-border bg-surface text-center">
      <div className="font-medium pulse-soft">Starting a new run…</div>
      <div className="text-sm text-fg-muted mt-1">
        Redirecting to the run page.
      </div>
    </div>
  );
}

export function EmptyState() {
  return (
    <div className="text-center py-16 px-6 rounded-[10px] border border-dashed border-border-strong">
      <div className="font-medium tracking-tight">No runs yet</div>
      <div className="text-sm text-fg-muted mt-1">
        Begin a new run to populate this list.
      </div>
    </div>
  );
}
