"use client";

import { CheckIcon } from "@/components/icons";
import type { ProgressbarStep } from "@/lib/api";
import { groupSteps, parseAllSteps, type ParsedStep } from "@/lib/steps";

interface Props {
  steps: ProgressbarStep[];
  active: boolean;
  pendingStepName?: string | null;
}

export function Stepper({ steps, active, pendingStepName }: Props) {
  const parsed = parseAllSteps(steps);
  const groups = groupSteps(parsed);
  const totalDone = parsed.filter((s) => s.done).length;

  return (
    <aside className="text-sm">
      <div className="flex items-baseline justify-between mb-4">
        <div className="h-section">Pipeline</div>
        <div className="text-xs text-fg-subtle mono">
          {totalDone}/{parsed.length}
        </div>
      </div>
      <ol className="space-y-6">
        {groups.map((g, gi) => (
          <li key={gi}>
            <div className="text-xs text-fg-muted mb-2.5">{g.title}</div>
            <ol className="relative">
              <span
                aria-hidden
                className="absolute left-[7px] top-1.5 bottom-1.5 w-px bg-border"
              />
              {g.steps.map((s) => {
                const isPending = active && pendingStepName === s.name;
                return <StepRow key={s.name} step={s} pending={isPending} />;
              })}
            </ol>
          </li>
        ))}
      </ol>
    </aside>
  );
}

function StepRow({ step, pending }: { step: ParsedStep; pending: boolean }) {
  let dot: React.ReactNode;
  if (step.done) {
    dot = (
      <span className="relative z-[1] inline-flex w-[15px] h-[15px] rounded-full bg-fg items-center justify-center">
        <CheckIcon className="w-2.5 h-2.5 text-bg" strokeWidth={3} />
      </span>
    );
  } else if (pending) {
    dot = (
      <span className="relative z-[1] inline-flex w-[15px] h-[15px] rounded-full ring-2 ring-fg bg-bg items-center justify-center">
        <span className="w-1.5 h-1.5 rounded-full bg-fg pulse-soft" />
      </span>
    );
  } else {
    dot = (
      <span className="relative z-[1] inline-flex w-[15px] h-[15px] rounded-full ring-1 ring-border bg-bg" />
    );
  }

  const labelClass = step.done
    ? "text-fg"
    : pending
    ? "text-fg font-medium"
    : "text-fg-subtle";

  return (
    <li className="flex items-start gap-2.5 py-1.5">
      <span className="mt-[2px]">{dot}</span>
      <div className="min-w-0 flex-1">
        <div className={`text-[0.86rem] leading-snug ${labelClass}`}>
          {step.label}
        </div>
        {step.done && step.answer !== undefined && step.answer !== null && (
          <div className="text-[0.72rem] text-fg-subtle mono mt-0.5 truncate">
            {String(
              typeof step.answer === "boolean"
                ? step.answer
                  ? "yes"
                  : "no"
                : step.answer
            )}
          </div>
        )}
      </div>
      {pending && <span className="badge badge-warn shrink-0">pending</span>}
    </li>
  );
}
