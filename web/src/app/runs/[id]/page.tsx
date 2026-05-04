"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { use, useState } from "react";
import { ActivePrompt } from "@/components/ActivePrompt";
import { JsonEditor } from "@/components/JsonEditor";
import { LogViewer } from "@/components/LogViewer";
import { Masthead } from "@/components/Masthead";
import { ProbeChart } from "@/components/ProbeChart";
import { Stamp } from "@/components/Stamp";
import { Stepper } from "@/components/Stepper";
import { ThresholdWidget } from "@/components/ThresholdWidget";
import { PlayIcon } from "@/components/icons";
import { api } from "@/lib/api";
import { formatRunIdDate } from "@/lib/format";

export default function RunPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const qc = useQueryClient();
  const [resumeError, setResumeError] = useState<string | null>(null);

  const session = useQuery({
    queryKey: ["session"],
    queryFn: () => api.getSession(),
    refetchInterval: 1500,
  });

  const run = useQuery({
    queryKey: ["run", id],
    queryFn: () => api.getRun(id),
    refetchInterval: (q) => {
      const s = qc.getQueryData<typeof session.data>(["session"]);
      return s && s.active && s.run_id === id ? 2000 : 8000;
    },
  });

  const startMutation = useMutation({
    mutationFn: () => api.startSession(id),
    onSuccess: () => {
      setResumeError(null);
      qc.invalidateQueries({ queryKey: ["session"] });
    },
    onError: (e: Error) => setResumeError(e.message),
  });

  const answerMutation = useMutation({
    mutationFn: (value: unknown) => api.answerSession(value),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["session"] });
      qc.invalidateQueries({ queryKey: ["run", id] });
    },
  });

  const sessionForThisRun =
    session.data?.active && session.data.run_id === id;
  const sessionForOtherRun =
    session.data?.active && session.data.run_id && session.data.run_id !== id;

  const progressbar = run.data?.progressbar.steps ?? [];
  const totalSteps = progressbar.length;
  const totalDone = progressbar.filter((s) => s.done).length;

  const arChoice = progressbar.find((s) => s.name === "auto_research_choice");
  const mode = arChoice?.answer ? "auto-research" : "normal";

  const projectStep = progressbar.find((s) => s.name === "select_project");
  const project =
    typeof projectStep?.answer === "string" ? projectStep.answer : null;

  const planSelect = progressbar.find((s) =>
    /cycle_\d+\/plan_select/.test(s.name) && s.done
  );
  const selectedPlanIdx =
    planSelect && typeof planSelect.answer === "number"
      ? (planSelect.answer as number) - 1
      : null;

  const plans = run.data?.files.dev_doc_confidenced?.dev_plans;

  return (
    <div className="mx-auto w-full max-w-[1400px] px-6 py-10">
      <Masthead />

      {/* Run summary header */}
      <div className="fadein flex items-center justify-between gap-6 flex-wrap mb-8">
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-semibold tracking-tight">
            {formatRunIdDate(id)}
          </h1>
          <span className="text-sm text-fg-muted mono">{id}</span>
          <Stamp tone={mode === "auto-research" ? "warn" : "neutral"}>
            {mode}
          </Stamp>
          {project && (
            <Stamp tone="neutral">
              workspace · {project}
            </Stamp>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-fg-muted mono">
            {totalDone} / {totalSteps} steps
          </span>
          {sessionForThisRun ? (
            <span className="badge badge-fail">
              <span className="w-1.5 h-1.5 rounded-full bg-fail pulse-soft" />
              Live
            </span>
          ) : sessionForOtherRun ? (
            <span className="badge badge-warn">Another run is active</span>
          ) : (
            <button
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
              className="btn btn-primary"
            >
              <PlayIcon className="w-3.5 h-3.5" />
              {startMutation.isPending ? "Starting…" : "Resume"}
            </button>
          )}
        </div>
      </div>

      {resumeError && (
        <div className="mb-6 px-4 py-3 rounded-[10px] border border-fail/30 bg-fail-bg text-fail text-sm mono">
          {resumeError}
        </div>
      )}

      <div className="grid grid-cols-12 gap-8">
        <aside className="col-span-12 lg:col-span-3 fadein delay-1">
          <Stepper
            steps={progressbar}
            active={!!sessionForThisRun}
            pendingStepName={null}
          />
        </aside>

        <main className="col-span-12 lg:col-span-9 fadein delay-2 space-y-6">
          {/* Status / active prompt */}
          {sessionForThisRun && session.data?.pending ? (
            <ActivePrompt
              pending={session.data.pending}
              onSubmit={async (v) => {
                await answerMutation.mutateAsync(v);
              }}
              submitting={answerMutation.isPending}
            />
          ) : sessionForThisRun && session.data?.finished ? (
            <div className="card p-5">
              <div className="flex items-center gap-2">
                <span className="badge badge-pass">Session finished</span>
              </div>
              {session.data.error && (
                <div className="mt-3 mono text-sm text-fail">
                  {session.data.error}
                </div>
              )}
            </div>
          ) : sessionForThisRun ? (
            <div className="card p-5">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-warn pulse-soft" />
                <span className="font-medium">Working…</span>
              </div>
              <div className="text-sm text-fg-muted mt-1.5">
                The agent is busy. Watch the log below for progress.
              </div>
            </div>
          ) : sessionForOtherRun ? (
            <div className="card p-5">
              <div className="font-medium">Another session is active</div>
              <div className="text-sm text-fg-muted mt-1.5">
                Run <span className="mono">{session.data!.run_id}</span> holds
                the workspace. Wait for it to finish, or visit it.
              </div>
            </div>
          ) : (
            <div className="card p-5">
              <div className="font-medium">Idle</div>
              <div className="text-sm text-fg-muted mt-1.5">
                Press <span className="mono">Resume</span> above to start the
                pipeline against this run.
              </div>
            </div>
          )}

          <ThresholdWidget
            runId={id}
            plans={plans}
            selectedPlanIdx={selectedPlanIdx}
          />

          <ProbeChart runId={id} />

          {run.data?.files.probe_confidenced && (
            <JsonEditor
              runId={id}
              fileName="probe_confidenced"
              initial={run.data.files.probe_confidenced}
              title="Probe designs (confidenced)"
              hint="Edit any probe — pick one when prompted above"
              height={460}
            />
          )}

          {run.data?.files.dev_doc_confidenced && (
            <JsonEditor
              runId={id}
              fileName="dev_doc_confidenced"
              initial={run.data.files.dev_doc_confidenced}
              title="Development plans (confidenced)"
              hint="Adjust threshold or content. Use the override widget to propagate."
              height={460}
            />
          )}

          <LogViewer runId={id} active={!!sessionForThisRun} />
        </main>
      </div>
    </div>
  );
}
