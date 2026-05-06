"use client";

import { useEffect, useState } from "react";
import { api, ProbeDesign, RunRecord } from "@/lib/api";
import { Button, Card, ConfidenceBar, Pill, SectionLabel, Spinner } from "./ui";

export function Stage1({
  run,
  onUpdate,
}: {
  run: RunRecord;
  onUpdate: () => void;
}) {
  const [context, setContext] = useState(run.context ?? "");
  const [designs, setDesigns] = useState<ProbeDesign[] | null>(null);
  const [generating, setGenerating] = useState(false);
  const [picking, setPicking] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setContext(run.context ?? "");
    if (run.stage >= 1 && (run.phase === "generated" || run.stage > 1)) {
      api
        .getStage1(run.run_id)
        .then((r) => setDesigns(r.probe_designs ?? null))
        .catch(() => {});
    } else {
      setDesigns(null);
    }
  }, [run.run_id, run.stage, run.phase, run.context]);

  async function handleGenerate() {
    setError(null);
    setGenerating(true);
    try {
      await api.setContext(run.run_id, context);
      await api.generateProbes(run.run_id);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setGenerating(false);
    }
  }

  async function handleSelect(idx: number) {
    setPicking(idx);
    setError(null);
    try {
      await api.selectProbe(run.run_id, idx);
      onUpdate();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setPicking(null);
    }
  }

  const selected = run.probe_index;

  return (
    <div className="space-y-6">
      <Header
        n={1}
        title="Probe Design"
        subtitle="Describe your project; the model will propose probe designs and rate its own confidence."
      />

      {/* Context input */}
      <section>
        <SectionLabel>Project context</SectionLabel>
        <textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          rows={4}
          className="mt-2 w-full rounded-md border border-ink-200 bg-white px-3 py-2 text-[13px] focus:border-ink-400"
          placeholder="One or two sentences about the project + dataset description."
          disabled={run.stage > 1}
        />
        <div className="mt-2 flex items-center gap-2">
          {run.stage === 1 && (
            <Button
              onClick={handleGenerate}
              disabled={generating || !context.trim()}
            >
              {generating ? (
                <>
                  <Spinner /> Generating…
                </>
              ) : designs ? (
                "Regenerate"
              ) : (
                "Generate Probes"
              )}
            </Button>
          )}
          {run.stage > 1 && (
            <Pill tone="pass">
              probe #{run.probe_index} selected · stage {run.stage}
            </Pill>
          )}
        </div>
      </section>

      {/* Designs list */}
      {designs && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <SectionLabel>Probe candidates ({designs.length})</SectionLabel>
            <div className="text-[11px] text-ink-500">
              {run.stage === 1 ? "select one to continue" : "selection locked"}
            </div>
          </div>
          {designs.map((d, i) => {
            const idx = i + 1;
            const isSel = selected === idx;
            return (
              <Card key={i} selected={isSel} className="p-4">
                <div className="flex items-start gap-3">
                  <div className="font-mono text-[11px] text-ink-500 w-7 mt-0.5">
                    {String(idx).padStart(2, "0")}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Pill tone="neutral">{d.probe_type}</Pill>
                      <span className="font-medium text-[13px] text-ink-900">
                        {d.probe_name}
                      </span>
                      <div className="ml-auto">
                        <ConfidenceBar value={d.confidence} />
                      </div>
                    </div>
                    <p className="mt-2 text-[12.5px] text-ink-700 leading-relaxed">
                      {d.content}
                    </p>
                    {d.possible_sources?.length > 0 && (
                      <div className="mt-2 text-[11px] text-ink-500">
                        <span className="uppercase tracking-wide font-medium">
                          sources:
                        </span>{" "}
                        {d.possible_sources.join(" · ")}
                      </div>
                    )}
                    {run.stage === 1 && (
                      <div className="mt-3">
                        <Button
                          size="sm"
                          variant={isSel ? "primary" : "secondary"}
                          onClick={() => handleSelect(idx)}
                          disabled={picking !== null}
                        >
                          {picking === idx ? <Spinner /> : isSel ? "Selected" : "Select & Continue"}
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </section>
      )}

      {error && (
        <div className="px-3 py-2 rounded-md text-[12px] text-red-600 bg-red-50 border border-red-100">
          {error}
        </div>
      )}
    </div>
  );
}

export function Header({
  n,
  title,
  subtitle,
}: {
  n: number;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="border-b border-ink-200 pb-4">
      <div className="flex items-baseline gap-3">
        <div className="font-mono text-[11px] text-ink-500">STAGE {n}</div>
        <h1 className="text-xl font-semibold tracking-tight text-ink-950">{title}</h1>
      </div>
      {subtitle && (
        <p className="mt-1.5 text-[13px] text-ink-600">{subtitle}</p>
      )}
    </div>
  );
}
