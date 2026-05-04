"use client";

import { useEffect, useRef, useState } from "react";
import type { PendingQuestion } from "@/lib/api";
import { CheckIcon, XIcon } from "@/components/icons";

interface Props {
  pending: PendingQuestion;
  onSubmit: (value: unknown) => Promise<void>;
  submitting: boolean;
}

export function ActivePrompt({ pending, onSubmit, submitting }: Props) {
  return (
    <div className="card p-6">
      <div className="flex items-center gap-2 text-sm">
        <span className="w-2 h-2 rounded-full bg-warn pulse-soft" />
        <span className="h-section !text-warn">Awaiting input</span>
      </div>
      <p className="mt-3 text-[0.95rem] leading-relaxed whitespace-pre-line text-fg">
        {pending.question}
      </p>
      <div className="mt-5">
        {pending.kind === "yn" && (
          <YnInput pending={pending} onSubmit={onSubmit} submitting={submitting} />
        )}
        {pending.kind === "int_range" && (
          <IntRangeInput pending={pending} onSubmit={onSubmit} submitting={submitting} />
        )}
        {pending.kind === "pos_int" && (
          <PosIntInput pending={pending} onSubmit={onSubmit} submitting={submitting} />
        )}
        {pending.kind === "text" && (
          <TextInput pending={pending} onSubmit={onSubmit} submitting={submitting} />
        )}
        {pending.kind === "select_project" && (
          <SelectProjectInput pending={pending} onSubmit={onSubmit} submitting={submitting} />
        )}
        {pending.kind === "select_run" && (
          <div className="text-sm text-fg-muted">
            (select_run is handled at the dashboard level — not expected here)
          </div>
        )}
      </div>
    </div>
  );
}

// ── Input variants ───────────────────────────────────────────────────────────

function YnInput({ pending, onSubmit, submitting }: Props) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <button
        className="btn btn-primary"
        onClick={() => onSubmit(true)}
        disabled={submitting}
      >
        <CheckIcon className="w-3.5 h-3.5" /> Yes
      </button>
      <button
        className="btn btn-secondary"
        onClick={() => onSubmit(false)}
        disabled={submitting}
      >
        <XIcon className="w-3.5 h-3.5" /> No
      </button>
      {pending.default !== undefined && pending.default !== null && (
        <span className="text-xs text-fg-muted ml-2">
          default: {pending.default ? "yes" : "no"}
        </span>
      )}
    </div>
  );
}

function IntRangeInput({ pending, onSubmit, submitting }: Props) {
  const [val, setVal] = useState<number>(pending.lo ?? 1);
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(val);
      }}
      className="flex items-center gap-2 flex-wrap"
    >
      <input
        type="number"
        min={pending.lo}
        max={pending.hi}
        value={val}
        onChange={(e) => setVal(+e.target.value)}
        className="input mono w-24"
      />
      <button className="btn btn-primary" disabled={submitting}>
        Submit
      </button>
      <span className="text-xs text-fg-muted ml-1">
        integer in [{pending.lo}, {pending.hi}]
      </span>
    </form>
  );
}

function PosIntInput({ pending, onSubmit, submitting }: Props) {
  const [val, setVal] = useState<string>(String(pending.default ?? 3));
  const submit = (v: string) => {
    const n = parseInt(v || String(pending.default ?? 3), 10);
    if (!Number.isFinite(n) || n <= 0) return;
    onSubmit(n);
  };
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit(val);
      }}
      className="flex items-center gap-2 flex-wrap"
    >
      <input
        type="number"
        min={1}
        value={val}
        onChange={(e) => setVal(e.target.value)}
        className="input mono w-24"
      />
      <button className="btn btn-primary" disabled={submitting}>
        Submit
      </button>
      <button
        type="button"
        onClick={() => submit(String(pending.default ?? 3))}
        className="btn btn-ghost btn-sm"
      >
        Use default ({pending.default})
      </button>
    </form>
  );
}

function SelectProjectInput({ pending, onSubmit, submitting }: Props) {
  const available = pending.available ?? [];
  const initial =
    typeof pending.default === "string" && available.includes(pending.default)
      ? pending.default
      : available[0] ?? "";
  const [val, setVal] = useState<string>(initial);
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (val) onSubmit(val);
      }}
      className="space-y-3"
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {available.map((name) => (
          <label
            key={name}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border cursor-pointer transition-colors ${
              val === name
                ? "border-fg bg-surface-2"
                : "border-border hover:border-border-strong"
            }`}
          >
            <input
              type="radio"
              name="project"
              value={name}
              checked={val === name}
              onChange={() => setVal(name)}
              className="accent-fg"
            />
            <span className="font-medium text-sm">{name}</span>
            {name === pending.default && (
              <span className="ml-auto badge badge-neutral">default</span>
            )}
          </label>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <button className="btn btn-primary" disabled={submitting || !val}>
          Use this workspace
        </button>
        <span className="text-xs text-fg-muted">
          The pipeline will run inside <span className="mono">project/{val}</span>.
        </span>
      </div>
    </form>
  );
}

function TextInput({ pending: _pending, onSubmit, submitting }: Props) {
  const [val, setVal] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => ref.current?.focus(), []);
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(val);
      }}
    >
      <textarea
        ref={ref}
        value={val}
        onChange={(e) => setVal(e.target.value)}
        rows={5}
        placeholder="Describe the project, dataset, and goal in 1–3 sentences…"
        className="input leading-relaxed"
      />
      <div className="mt-3 flex items-center justify-between gap-3 flex-wrap">
        <span className="text-xs text-fg-muted">
          {val.length} characters
        </span>
        <button
          className="btn btn-primary"
          disabled={submitting || val.trim().length === 0}
        >
          Submit
        </button>
      </div>
    </form>
  );
}
