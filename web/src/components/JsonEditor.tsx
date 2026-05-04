"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

const Monaco = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  runId: string;
  fileName: "probe_confidenced" | "dev_doc_confidenced" | "probe_designs" | "dev_doc";
  initial: unknown;
  height?: number;
  title: string;
  hint?: string;
}

export function JsonEditor({ runId, fileName, initial, height = 420, title, hint }: Props) {
  const qc = useQueryClient();
  const [text, setText] = useState(() =>
    initial ? JSON.stringify(initial, null, 2) : "{}"
  );
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<number | null>(null);

  useEffect(() => {
    if (initial) setText(JSON.stringify(initial, null, 2));
  }, [initial]);

  const save = useMutation({
    mutationFn: async (raw: string) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(raw);
      } catch (e) {
        throw new Error("Invalid JSON: " + (e as Error).message);
      }
      return api.patchFile(runId, fileName, parsed);
    },
    onSuccess: () => {
      setError(null);
      setSavedAt(Date.now());
      qc.invalidateQueries({ queryKey: ["run", runId] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const dirty = (() => {
    if (!initial) return text !== "{}";
    try {
      return JSON.stringify(JSON.parse(text)) !== JSON.stringify(initial);
    } catch {
      return true;
    }
  })();

  return (
    <section className="card overflow-hidden">
      <div className="flex items-center justify-between gap-3 px-5 py-3 border-b border-border">
        <div>
          <div className="font-medium tracking-tight text-[0.95rem]">
            {title}
          </div>
          {hint && (
            <div className="text-xs text-fg-muted mt-0.5">{hint}</div>
          )}
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {dirty && <span className="text-xs text-warn mono">● unsaved</span>}
          {!dirty && savedAt && (
            <span className="text-xs text-pass mono">
              ✓ saved {new Date(savedAt).toLocaleTimeString()}
            </span>
          )}
          <button
            disabled={!dirty || save.isPending}
            onClick={() => save.mutate(text)}
            className="btn btn-primary btn-sm"
          >
            {save.isPending ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
      <div style={{ height }} className="bg-surface">
        <Monaco
          height="100%"
          defaultLanguage="json"
          value={text}
          onChange={(v) => setText(v ?? "")}
          options={{
            fontFamily: "var(--font-geist-mono)",
            fontSize: 13,
            minimap: { enabled: false },
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            wordWrap: "on",
            renderWhitespace: "boundary",
            tabSize: 2,
            padding: { top: 12, bottom: 12 },
          }}
          theme="vs"
        />
      </div>
      {error && (
        <div className="px-5 py-3 border-t border-fail text-fail text-xs mono bg-fail-bg">
          {error}
        </div>
      )}
    </section>
  );
}
