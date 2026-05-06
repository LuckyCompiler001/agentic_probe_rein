"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";
import { SectionLabel } from "./ui";

export function LogPanel({
  runId,
  live,
}: {
  runId: string;
  live: boolean;
}) {
  const [lines, setLines] = useState<string[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLines([]);
    let cancelled = false;
    let es: EventSource | null = null;

    if (live) {
      es = new EventSource(`${API_BASE}/api/runs/${runId}/log/stream`);
      es.onmessage = (ev) => {
        if (cancelled) return;
        if (!ev.data || ev.data === "ping") return;
        setLines((prev) => [...prev, ev.data].slice(-1000));
      };
      es.onerror = () => {
        // Silent; reconnect handled by browser.
      };
    } else {
      // Pull static log once.
      fetch(`${API_BASE}/api/runs/${runId}/log`)
        .then((r) => r.json())
        .then((d) => {
          if (cancelled) return;
          const text: string = d.log ?? "";
          setLines(text.split("\n").slice(-1000));
        })
        .catch(() => {});
    }

    return () => {
      cancelled = true;
      es?.close();
    };
  }, [runId, live]);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <div className="rounded-md border border-ink-200 bg-ink-950 overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-ink-800 bg-ink-900">
        <div className="flex gap-1">
          <div className="w-2 h-2 rounded-full bg-red-400/70" />
          <div className="w-2 h-2 rounded-full bg-amber-400/70" />
          <div className="w-2 h-2 rounded-full bg-green-400/70" />
        </div>
        <div className="text-[11px] text-ink-400 font-mono">agent.log</div>
        {live && (
          <span className="ml-auto text-[10px] text-green-300/80 flex items-center gap-1">
            <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
            live
          </span>
        )}
      </div>
      <div
        ref={containerRef}
        className="font-mono text-[11.5px] leading-[1.6] text-ink-200 p-3 h-72 overflow-y-auto whitespace-pre-wrap"
      >
        {lines.length === 0 ? (
          <span className="text-ink-500 italic">no output yet…</span>
        ) : (
          lines.map((l, i) => <div key={i}>{l || " "}</div>)
        )}
      </div>
    </div>
  );
}

export function LogSection({
  runId,
  live,
}: {
  runId: string;
  live: boolean;
}) {
  return (
    <section className="space-y-2">
      <SectionLabel>Agent log</SectionLabel>
      <LogPanel runId={runId} live={live} />
    </section>
  );
}
