"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

interface Props {
  runId: string;
  active: boolean;
}

export function LogViewer({ runId, active }: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [follow, setFollow] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLines([]);
    const url = api.logStreamUrl(runId);
    const es = new EventSource(url);
    es.onmessage = (ev) => {
      try {
        const text = JSON.parse(ev.data) as string;
        setLines((prev) => {
          const next = [...prev, text];
          return next.length > 4000 ? next.slice(-4000) : next;
        });
      } catch {}
    };
    es.onerror = () => {};
    return () => es.close();
  }, [runId]);

  useEffect(() => {
    if (follow && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines, follow]);

  return (
    <section className="card overflow-hidden">
      <div className="flex items-center justify-between gap-3 px-5 py-3 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="h-section !text-fg">Log</div>
          <span
            className={`badge ${active ? "badge-fail" : "badge-neutral"}`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                active ? "bg-fail pulse-soft" : "bg-fg-subtle"
              }`}
            />
            {active ? "live" : "idle"}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setFollow((f) => !f)}
            className="btn btn-ghost btn-sm"
          >
            {follow ? "Follow: on" : "Follow: off"}
          </button>
          <button
            type="button"
            onClick={() => setLines([])}
            className="btn btn-ghost btn-sm"
          >
            Clear
          </button>
        </div>
      </div>
      <div
        ref={containerRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          const atBottom =
            el.scrollHeight - el.scrollTop - el.clientHeight < 16;
          if (!atBottom && follow) setFollow(false);
        }}
        className="mono text-[0.78rem] text-fg leading-[1.6] whitespace-pre overflow-auto h-[26rem] px-5 py-4 bg-surface-2/40"
      >
        {lines.length === 0 ? (
          <div className="text-fg-muted">
            No output yet — the worker hasn’t written anything.
          </div>
        ) : (
          lines.map((l, i) => (
            <div key={i} className="flex gap-3">
              <span className="text-fg-subtle select-none w-10 text-right shrink-0">
                {i + 1}
              </span>
              <span className="flex-1">{l}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
