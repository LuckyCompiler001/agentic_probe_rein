"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Masthead } from "@/components/Masthead";
import {
  EmptyState,
  NewRunBlock,
  NewRunPlaceholder,
  RunEntry,
} from "@/components/RunEntry";
import { api } from "@/lib/api";

export default function Dashboard() {
  const qc = useQueryClient();
  const router = useRouter();
  const [starting, setStarting] = useState(false);

  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: () => api.listRuns(),
    refetchInterval: 4000,
  });

  const session = useQuery({
    queryKey: ["session"],
    queryFn: () => api.getSession(),
    refetchInterval: 1500,
  });

  const startMutation = useMutation({
    mutationFn: (run_id: string | null) => api.startSession(run_id),
    onSuccess: async () => {
      setStarting(true);
      const start = Date.now();
      while (Date.now() - start < 8000) {
        await new Promise((r) => setTimeout(r, 250));
        const s = await api.getSession();
        if (s.run_id) {
          qc.invalidateQueries({ queryKey: ["session"] });
          router.push(`/runs/${s.run_id}`);
          return;
        }
      }
      setStarting(false);
    },
  });

  const list = runs.data?.runs ?? [];
  const sessionActive = !!session.data?.active;

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-12">
      <Masthead
        rightSlot={
          sessionActive && session.data?.run_id ? (
            <a
              href={`/runs/${session.data.run_id}`}
              className="badge badge-fail flex items-center gap-2"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-fail pulse-soft" />
              Session live · run {session.data.run_id} →
            </a>
          ) : null
        }
      />

      <section className="fadein delay-1">
        <div className="flex items-baseline justify-between mb-3 px-2">
          <h2 className="text-2xl font-semibold tracking-tight">Runs</h2>
          <span className="text-xs text-fg-muted mono">
            {list.length} on file
          </span>
        </div>

        {runs.isLoading ? (
          <div className="text-sm text-fg-muted py-8 px-2">Loading…</div>
        ) : list.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="card overflow-hidden divide-y divide-border">
            {list
              .slice()
              .reverse()
              .map((r, i) => (
                <div
                  key={r.run_id}
                  className="fadein"
                  style={{ animationDelay: `${0.05 + i * 0.03}s` }}
                >
                  <RunEntry run={r} index={list.length - i} />
                </div>
              ))}
          </div>
        )}

        {starting ? (
          <NewRunPlaceholder />
        ) : sessionActive ? (
          <div className="mt-6 px-5 py-4 rounded-[10px] border border-warn/30 bg-warn-bg text-sm text-warn">
            New runs disabled while a session is live.
          </div>
        ) : (
          <NewRunBlock onNew={() => startMutation.mutate(null)} />
        )}
      </section>
    </div>
  );
}
