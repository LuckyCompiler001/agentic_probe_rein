"use client";

import Link from "next/link";

export function Masthead({ rightSlot }: { rightSlot?: React.ReactNode }) {
  return (
    <header className="border-b border-border pb-5 mb-10">
      <div className="flex items-center justify-between gap-6 flex-wrap">
        <Link href="/" className="group flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-fg flex items-center justify-center text-bg text-sm font-mono">
            ◆
          </div>
          <div>
            <div className="text-base font-semibold tracking-tight">
              Agentic Probe
            </div>
            <div className="text-xs text-fg-muted -mt-0.5">
              pipeline console
            </div>
          </div>
        </Link>
        {rightSlot && <div className="flex-1 flex justify-end">{rightSlot}</div>}
      </div>
    </header>
  );
}
