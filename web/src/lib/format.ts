// Decode the YYYYMMDDHHMMSS run id back to a readable string.
export function parseRunIdToDate(runId: string): Date | null {
  if (!/^\d{14}$/.test(runId)) return null;
  const y = +runId.slice(0, 4);
  const m = +runId.slice(4, 6) - 1;
  const d = +runId.slice(6, 8);
  const hh = +runId.slice(8, 10);
  const mm = +runId.slice(10, 12);
  const ss = +runId.slice(12, 14);
  return new Date(y, m, d, hh, mm, ss);
}

export function formatRunIdDate(runId: string): string {
  const d = parseRunIdToDate(runId);
  if (!d) return runId;
  return d.toLocaleString("en-US", {
    year: "2-digit",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function relativeTime(epochSeconds: number | undefined): string {
  if (!epochSeconds) return "";
  const diffMs = Date.now() - epochSeconds * 1000;
  const sec = Math.floor(diffMs / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 30) return `${day}d ago`;
  const mo = Math.floor(day / 30);
  return `${mo}mo ago`;
}

export function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}
