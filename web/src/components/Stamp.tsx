type Tone = "pass" | "fail" | "warn" | "neutral" | "iron" | "moss" | "amber" | "faint";

const TONE_TO_CLASS: Record<Tone, string> = {
  pass: "badge-pass",
  fail: "badge-fail",
  warn: "badge-warn",
  neutral: "badge-neutral",
  // legacy aliases (so we don't have to touch every callsite at once)
  iron: "badge-fail",
  moss: "badge-pass",
  amber: "badge-warn",
  faint: "badge-neutral",
};

export function Stamp({
  tone = "neutral",
  children,
}: {
  tone?: Tone;
  children: React.ReactNode;
}) {
  return <span className={`badge ${TONE_TO_CLASS[tone]}`}>{children}</span>;
}

// Alias for clarity at new call sites.
export const Badge = Stamp;
