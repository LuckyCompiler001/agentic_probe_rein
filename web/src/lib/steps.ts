// Translate the raw progressbar step name into a human-friendly label
// and structural metadata for rendering the stepper.
//
// Step name patterns (from the controller pipeline):
//   setup_confirm                         — preamble
//   auto_research_choice                  — preamble
//   project_context                       — preamble (normal mode)
//   probe_generation                      — preamble
//   probe_confidence                      — preamble
//   cycle_<N>/probe_select                — cycle entry (normal mode)
//   cycle_<N>/dev_doc_generation
//   cycle_<N>/dev_doc_confidence
//   cycle_<N>/plan_select
//   cycle_<N>/implementation
//   cycle_<N>/exception_check_1
//   cycle_<N>/iter_count
//   cycle_<N>/iter_<I>/improve
//   cycle_<N>/iter_<I>/exception_check
//   cycle_<N>/continue_confirm
//   ar/probe_setup                        — auto-research mode preamble
//   ar/exception_check_1
//   ar/comment
//   ar/exception_check_2
//   ar/iter_count
//   ar/iter_<I>/improve
//   ar/iter_<I>/exception_check

import type { ProgressbarStep } from "./api";

export interface ParsedStep {
  name: string;
  label: string;
  kind: "question" | "action" | "exception_check";
  bucket: "preamble" | "cycle" | "iteration" | "auto_research" | "ar_iter";
  cycle?: number;
  iter?: number;
  done: boolean;
  answer?: unknown;
}

const PREAMBLE_LABEL: Record<string, { label: string; kind: ParsedStep["kind"] }> = {
  select_project: { label: "Pick a project workspace", kind: "question" },
  setup_confirm: { label: "Confirm setup", kind: "question" },
  auto_research_choice: { label: "Choose mode", kind: "question" },
  project_context: { label: "Describe the project", kind: "question" },
  probe_generation: { label: "Generate probe designs", kind: "action" },
  probe_confidence: { label: "Score probe confidences", kind: "action" },
};

const CYCLE_LABEL: Record<string, { label: string; kind: ParsedStep["kind"] }> = {
  probe_select: { label: "Select a probe", kind: "question" },
  dev_doc_generation: { label: "Generate development plans", kind: "action" },
  dev_doc_confidence: { label: "Score plan confidences", kind: "action" },
  plan_select: { label: "Select a plan", kind: "question" },
  implementation: { label: "Implement & integrate prober", kind: "action" },
  exception_check_1: { label: "Validate training", kind: "exception_check" },
  iter_count: { label: "Choose iteration count", kind: "question" },
  continue_confirm: { label: "Continue or exit", kind: "question" },
};

const AR_LABEL: Record<string, { label: string; kind: ParsedStep["kind"] }> = {
  probe_setup: { label: "Auto-research probe setup", kind: "action" },
  exception_check_1: { label: "Validate training (1)", kind: "exception_check" },
  comment: { label: "Annotate train.py with improvements", kind: "action" },
  exception_check_2: { label: "Validate training (2)", kind: "exception_check" },
  iter_count: { label: "Choose iteration count", kind: "question" },
};

export function parseStep(step: ProgressbarStep): ParsedStep | null {
  const { name, done, answer } = step;

  if (name in PREAMBLE_LABEL) {
    const { label, kind } = PREAMBLE_LABEL[name];
    return { name, label, kind, bucket: "preamble", done, answer };
  }

  const cycleIterMatch = name.match(/^cycle_(\d+)\/iter_(\d+)\/(improve|exception_check)$/);
  if (cycleIterMatch) {
    const [, c, i, kindStr] = cycleIterMatch;
    return {
      name,
      label: kindStr === "improve" ? `Iterate (round ${+i + 1})` : `Validate training`,
      kind: kindStr === "improve" ? "action" : "exception_check",
      bucket: "iteration",
      cycle: +c,
      iter: +i,
      done,
      answer,
    };
  }

  const cycleMatch = name.match(/^cycle_(\d+)\/(.+)$/);
  if (cycleMatch) {
    const [, c, sub] = cycleMatch;
    if (sub in CYCLE_LABEL) {
      const { label, kind } = CYCLE_LABEL[sub];
      return { name, label, kind, bucket: "cycle", cycle: +c, done, answer };
    }
  }

  const arIterMatch = name.match(/^ar\/iter_(\d+)\/(improve|exception_check)$/);
  if (arIterMatch) {
    const [, i, kindStr] = arIterMatch;
    return {
      name,
      label: kindStr === "improve" ? `Iterate (round ${+i + 1})` : `Validate training`,
      kind: kindStr === "improve" ? "action" : "exception_check",
      bucket: "ar_iter",
      iter: +i,
      done,
      answer,
    };
  }

  if (name.startsWith("ar/")) {
    const sub = name.slice(3);
    if (sub in AR_LABEL) {
      const { label, kind } = AR_LABEL[sub];
      return { name, label, kind, bucket: "auto_research", done, answer };
    }
  }

  return null;
}

export function parseAllSteps(steps: ProgressbarStep[]): ParsedStep[] {
  return steps.map(parseStep).filter((s): s is ParsedStep => s !== null);
}

// Group cycles together (preamble, then each cycle bundle, then ar/ steps).
export interface StepGroup {
  title: string;
  steps: ParsedStep[];
}

export function groupSteps(parsed: ParsedStep[]): StepGroup[] {
  const groups: StepGroup[] = [];
  const preamble = parsed.filter((s) => s.bucket === "preamble");
  if (preamble.length) groups.push({ title: "Preamble", steps: preamble });

  // cycles
  const cycles = new Map<number, ParsedStep[]>();
  parsed.filter((s) => s.bucket === "cycle" || s.bucket === "iteration").forEach((s) => {
    const c = s.cycle ?? 0;
    if (!cycles.has(c)) cycles.set(c, []);
    cycles.get(c)!.push(s);
  });
  for (const [c, steps] of [...cycles.entries()].sort((a, b) => a[0] - b[0])) {
    groups.push({ title: `Cycle ${c + 1}`, steps });
  }

  const ar = parsed.filter((s) => s.bucket === "auto_research" || s.bucket === "ar_iter");
  if (ar.length) groups.push({ title: "Auto-research", steps: ar });

  return groups;
}
