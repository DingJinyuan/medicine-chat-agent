"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CalendarClock, Siren, Sparkles } from "lucide-react";
import type { TriageLabel, TriagePrediction } from "@/lib/types";

const TRIAGE_CONFIG: Record<
  TriageLabel,
  { label: string; description: string; icon: typeof Siren; classes: string; dot: string }
> = {
  emergency: {
    label: "Seek emergency care",
    description: "This sounds urgent — call your local emergency number or go to the ER.",
    icon: Siren,
    classes: "bg-red-50 border-red-300 text-red-900 dark:bg-red-950/40 dark:border-red-800 dark:text-red-200",
    dot: "bg-red-500",
  },
  urgent: {
    label: "See a clinician soon",
    description: "Consider same-day or next-day care rather than waiting.",
    icon: AlertTriangle,
    classes: "bg-amber-50 border-amber-300 text-amber-900 dark:bg-amber-950/40 dark:border-amber-800 dark:text-amber-200",
    dot: "bg-amber-500",
  },
  routine: {
    label: "Routine — worth scheduling",
    description: "Not time-critical; a regular appointment is a reasonable next step.",
    icon: CalendarClock,
    classes: "bg-sky-50 border-sky-300 text-sky-900 dark:bg-sky-950/40 dark:border-sky-800 dark:text-sky-200",
    dot: "bg-sky-500",
  },
  self_care: {
    label: "Usually manageable at home",
    description: "Often improves with rest, fluids, and over-the-counter care.",
    icon: Sparkles,
    classes: "bg-emerald-50 border-emerald-300 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-800 dark:text-emerald-200",
    dot: "bg-emerald-500",
  },
};

export function TriageBanner({ triage }: { triage: TriagePrediction }) {
  const config = TRIAGE_CONFIG[triage.label];
  if (!config) return null;
  const Icon = config.icon;
  const isEmergency = triage.label === "emergency";

  return (
    <motion.div
      initial={{ opacity: 0, x: -10, scale: 0.96 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`relative flex items-start gap-2.5 overflow-hidden rounded-lg border px-3 py-2.5 text-sm ${config.classes}`}
    >
      <span className="relative mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
        {isEmergency && <span className={`animate-pulse-ring absolute h-4 w-4 rounded-full ${config.dot}`} />}
        <Icon className="relative h-4 w-4" />
      </span>
      <div>
        <p className="font-medium leading-tight">{config.label}</p>
        <p className="text-xs opacity-80 leading-snug mt-0.5">{config.description}</p>
      </div>
    </motion.div>
  );
}
