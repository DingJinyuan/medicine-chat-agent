"use client";

import { motion } from "framer-motion";
import { BadgeCheck, TriangleAlert } from "lucide-react";

interface ConfidenceBannerProps {
  confidenceScore: number | null;
  isReliable: boolean | null;
}

export function ConfidenceBanner({ confidenceScore, isReliable }: ConfidenceBannerProps) {
  if (confidenceScore == null && isReliable == null) return null;

  const reliable = isReliable === true;
  const score =
    confidenceScore != null ? `${(confidenceScore * 100).toFixed(0)}%` : null;
  const Icon = reliable ? BadgeCheck : TriangleAlert;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10, scale: 0.96 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`relative flex items-start gap-2.5 overflow-hidden rounded-lg border px-3 py-2.5 text-sm ${
        reliable
          ? "bg-emerald-50 border-emerald-300 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-800 dark:text-emerald-200"
          : "bg-amber-50 border-amber-300 text-amber-900 dark:bg-amber-950/40 dark:border-amber-800 dark:text-amber-200"
      }`}
    >
      <Icon className="relative mt-0.5 h-4 w-4 shrink-0" />
      <div>
        <p className="font-medium leading-tight">
          {reliable ? "Evidence-grounded" : "Low confidence"}
          {score ? ` · ${score}` : ""}
        </p>
        <p className="text-xs opacity-80 leading-snug mt-0.5">
          {reliable
            ? "Fact-checked against the retrieved literature."
            : "Some claims may be unsupported — review with caution."}
        </p>
      </div>
    </motion.div>
  );
}
