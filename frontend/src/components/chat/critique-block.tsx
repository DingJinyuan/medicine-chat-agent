"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, ShieldCheck } from "lucide-react";

export function CritiqueBlock({ critique }: { critique: string }) {
  const [expanded, setExpanded] = useState(false);
  if (!critique?.trim()) return null;

  return (
    <div className="mt-2 max-w-full">
      <button
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="inline-flex items-center gap-1.5 rounded-md border bg-muted/50 px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <ShieldCheck className="h-3.5 w-3.5" />
        Fact-check critique
        <motion.span animate={{ rotate: expanded ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown className="h-3 w-3" />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="mt-1.5 rounded-md border bg-muted/50 p-3 text-xs leading-relaxed text-muted-foreground">
              {critique}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
