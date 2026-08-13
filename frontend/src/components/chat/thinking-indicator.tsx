"use client";

import { motion } from "framer-motion";
import { Stethoscope } from "lucide-react";

const dotTransition = (delay: number) => ({
  duration: 0.9,
  repeat: Infinity,
  ease: "easeInOut" as const,
  delay,
});

export function ThinkingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="flex items-center gap-3 pl-11"
    >
      <div className="flex items-center gap-2 rounded-2xl border bg-card px-3.5 py-2.5">
        <Stethoscope className="h-3.5 w-3.5 text-primary" />
        <div className="flex items-center gap-1">
          {[0, 0.15, 0.3].map((delay) => (
            <motion.span
              key={delay}
              className="h-1.5 w-1.5 rounded-full bg-primary"
              animate={{ opacity: [0.3, 1, 0.3], y: [0, -3, 0] }}
              transition={dotTransition(delay)}
            />
          ))}
        </div>
      </div>
    </motion.div>
  );
}
