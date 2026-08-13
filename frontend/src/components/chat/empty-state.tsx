"use client";

import { motion, type Variants } from "framer-motion";
import { Activity, Brain, Sparkles, Sun } from "lucide-react";

const EXAMPLE_PROMPTS = [
  { text: "What are common symptoms of type 2 diabetes?", icon: Activity },
  { text: "What triggers migraines?", icon: Brain },
  { text: "How is seasonal allergy different from a cold?", icon: Sparkles },
  { text: "What should I do about a mild sunburn?", icon: Sun },
];

const container: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08, delayChildren: 0.15 } },
};

const item: Variants = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut" } },
};

export function EmptyState({ onSelectPrompt }: { onSelectPrompt: (prompt: string) => void }) {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="flex flex-1 flex-col items-center justify-center gap-7 px-4 text-center"
    >
      <motion.div
        variants={item}
        animate={{ y: [0, -8, 0] }}
        transition={{ y: { duration: 4, repeat: Infinity, ease: "easeInOut" } }}
        className="relative flex h-16 w-16 items-center justify-center rounded-3xl bg-gradient-to-br from-primary/20 to-primary/5 shadow-lg shadow-primary/10 ring-1 ring-primary/10"
      >
        <span className="text-3xl">🩺</span>
      </motion.div>

      <motion.div variants={item} className="space-y-1.5">
        <h2 className="text-xl font-semibold tracking-tight">Ask MediSense a health question</h2>
        <p className="max-w-sm text-sm text-muted-foreground">
          General health information grounded in MedlinePlus, with a triage
          check and source citations on every answer.
        </p>
      </motion.div>

      <motion.div variants={item} className="grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {EXAMPLE_PROMPTS.map(({ text, icon: Icon }) => (
          <motion.button
            key={text}
            variants={item}
            whileHover={{ y: -2, transition: { duration: 0.15 } }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onSelectPrompt(text)}
            className="group flex items-center gap-2.5 rounded-xl border bg-card/60 px-3.5 py-3 text-left text-xs text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:border-primary/40 hover:bg-card hover:text-foreground"
          >
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary transition-transform group-hover:scale-110">
              <Icon className="h-3.5 w-3.5" />
            </span>
            {text}
          </motion.button>
        ))}
      </motion.div>
    </motion.div>
  );
}
