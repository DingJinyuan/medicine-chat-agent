"use client";

import { motion } from "framer-motion";
import { Activity, Brain, FlaskConical, Microscope } from "lucide-react";
import { staggerContainer as container, staggerItem as item } from "@/lib/motion";

const EXAMPLE_PROMPTS = [
  {
    text: "What are the evidence-based first-line treatments for type 2 diabetes?",
    icon: Activity,
  },
  {
    text: "Is metformin safe in patients with chronic kidney disease stage 3?",
    icon: FlaskConical,
  },
  {
    text: "What's the guideline recommendation for hypertension in older adults?",
    icon: Brain,
  },
  {
    text: "Compare SGLT2 inhibitors versus DPP-4 inhibitors for glycemic control.",
    icon: Microscope,
  },
];

export function ExpertEmptyState({ onSelectPrompt }: { onSelectPrompt: (prompt: string) => void }) {
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
        className="relative flex h-16 w-16 items-center justify-center rounded-3xl bg-gradient-to-br from-teal-500/20 to-emerald-500/5 shadow-lg shadow-teal-500/10 ring-1 ring-teal-500/10"
      >
        <span className="text-3xl">🔬</span>
      </motion.div>

      <motion.div variants={item} className="space-y-1.5">
        <h2 className="text-xl font-semibold tracking-tight">Ask Clinical AI a clinical question</h2>
        <p className="max-w-sm text-sm text-muted-foreground">
          Evidence-based answers synthesized from PubMed and clinical guidelines,
          with fact-checking, confidence scoring, and citations on every answer.
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
            className="group flex items-center gap-2.5 rounded-xl border bg-card/60 px-3.5 py-3 text-left text-xs text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:border-teal-500/40 hover:bg-card hover:text-foreground"
          >
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-teal-500/10 text-teal-600 transition-transform group-hover:scale-110 dark:text-teal-400">
              <Icon className="h-3.5 w-3.5" />
            </span>
            {text}
          </motion.button>
        ))}
      </motion.div>
    </motion.div>
  );
}
