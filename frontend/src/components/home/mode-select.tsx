"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, HeartPulse, Microscope, Sparkles } from "lucide-react";
import { AnimatedBackground } from "@/components/chat/animated-background";
import { ThemeToggle } from "@/components/theme-toggle";
import { staggerContainer as container, staggerItem as item } from "@/lib/motion";

const MODES = [
  {
    href: "/expert",
    icon: Microscope,
    title: "Clinician Mode",
    subtitle: "For physicians & researchers",
    description:
      "Multi-agent clinical RAG with fact-checking and ML readmission risk prediction, grounded in PubMed abstracts and clinical guidelines.",
    tags: ["Clinical RAG", "Fact-check", "Risk ML"],
    accent: "from-teal-500/20 to-emerald-500/5",
    iconClass: "bg-teal-500/10 text-teal-600 dark:text-teal-400",
    ring: "hover:border-teal-500/40",
  },
  {
    href: "/patient",
    icon: HeartPulse,
    title: "MediSense",
    subtitle: "For patients & the public",
    description:
      "Triage-safe health Q&A with emergency shortcutting and safety guardrails, grounded in MedlinePlus health topics.",
    tags: ["Triage", "Safety guardrails", "MedlinePlus"],
    accent: "from-sky-500/20 to-blue-500/5",
    iconClass: "bg-sky-500/10 text-sky-600 dark:text-sky-400",
    ring: "hover:border-sky-500/40",
  },
];

export function ModeSelect() {
  return (
    <div className="relative flex min-h-dvh flex-col">
      <AnimatedBackground />

      <header className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2.5">
          <motion.div
            whileHover={{ rotate: [0, -8, 8, 0], transition: { duration: 0.4 } }}
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10"
          >
            <Sparkles className="h-4 w-4 text-primary" />
          </motion.div>
          <span className="text-sm font-semibold">Medicine AI Pro</span>
        </div>
        <ThemeToggle />
      </header>

      <motion.main
        variants={container}
        initial="hidden"
        animate="show"
        className="flex flex-1 flex-col items-center justify-center gap-8 px-4 py-10 text-center"
      >
        <motion.div variants={item} className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            Two clinical AI workflows, one app
          </h1>
          <p className="mx-auto max-w-md text-sm text-muted-foreground">
            Choose your entry point — the clinician-facing RAG assistant or the
            patient-facing triage chatbot.
          </p>
        </motion.div>

        <div className="grid w-full max-w-3xl grid-cols-1 gap-4 sm:grid-cols-2">
          {MODES.map(
            ({ href, icon: Icon, title, subtitle, description, tags, accent, iconClass, ring }) => (
              <motion.div key={href} variants={item} className="h-full">
                <Link
                  href={href}
                  className={`group relative flex h-full flex-col gap-4 overflow-hidden rounded-2xl border bg-card/60 p-6 text-left shadow-sm backdrop-blur-sm transition-colors hover:bg-card ${ring}`}
                >
                  <div
                    className={`pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-br ${accent}`}
                  />
                  <div className="relative flex items-center justify-between">
                    <span
                      className={`flex h-12 w-12 items-center justify-center rounded-xl ${iconClass}`}
                    >
                      <Icon className="h-6 w-6" />
                    </span>
                    <ArrowRight className="h-5 w-5 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-foreground" />
                  </div>
                  <div className="relative space-y-1.5">
                    <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {subtitle}
                    </p>
                    <p className="text-sm text-muted-foreground">{description}</p>
                  </div>
                  <div className="relative mt-auto flex flex-wrap gap-1.5">
                    {tags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded-full border bg-muted/50 px-2 py-0.5 text-[11px] text-muted-foreground"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </Link>
              </motion.div>
            ),
          )}
        </div>
      </motion.main>
    </div>
  );
}
