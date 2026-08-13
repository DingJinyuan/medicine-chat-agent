"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { SourceChunk } from "@/lib/types";

export function SourceCitations({ sources }: { sources: SourceChunk[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!sources.length) return null;

  const uniqueByTopic = Array.from(new Map(sources.map((s) => [s.topic, s])).values());

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.2, duration: 0.3 }}
      className="mt-2 flex flex-wrap gap-1.5"
    >
      {uniqueByTopic.map((source, i) => {
        const isOpen = expanded === source.topic;
        return (
          <motion.div
            key={source.topic}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.25 + i * 0.05, duration: 0.25 }}
            className="w-full sm:w-auto"
          >
            <button
              onClick={() => setExpanded(isOpen ? null : source.topic)}
              className="inline-flex items-center gap-1"
            >
              <Badge
                variant="secondary"
                className="cursor-pointer gap-1 font-normal hover:bg-accent hover:text-accent-foreground transition-colors"
              >
                {source.topic}
                <motion.span animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.2 }}>
                  <ChevronDown className="h-3 w-3" />
                </motion.span>
              </Badge>
            </button>
            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  className="overflow-hidden"
                >
                  <div className="mt-1.5 max-w-md rounded-md border bg-muted/50 p-2.5 text-xs text-muted-foreground">
                    <p className="line-clamp-4">{source.text}</p>
                    {source.url && (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-1.5 inline-flex items-center gap-1 text-primary hover:underline"
                      >
                        MedlinePlus source <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        );
      })}
    </motion.div>
  );
}
