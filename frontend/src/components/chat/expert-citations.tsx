"use client";

import { motion } from "framer-motion";
import { FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export function ExpertCitations({ citations }: { citations: string[] }) {
  if (!citations?.length) return null;

  const unique = Array.from(new Set(citations));

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.2, duration: 0.3 }}
      className="mt-2 flex flex-wrap gap-1.5"
    >
      {unique.map((cite, i) => (
        <motion.div
          key={cite}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.25 + i * 0.05, duration: 0.25 }}
        >
          <Badge variant="secondary" className="gap-1 font-normal">
            <FileText className="h-3 w-3" />
            <span className="text-muted-foreground">[{i + 1}]</span>
            {cite}
          </Badge>
        </motion.div>
      ))}
    </motion.div>
  );
}
