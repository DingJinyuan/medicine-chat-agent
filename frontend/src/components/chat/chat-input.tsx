"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { motion } from "framer-motion";
import { ArrowUp } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

export function ChatInput({ value, onChange, onSubmit, disabled }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [focused, setFocused] = useState(false);
  const canSend = !disabled && value.trim().length > 0;

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    onChange(e.target.value);
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onSubmit();
    }
  }

  return (
    <div className="border-t bg-background p-3">
      <motion.div
        animate={{
          boxShadow: focused
            ? "0 0 0 3px var(--color-ring), 0 4px 20px -4px color-mix(in oklch, var(--color-primary) 25%, transparent)"
            : "0 0 0 0px transparent",
        }}
        transition={{ duration: 0.2 }}
        className="flex items-end gap-2 rounded-xl"
      >
        <Textarea
          ref={textareaRef}
          value={value}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Describe your symptom or ask a health question..."
          rows={1}
          disabled={disabled}
          className="max-h-40 min-h-10 flex-1 resize-none rounded-xl focus-visible:ring-0"
        />
        <motion.button
          type="button"
          disabled={!canSend}
          onClick={onSubmit}
          whileHover={canSend ? { scale: 1.06 } : undefined}
          whileTap={canSend ? { scale: 0.92 } : undefined}
          animate={canSend ? { opacity: 1 } : { opacity: 0.5 }}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground disabled:cursor-not-allowed"
        >
          <ArrowUp className="h-4 w-4" />
        </motion.button>
      </motion.div>
    </div>
  );
}
