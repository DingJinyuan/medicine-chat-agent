"use client";

import { useEffect, useRef, useState, type ComponentType, type ReactNode } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, Stethoscope } from "lucide-react";
import { ChatInput } from "@/components/chat/chat-input";
import { DisclaimerFooter } from "@/components/chat/disclaimer-footer";
import { ThemeToggle } from "@/components/theme-toggle";
import { ThinkingIndicator } from "@/components/chat/thinking-indicator";
import { AnimatedBackground } from "@/components/chat/animated-background";

interface BaseMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface BaseChatPanelProps<T extends BaseMessage> {
  title: string;
  subtitle: string;
  icon: ComponentType<{ className?: string }>;
  iconWrapClass: string;
  iconClass: string;
  thinkingIcon?: ComponentType<{ className?: string }>;
  disclaimerMode?: "patient" | "expert";
  renderEmptyState: (onSelectPrompt: (prompt: string) => void) => ReactNode;
  renderBubble: (message: T) => ReactNode;
  sendMessage: (
    content: string,
    sessionId?: string,
  ) => Promise<{ session_id: string; assistant: T }>;
}

export function BaseChatPanel<T extends BaseMessage>({
  title,
  subtitle,
  icon: Icon,
  iconWrapClass,
  iconClass,
  thinkingIcon = Stethoscope,
  disclaimerMode = "patient",
  renderEmptyState,
  renderBubble,
  sendMessage,
}: BaseChatPanelProps<T>) {
  const [messages, setMessages] = useState<T[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(text?: string) {
    const content = (text ?? input).trim();
    if (!content || loading) return;

    const userMessage = { id: crypto.randomUUID(), role: "user" as const, content } as T;
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const { session_id, assistant } = await sendMessage(content, sessionId);
      setSessionId(session_id);
      setMessages((prev) => [...prev, assistant]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative flex h-dvh flex-col">
      <AnimatedBackground />

      <motion.header
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="flex items-center gap-2.5 border-b bg-background/70 px-4 py-3 backdrop-blur-md"
      >
        <Link
          href="/"
          aria-label="Back to mode selection"
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <motion.div
          whileHover={{ rotate: [0, -8, 8, 0], transition: { duration: 0.4 } }}
          className={`flex h-8 w-8 items-center justify-center rounded-lg ${iconWrapClass}`}
        >
          <Icon className={`h-4 w-4 ${iconClass}`} />
        </motion.div>
        <div className="flex-1">
          <h1 className="text-sm font-semibold leading-none">{title}</h1>
          <p className="text-[11px] text-muted-foreground mt-0.5">{subtitle}</p>
        </div>
        <ThemeToggle />
      </motion.header>

      {messages.length === 0 ? (
        renderEmptyState((prompt) => handleSubmit(prompt))
      ) : (
        <div className="flex-1 space-y-5 overflow-y-auto px-4 py-5">
          <AnimatePresence initial={false}>
            {messages.map((message) => renderBubble(message))}
            {loading && <ThinkingIndicator key="thinking" icon={thinkingIcon} />}
            {error && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="ml-11 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={scrollAnchorRef} />
        </div>
      )}

      <ChatInput value={input} onChange={setInput} onSubmit={() => handleSubmit()} disabled={loading} />
      <DisclaimerFooter mode={disclaimerMode} />
    </div>
  );
}
