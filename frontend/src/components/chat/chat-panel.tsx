"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Stethoscope } from "lucide-react";
import { sendChatMessage } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";
import { MessageBubble } from "@/components/chat/message-bubble";
import { EmptyState } from "@/components/chat/empty-state";
import { ChatInput } from "@/components/chat/chat-input";
import { DisclaimerFooter } from "@/components/chat/disclaimer-footer";
import { ThemeToggle } from "@/components/theme-toggle";
import { ThinkingIndicator } from "@/components/chat/thinking-indicator";
import { AnimatedBackground } from "@/components/chat/animated-background";

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
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

    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: "user", content };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await sendChatMessage(content, sessionId);
      setSessionId(response.session_id);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          triage: response.triage,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong talking to MediSense.");
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
        <motion.div
          whileHover={{ rotate: [0, -8, 8, 0], transition: { duration: 0.4 } }}
          className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10"
        >
          <Stethoscope className="h-4 w-4 text-primary" />
        </motion.div>
        <div className="flex-1">
          <h1 className="text-sm font-semibold leading-none">MediSense AI</h1>
          <p className="text-[11px] text-muted-foreground mt-0.5">Health information demo, not a real clinician</p>
        </div>
        <ThemeToggle />
      </motion.header>

      {messages.length === 0 ? (
        <EmptyState onSelectPrompt={(prompt) => handleSubmit(prompt)} />
      ) : (
        <div className="flex-1 space-y-5 overflow-y-auto px-4 py-5">
          <AnimatePresence initial={false}>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {loading && <ThinkingIndicator key="thinking" />}
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
      <DisclaimerFooter />
    </div>
  );
}
