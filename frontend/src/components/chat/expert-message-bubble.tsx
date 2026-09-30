import { motion } from "framer-motion";
import { Microscope, User } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ConfidenceBanner } from "@/components/chat/confidence-banner";
import { ExpertCitations } from "@/components/chat/expert-citations";
import { CritiqueBlock } from "@/components/chat/critique-block";
import { MarkdownAnswer } from "@/components/chat/markdown-answer";
import type { ExpertMessage } from "@/lib/types";

export function ExpertMessageBubble({ message }: { message: ExpertMessage }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}
    >
      <Avatar className="h-8 w-8 shrink-0 mt-0.5">
        <AvatarFallback className={isUser ? "bg-secondary" : "bg-primary text-primary-foreground"}>
          {isUser ? <User className="h-4 w-4" /> : <Microscope className="h-4 w-4" />}
        </AvatarFallback>
      </Avatar>

      <div className={`flex max-w-[85%] flex-col gap-2 sm:max-w-[75%] ${isUser ? "items-end" : "items-start"}`}>
        {!isUser && (message.confidenceScore != null || message.isReliable != null) && (
          <ConfidenceBanner
            confidenceScore={message.confidenceScore ?? null}
            isReliable={message.isReliable ?? null}
          />
        )}

        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08, duration: 0.3 }}
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? "bg-primary text-primary-foreground rounded-tr-sm whitespace-pre-wrap"
              : "bg-card border rounded-tl-sm"
          }`}
        >
          {isUser ? message.content : <MarkdownAnswer content={message.content} />}
        </motion.div>

        {!isUser && <ExpertCitations citations={message.citations ?? []} />}
        {!isUser && <CritiqueBlock critique={message.critique ?? ""} />}
      </div>
    </motion.div>
  );
}
