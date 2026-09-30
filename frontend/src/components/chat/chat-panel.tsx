"use client";

import { Stethoscope } from "lucide-react";
import { sendPatientMessage } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";
import { BaseChatPanel } from "@/components/chat/base-chat-panel";
import { MessageBubble } from "@/components/chat/message-bubble";
import { EmptyState } from "@/components/chat/empty-state";

export function ChatPanel() {
  return (
    <BaseChatPanel<ChatMessage>
      title="MediSense AI"
      subtitle="Health information demo, not a real clinician"
      icon={Stethoscope}
      iconWrapClass="bg-primary/10"
      iconClass="text-primary"
      disclaimerMode="patient"
      renderEmptyState={(onSelectPrompt) => <EmptyState onSelectPrompt={onSelectPrompt} />}
      renderBubble={(message) => <MessageBubble key={message.id} message={message} />}
      sendMessage={async (content, sessionId) => {
        const response = await sendPatientMessage(content, sessionId);
        return {
          session_id: response.session_id,
          assistant: {
            id: crypto.randomUUID(),
            role: "assistant",
            content: response.answer,
            sources: response.sources,
            triage: response.triage,
          },
        };
      }}
    />
  );
}
