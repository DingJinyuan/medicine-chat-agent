"use client";

import { Microscope } from "lucide-react";
import { sendExpertMessage } from "@/lib/api";
import type { ExpertMessage } from "@/lib/types";
import { BaseChatPanel } from "@/components/chat/base-chat-panel";
import { ExpertMessageBubble } from "@/components/chat/expert-message-bubble";
import { ExpertEmptyState } from "@/components/chat/expert-empty-state";

export function ExpertChatPanel() {
  return (
    <BaseChatPanel<ExpertMessage>
      title="Clinical AI"
      subtitle="Evidence-based decision support for physicians"
      icon={Microscope}
      iconWrapClass="bg-teal-500/10"
      iconClass="text-teal-600 dark:text-teal-400"
      thinkingIcon={Microscope}
      disclaimerMode="expert"
      renderEmptyState={(onSelectPrompt) => <ExpertEmptyState onSelectPrompt={onSelectPrompt} />}
      renderBubble={(message) => <ExpertMessageBubble key={message.id} message={message} />}
      sendMessage={async (content, sessionId) => {
        const response = await sendExpertMessage(content, sessionId);
        return {
          session_id: response.session_id,
          assistant: {
            id: crypto.randomUUID(),
            role: "assistant",
            content: response.answer,
            citations: response.citations,
            confidenceScore: response.confidence_score,
            isReliable: response.is_reliable,
            critique: response.critique,
          },
        };
      }}
    />
  );
}
