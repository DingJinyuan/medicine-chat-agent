import type { ChatApiResponse } from "@/lib/types";

export async function sendChatMessage(message: string, sessionId?: string): Promise<ChatApiResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId ?? null }),
  });

  const data = await res.json();

  if (!res.ok) {
    // 后端错误格式：{"error":{"code","message"}}；兼容旧格式 {"detail":"..."}
    const detail =
      typeof data?.error?.message === "string"
        ? data.error.message
        : typeof data?.detail === "string"
          ? data.detail
          : "Something went wrong.";
    throw new Error(detail);
  }

  return data as ChatApiResponse;
}
