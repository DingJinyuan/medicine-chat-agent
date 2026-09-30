import type { PatientChatResponse, ExpertChatResponse, Mode } from "@/lib/types";

interface ChatBody {
  message: string;
  session_id: string | null;
  mode: Mode;
}

async function postChat<T>(body: ChatBody): Promise<T> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    // 后端格式 {"error":{"code","message"}}；代理层 {"error":"..."}；旧格式 {"detail":"..."}
    const detail =
      typeof data?.error?.message === "string"
        ? data.error.message
        : typeof data?.error === "string"
          ? data.error
          : typeof data?.detail === "string"
            ? data.detail
            : "Something went wrong.";
    throw new Error(detail);
  }

  return data as T;
}

export function sendPatientMessage(
  message: string,
  sessionId?: string,
): Promise<PatientChatResponse> {
  return postChat<PatientChatResponse>({
    message,
    session_id: sessionId ?? null,
    mode: "patient",
  });
}

export function sendExpertMessage(
  message: string,
  sessionId?: string,
): Promise<ExpertChatResponse> {
  return postChat<ExpertChatResponse>({
    message,
    session_id: sessionId ?? null,
    mode: "expert",
  });
}
