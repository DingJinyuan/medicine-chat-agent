export type TriageLabel = "emergency" | "urgent" | "routine" | "self_care";

export interface SourceChunk {
  topic: string;
  url: string;
  text: string;
  score: number;
}

export interface TriagePrediction {
  label: TriageLabel;
  confidence: number;
}

export interface ChatApiResponse {
  session_id: string;
  answer: string;
  sources: SourceChunk[];
  triage: TriagePrediction;
  guardrail_rewritten: boolean;
  injection_flagged: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
  triage?: TriagePrediction;
}
