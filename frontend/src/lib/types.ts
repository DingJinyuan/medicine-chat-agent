export type Mode = "patient" | "expert";

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

// 患者版（MediSense）API 响应
export interface PatientChatResponse {
  session_id: string;
  mode: string;
  answer: string;
  sources: SourceChunk[];
  triage: TriagePrediction | null;
  // 后端契约字段：护栏是否改写 / 是否检测到注入，UI 暂不展示，保留供后续使用
  guardrail_rewritten: boolean;
  injection_flagged: boolean;
}

// 专家版 API 响应：多 Agent 临床 RAG + 事实核查
export interface ExpertChatResponse {
  session_id: string;
  mode: string;
  answer: string;
  citations: string[];
  confidence_score: number | null;
  is_reliable: boolean | null;
  critique: string;
}

// 患者版消息（分诊 + 溯源）
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
  triage?: TriagePrediction | null;
}

// 专家版消息（可靠性评分 + 引用 + 批判评语）
export interface ExpertMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  confidenceScore?: number | null;
  isReliable?: boolean | null;
  critique?: string;
}
