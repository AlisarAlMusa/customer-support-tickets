export type UnknownRecord = Record<string, unknown>;

export interface TicketQueryPayload {
  query: string;
}

export interface SourceTicket {
  id?: string;
  customerText?: string;
  companyResponse?: string;
  company?: string;
  source?: string;
  similarityScore?: number;
  createdAt?: string;
  raw?: UnknownRecord;
}

export interface AnswerResult {
  answer: string;
  analysis?: string;
  confidence?: number | string;
  cost?: string | number;
  latencyMs: number;
  raw?: UnknownRecord;
}

export interface PredictionResult {
  label: string;
  analysis?: string;
  confidence?: number | string;
  cost?: string | number;
  latencyMs: number;
  raw?: UnknownRecord;
}

export interface AnswerResponseNormalized {
  rag: AnswerResult;
  nonRag: AnswerResult;
  sources: SourceTicket[];
  raw?: UnknownRecord;
}

export interface PredictResponseNormalized {
  ml: PredictionResult;
  llm: PredictionResult;
  raw?: UnknownRecord;
}

export interface AnalyzeTicketResult {
  answer: AnswerResponseNormalized;
  predict: PredictResponseNormalized;
}

export interface ComparisonRow {
  id: string;
  label: string;
  output: string;
  confidence: number | string | undefined;
  latencyMs: number;
  cost?: string | number;
}
