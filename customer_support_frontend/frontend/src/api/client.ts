import axios, { AxiosError } from "axios";
import type {
  AnalyzeTicketResult,
  AnswerResponseNormalized,
  PredictionResult,
  PredictResponseNormalized,
  SourceTicket,
  TicketQueryPayload,
  UnknownRecord,
} from "../types/api";

const DEFAULT_BASE_URL = "http://localhost:8000";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || DEFAULT_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

function asRecord(value: unknown): UnknownRecord | undefined {
  return typeof value === "object" && value !== null ? (value as UnknownRecord) : undefined;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function pickValue(record: UnknownRecord | undefined, keys: string[]): unknown {
  if (!record) {
    return undefined;
  }

  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) {
      return record[key];
    }
  }

  return undefined;
}

function pickString(record: UnknownRecord | undefined, keys: string[], fallback = "N/A") {
  const value = pickValue(record, keys);

  if (typeof value === "string" && value.trim()) {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return fallback;
}

function pickOptionalString(record: UnknownRecord | undefined, keys: string[]) {
  const value = pickValue(record, keys);

  if (typeof value === "string" && value.trim()) {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return undefined;
}

function pickOptionalNumber(record: UnknownRecord | undefined, keys: string[]) {
  const value = pickValue(record, keys);

  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }

  return undefined;
}

function normalizeCost(record: UnknownRecord | undefined, keys: string[]) {
  const value = pickValue(record, keys);

  if (typeof value === "number" || typeof value === "string") {
    return value;
  }

  return "N/A";
}

function buildTicketPayload(query: string): TicketQueryPayload {
  return { query };
  // If your FastAPI backend expects a different request field such as `text`
  // or `ticket`, update the payload shape here in one place.
}

function normalizeSourceTicket(item: unknown, index: number): SourceTicket {
  const record = asRecord(item);

  return {
    id: pickOptionalString(record, ["id", "_id"]) || `source-${index}`,
    customerText: pickOptionalString(record, [
      "customer_text",
      "customerText",
      "ticket_text",
      "query",
      "text",
      "content",
    ]),
    companyResponse: pickOptionalString(record, [
      "company_response",
      "companyResponse",
      "response",
      "answer",
      "resolution",
    ]),
    company: pickOptionalString(record, ["company", "brand", "organization"]),
    source: pickOptionalString(record, ["source", "dataset", "origin"]),
    similarityScore: pickOptionalNumber(record, [
      "similarity_score",
      "similarityScore",
      "score",
      "distance",
    ]),
    createdAt: pickOptionalString(record, ["created_at", "createdAt", "date"]),
    raw: record,
  };
}

function normalizeAnswerResponse(raw: unknown, latencyMs: number): AnswerResponseNormalized {
  const record = asRecord(raw);
  const ragNode =
    asRecord(pickValue(record, ["rag", "rag_result", "ragAnswer"])) || record;
  const nonRagNode =
    asRecord(pickValue(record, ["non_rag", "nonRag", "base", "non_rag_result"])) || record;
  const sourcesNode =
    asArray(
      pickValue(record, [
        "sources",
        "retrieved_sources",
        "retrieved_tickets",
        "tickets",
        "matches",
      ]),
    ) || [];

  return {
    rag: {
      answer: pickString(ragNode, ["answer", "rag_answer", "response"], "No RAG answer returned."),
      analysis: pickOptionalString(ragNode, ["analysis", "reasoning", "explanation"]),
      confidence:
        pickOptionalNumber(ragNode, ["confidence", "accuracy", "score"]) ??
        pickOptionalString(ragNode, ["confidence", "accuracy"]),
      cost: normalizeCost(ragNode, ["cost", "token_cost", "price"]),
      latencyMs,
      raw: ragNode,
    },
    nonRag: {
      answer: pickString(
        nonRagNode,
        ["answer", "non_rag_answer", "nonrag_answer", "response", "base_answer"],
        "No non-RAG answer returned.",
      ),
      analysis: pickOptionalString(nonRagNode, ["analysis", "reasoning", "explanation"]),
      confidence:
        pickOptionalNumber(nonRagNode, ["confidence", "accuracy", "score"]) ??
        pickOptionalString(nonRagNode, ["confidence", "accuracy"]),
      cost: normalizeCost(nonRagNode, ["cost", "token_cost", "price"]),
      latencyMs,
      raw: nonRagNode,
    },
    sources: sourcesNode.map(normalizeSourceTicket),
    raw: record,
  };
}

function normalizePredictionNode(
  node: UnknownRecord | undefined,
  labelKeys: string[],
  analysisKeys: string[],
  latencyMs: number,
  fallbackLabel: string,
): PredictionResult {
  return {
    label: pickString(node, labelKeys, fallbackLabel),
    analysis: pickOptionalString(node, analysisKeys),
    confidence:
      pickOptionalNumber(node, ["confidence", "accuracy", "probability", "score"]) ??
      pickOptionalString(node, ["confidence", "accuracy"]),
    cost: normalizeCost(node, ["cost", "token_cost", "price"]),
    latencyMs,
    raw: node,
  };
}

function normalizePredictResponse(raw: unknown, latencyMs: number): PredictResponseNormalized {
  const record = asRecord(raw);
  const mlNode = asRecord(pickValue(record, ["ml", "ml_prediction", "machine_learning"])) || record;
  const llmNode =
    asRecord(pickValue(record, ["llm", "llm_prediction", "zero_shot", "zeroShot"])) || record;

  return {
    ml: normalizePredictionNode(
      mlNode,
      ["prediction", "label", "priority", "class", "ml_prediction"],
      ["analysis", "reasoning", "explanation", "ml_analysis"],
      latencyMs,
      "No ML prediction returned.",
    ),
    llm: normalizePredictionNode(
      llmNode,
      ["prediction", "label", "priority", "class", "llm_prediction"],
      ["analysis", "reasoning", "explanation", "llm_analysis"],
      latencyMs,
      "No LLM prediction returned.",
    ),
    raw: record,
  };
}

async function timedPost<T>(path: string, payload: TicketQueryPayload, normalizer: (raw: unknown, latencyMs: number) => T) {
  const start = performance.now();
  const response = await api.post(path, payload);
  const latencyMs = performance.now() - start;
  return normalizer(response.data, latencyMs);
}

export async function analyzeTicket(query: string): Promise<AnalyzeTicketResult> {
  try {
    const payload = buildTicketPayload(query);

    const [answer, predict] = await Promise.all([
      timedPost("/answer", payload, normalizeAnswerResponse),
      timedPost("/predict", payload, normalizePredictResponse),
    ]);

    return { answer, predict };
  } catch (error) {
    if (error instanceof AxiosError) {
      const apiMessage =
        asRecord(error.response?.data)?.detail ??
        asRecord(error.response?.data)?.message ??
        error.message;

      throw new Error(String(apiMessage), { cause: error });
    }

    throw error instanceof Error
      ? error
      : new Error("Unknown API error", { cause: error });
  }
}
