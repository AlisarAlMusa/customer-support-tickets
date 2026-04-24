import { useMemo, useState } from "react";
import { analyzeTicket } from "../api/client";
import ComparisonTable from "../components/ComparisonTable";
import ErrorBanner from "../components/ErrorBanner";
import LoadingState from "../components/LoadingState";
import QueryBox from "../components/QueryBox";
import ResultCard from "../components/ResultCard";
import SourcePanel from "../components/SourcePanel";
import type { AnalyzeTicketResult, ComparisonRow } from "../types/api";

const EXAMPLE_QUERY =
  "My internet connection is very slow and keeps disconnecting.";

function formatConfidence(value: number | string | undefined) {
  if (typeof value === "number") {
    return `${Math.round(value)}%`;
  }

  return value ? String(value) : "N/A";
}

function formatCost(value: number | string | undefined) {
  if (typeof value === "number") {
    return `$${value.toFixed(4)}`;
  }

  return value ? String(value) : "N/A";
}

function Dashboard() {
  const [query, setQuery] = useState(EXAMPLE_QUERY);
  const [result, setResult] = useState<AnalyzeTicketResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const comparisonRows = useMemo<ComparisonRow[]>(() => {
    if (!result) {
      return [];
    }

    return [
      {
        id: "rag",
        label: "RAG",
        output: result.answer.rag.answer,
        confidence: result.answer.rag.confidence,
        latencyMs: result.answer.rag.latencyMs,
        cost: result.answer.rag.cost,
      },
      {
        id: "non-rag",
        label: "Non-RAG",
        output: result.answer.nonRag.answer,
        confidence: result.answer.nonRag.confidence,
        latencyMs: result.answer.nonRag.latencyMs,
        cost: result.answer.nonRag.cost,
      },
      {
        id: "ml",
        label: "ML",
        output: result.predict.ml.label,
        confidence: result.predict.ml.confidence,
        latencyMs: result.predict.ml.latencyMs,
        cost: result.predict.ml.cost,
      },
      {
        id: "llm",
        label: "LLM Zero-Shot",
        output: result.predict.llm.label,
        confidence: result.predict.llm.confidence,
        latencyMs: result.predict.llm.latencyMs,
        cost: result.predict.llm.cost,
      },
    ];
  }, [result]);

  const handleAnalyze = async () => {
    const trimmedQuery = query.trim();

    if (!trimmedQuery) {
      setError("Please enter a support ticket before submitting.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const data = await analyzeTicket(trimmedQuery);
      setResult(data);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : "An unexpected error occurred while calling the API.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="dashboard-shell">
      <div className="background-orb background-orb-left" />
      <div className="background-orb background-orb-right" />

      <div className="dashboard-content">
        <QueryBox
          value={query}
          onChange={setQuery}
          onSubmit={handleAnalyze}
          isLoading={isLoading}
        />

        {error ? <ErrorBanner message={error} /> : null}

        {isLoading ? <LoadingState /> : null}

        {!isLoading && !result ? (
          <section className="panel empty-state">
            <span className="section-label">Ready</span>
            <h2>Run your first ticket analysis</h2>
            <p>
              Results from retrieval answering and priority prediction will appear
              here once you submit a query.
            </p>
          </section>
        ) : null}

        {result ? (
          <>
            <section className="results-grid">
              <ResultCard
                title="Answering"
                subtitle="RAG response"
                value={result.answer.rag.answer}
                analysis={result.answer.rag.analysis}
                confidence={formatConfidence(result.answer.rag.confidence)}
                latency={`${result.answer.rag.latencyMs.toFixed(0)} ms`}
                cost={formatCost(result.answer.rag.cost)}
              />
              <ResultCard
                title="Answering"
                subtitle="Non-RAG response"
                value={result.answer.nonRag.answer}
                analysis={result.answer.nonRag.analysis}
                confidence={formatConfidence(result.answer.nonRag.confidence)}
                latency={`${result.answer.nonRag.latencyMs.toFixed(0)} ms`}
                cost={formatCost(result.answer.nonRag.cost)}
              />
              <ResultCard
                title="Prediction"
                subtitle="ML priority prediction"
                value={result.predict.ml.label}
                analysis={result.predict.ml.analysis}
                confidence={formatConfidence(result.predict.ml.confidence)}
                latency={`${result.predict.ml.latencyMs.toFixed(0)} ms`}
                cost={formatCost(result.predict.ml.cost)}
              />
              <ResultCard
                title="Prediction"
                subtitle="LLM zero-shot prediction"
                value={result.predict.llm.label}
                analysis={result.predict.llm.analysis}
                confidence={formatConfidence(result.predict.llm.confidence)}
                latency={`${result.predict.llm.latencyMs.toFixed(0)} ms`}
                cost={formatCost(result.predict.llm.cost)}
              />
            </section>

            <ComparisonTable rows={comparisonRows} />
            <SourcePanel sources={result.answer.sources} />
          </>
        ) : null}
      </div>
    </main>
  );
}

export default Dashboard;
