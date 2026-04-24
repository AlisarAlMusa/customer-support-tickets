interface QueryBoxProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

function QueryBox({ value, onChange, onSubmit, isLoading }: QueryBoxProps) {
  return (
    <section className="panel panel-hero">
      <div className="hero-copy">
        <span className="eyebrow">AI Customer Support Tickets</span>
        <h1>Analyze support tickets with retrieval, prediction, and side-by-side insight.</h1>
        <p>
          Submit a customer issue once and compare RAG, non-RAG, ML, and LLM
          zero-shot outputs in one clean dashboard.
        </p>
      </div>

      <div className="query-box">
        <label className="section-label" htmlFor="ticket-query">
          Support ticket or customer message
        </label>
        <textarea
          id="ticket-query"
          className="query-input"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="My internet connection is very slow and keeps disconnecting."
          rows={6}
          disabled={isLoading}
        />

        <div className="query-actions">
          <p className="helper-text">
            This will call <code>/answer</code> and <code>/predict</code> using the same
            ticket text.
          </p>
          <button
            type="button"
            className="primary-button"
            onClick={onSubmit}
            disabled={isLoading || !value.trim()}
          >
            {isLoading ? "Analyzing..." : "Analyze Ticket"}
          </button>
        </div>
      </div>
    </section>
  );
}

export default QueryBox;
