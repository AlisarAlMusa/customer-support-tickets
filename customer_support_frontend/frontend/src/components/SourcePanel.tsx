import type { SourceTicket } from "../types/api";

interface SourcePanelProps {
  sources: SourceTicket[];
}

function SourcePanel({ sources }: SourcePanelProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <span className="section-label">Retrieved Sources</span>
          <h2>RAG evidence panel</h2>
        </div>
      </div>

      {sources.length === 0 ? (
        <div className="empty-state compact-state">
          <p>No retrieved tickets were returned for this query.</p>
        </div>
      ) : (
        <div className="source-list">
          {sources.map((source, index) => (
            <article className="source-card" key={source.id || `${source.company}-${index}`}>
              <div className="source-meta">
                <span>Source #{index + 1}</span>
                <span>{source.company || source.source || "Unknown source"}</span>
                <span>
                  Similarity:{" "}
                  {typeof source.similarityScore === "number"
                    ? source.similarityScore.toFixed(3)
                    : "N/A"}
                </span>
                <span>{source.createdAt || "Date unavailable"}</span>
              </div>

              <div className="source-copy">
                <div>
                  <h3>Customer text</h3>
                  <p>{source.customerText || "No customer text available."}</p>
                </div>
                <div>
                  <h3>Company response</h3>
                  <p>{source.companyResponse || "No company response available."}</p>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

export default SourcePanel;
