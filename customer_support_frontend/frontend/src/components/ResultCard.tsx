interface ResultCardProps {
  title: string;
  subtitle: string;
  value: string;
  analysis?: string;
  confidence?: string;
  latency: string;
  cost: string;
}

function ResultCard({
  title,
  subtitle,
  value,
  analysis,
  confidence,
  latency,
  cost,
}: ResultCardProps) {
  return (
    <article className="panel result-card">
      <div className="panel-header">
        <div>
          <span className="section-label">{title}</span>
          <h2>{subtitle}</h2>
        </div>
        <div className="stat-grid compact">
          <div className="stat-chip">
            <span>Confidence</span>
            <strong>{confidence}</strong>
          </div>
          <div className="stat-chip">
            <span>Latency</span>
            <strong>{latency}</strong>
          </div>
          <div className="stat-chip">
            <span>Cost</span>
            <strong>{cost}</strong>
          </div>
        </div>
      </div>

      <div className="content-block">
        <h3>Output</h3>
        <p>{value}</p>
      </div>

      <div className="content-block">
        <h3>Analysis</h3>
        <p>{analysis || "No analysis was returned by the backend."}</p>
      </div>
    </article>
  );
}

export default ResultCard;
