import type { ComparisonRow } from "../types/api";

interface ComparisonTableProps {
  rows: ComparisonRow[];
}

function ComparisonTable({ rows }: ComparisonTableProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <span className="section-label">Comparison</span>
          <h2>Model and answer summary</h2>
        </div>
      </div>

      <div className="table-wrap">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>System</th>
              <th>Output</th>
              <th>Confidence / Accuracy</th>
              <th>Latency</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>{row.label}</td>
                <td>{row.output}</td>
                <td>{String(row.confidence ?? "N/A")}</td>
                <td>{row.latencyMs.toFixed(0)} ms</td>
                <td>{String(row.cost ?? "N/A")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default ComparisonTable;
