function LoadingState() {
  return (
    <section className="panel loading-panel" aria-live="polite" aria-busy="true">
      <div className="spinner" />
      <div>
        <h2>Analyzing your ticket</h2>
        <p>Fetching answers, predictions, sources, and comparison metrics.</p>
      </div>
    </section>
  );
}

export default LoadingState;
