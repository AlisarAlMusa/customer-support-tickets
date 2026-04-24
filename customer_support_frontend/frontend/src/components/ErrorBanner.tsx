interface ErrorBannerProps {
  message: string;
}

function ErrorBanner({ message }: ErrorBannerProps) {
  return (
    <section className="error-banner" role="alert">
      <strong>Request failed.</strong>
      <span>{message}</span>
    </section>
  );
}

export default ErrorBanner;
