import { Link } from "wouter";
import { Compass, ArrowRight } from "lucide-react";

export default function NotFound() {
  return (
    <div
      className="product-page"
      style={{
        minHeight: "70vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
      }}
    >
      <div
        style={{
          width: "64px",
          height: "64px",
          borderRadius: "50%",
          background: "var(--espresso)",
          color: "var(--cream)",
          display: "grid",
          placeItems: "center",
          marginBottom: "24px",
        }}
      >
        <Compass size={32} />
      </div>

      <span className="section-eyebrow">404 // Spatial Anomaly</span>
      <h1 className="section-title" style={{ fontSize: "2.4rem", margin: "8px 0 16px" }}>
        Room Not Found
      </h1>
      <p
        style={{
          maxWidth: "460px",
          color: "var(--soft-ink)",
          lineHeight: 1.6,
          marginBottom: "28px",
        }}
      >
        The architectural coordinate you requested does not exist or has been rearranged. Return to the studio to resume designing.
      </p>

      <div style={{ display: "flex", gap: "12px" }}>
        <Link href="/" className="button ghost">
          Return Home
        </Link>
        <Link href="/studio" className="button primary">
          <span>Enter Studio</span>
          <ArrowRight size={14} />
        </Link>
      </div>
    </div>
  );
}
