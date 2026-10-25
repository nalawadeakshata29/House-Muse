import { useState, useEffect } from "react";
import { Link } from "wouter";
import { Plus, ArrowRight, Layers, Sparkles, Ruler, Clock, FolderOpen, RefreshCw } from "lucide-react";
import { getProjects, type Project } from "../lib/api";

export default function Dashboard() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadProjects = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getProjects();
      setProjects(Array.isArray(data) ? data : []);
    } catch (err: unknown) {
      console.error("Failed to load projects", err);
      // If unauthorized or network error, projects remain empty
      setProjects([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProjects();
  }, []);

  return (
    <div className="product-page">
      <div className="dashboard-hero-header">
        <div>
          <span className="section-eyebrow">Studio Overview</span>
          <h1 className="section-title">Design Workspace</h1>
        </div>
        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <button
            onClick={loadProjects}
            className="button ghost"
            title="Refresh saved projects"
            style={{ padding: "10px 14px" }}
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>
          <Link href="/studio" className="button primary">
            <Plus size={16} />
            <span>New Room Project</span>
          </Link>
        </div>
      </div>

      <div className="arch-bento-grid">
        {/* Left Feature Bento */}
        <div className="arch-bento-feature">
          <img
            src="/images/livingroom.jpg"
            className="feature-img"
            alt="Japandi Living Sanctuary"
          />
          <div className="feature-overlay-top">
            <span className="feature-tag">Curated Concept</span>
          </div>
          <div className="feature-overlay-bottom">
            <div>
              <h3 className="feature-title">Japandi Living Sanctuary</h3>
              <p className="feature-desc">
                Warm minimal organic wood tones, tactile linen textiles, and harmonic architectural balance designed for serene dwelling.
              </p>
            </div>
            <Link href="/studio" className="feature-cta-pill">
              <span>Launch Studio</span>
              <ArrowRight size={14} />
            </Link>
          </div>
        </div>

        {/* Right Stats & Espresso Card Column */}
        <div className="arch-bento-stats-column">
          <div className="arch-stat-box">
            <div>
              <span className="stat-num">{projects.length}</span>
              <p className="stat-copy">
                Curated room transformations designed and saved to your architectural portfolio.
              </p>
            </div>
            <div className="stat-sub">
              <span>Library Projects</span>
              <span>Saved</span>
            </div>
          </div>

          <div className="arch-stat-box">
            <div>
              <span className="stat-num">9</span>
              <p className="stat-copy">
                Architectural design styles available with calibrated color harmony matrices.
              </p>
            </div>
            <div className="stat-sub">
              <span>Aesthetic Engines</span>
              <span>Active</span>
            </div>
          </div>

          <div className="arch-espresso-card">
            <div className="espresso-quote-icon">“</div>
            <p className="espresso-quote-text">
              Space is the breath of art. When proportion meets harmony, every room becomes a muse.
            </p>
            <div className="espresso-card-footer">
              <Link href="/studio" className="espresso-card-link">
                <span>Explore Design Studio</span>
                <ArrowRight size={14} />
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Saved Projects Gallery */}
      <section className="dashboard-gallery-section">
        <div className="gallery-header">
          <div>
            <span className="section-eyebrow">Your Portfolio</span>
            <h2 className="section-title" style={{ fontSize: "1.75rem", margin: "4px 0 0" }}>
              Saved Transformations
            </h2>
          </div>
          <span className="gallery-badge">
            <Layers size={13} />
            <span>{projects.length} {projects.length === 1 ? "project" : "projects"}</span>
          </span>
        </div>

        {loading ? (
          <div
            style={{
              padding: "60px 20px",
              textAlign: "center",
              color: "var(--soft-ink)",
              background: "#fff",
              borderRadius: "14px",
              border: "1px solid var(--line)",
            }}
          >
            <RefreshCw size={24} className="animate-spin" style={{ margin: "0 auto 12px" }} />
            <p style={{ margin: 0 }}>Loading your studio portfolio...</p>
          </div>
        ) : projects.length === 0 ? (
          <div
            style={{
              padding: "64px 24px",
              textAlign: "center",
              background: "#FFFFFF",
              borderRadius: "16px",
              border: "1px dashed var(--line)",
            }}
          >
            <div
              style={{
                width: "56px",
                height: "56px",
                borderRadius: "50%",
                background: "var(--cream)",
                display: "grid",
                placeItems: "center",
                margin: "0 auto 18px",
                color: "var(--espresso)",
              }}
            >
              <FolderOpen size={26} />
            </div>
            <h3
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "1.4rem",
                color: "var(--espresso)",
                margin: "0 0 8px",
              }}
            >
              No saved spaces yet
            </h3>
            <p
              style={{
                color: "var(--soft-ink)",
                maxWidth: "420px",
                margin: "0 auto 24px",
                fontSize: "14px",
                lineHeight: 1.6,
              }}
            >
              Upload an empty or furnished room photo in the Studio to generate realistic photorealistic redesigns and tailored budgeting.
            </p>
            <Link href="/studio" className="button primary">
              <Plus size={15} />
              <span>Create Your First Room</span>
            </Link>
          </div>
        ) : (
          <div className="showcase-grid">
            {projects.map((proj) => {
              const displayImage =
                proj.generated_image_url ||
                proj.source_image ||
                "/images/livingroom.jpg";

              return (
                <div key={proj.id} className="room-card">
                  <div className="room-card-media">
                    <img
                      src={displayImage}
                      alt={proj.name || `${proj.room_type} Project`}
                      loading="lazy"
                    />
                    <span className="room-type-tag">
                      {proj.room_type ? proj.room_type.replace("_", " ") : "Room"}
                    </span>
                  </div>

                  <div className="room-card-body">
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        gap: "10px",
                        marginBottom: "8px",
                      }}
                    >
                      <h4 className="room-card-title">
                        {proj.name || `${proj.room_type.replace("_", " ").toUpperCase()}`}
                      </h4>
                      {proj.styles && proj.styles.length > 0 && (
                        <span className="style-chip">
                          {proj.styles[0].replace("_", " ")}
                        </span>
                      )}
                    </div>

                    <div className="room-specs-row">
                      <span>
                        <Ruler size={11} style={{ display: "inline", marginRight: "4px" }} />
                        {proj.length}′ × {proj.width}′ × {proj.height}′
                      </span>
                      <span>
                        ₹{Number(proj.budget).toLocaleString("en-IN")}
                      </span>
                    </div>

                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        marginTop: "auto",
                        paddingTop: "12px",
                      }}
                    >
                      <span
                        style={{
                          fontSize: "11px",
                          color: "#9A8F83",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "4px",
                        }}
                      >
                        <Clock size={11} />
                        {new Date(proj.created_at).toLocaleDateString()}
                      </span>
                      <Link
                        href="/studio"
                        className="button ghost"
                        style={{ padding: "6px 14px", fontSize: "11px" }}
                      >
                        Open in Studio
                      </Link>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
