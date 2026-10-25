import { useState, useEffect } from "react";
import { useLocation, useRoute, Link } from "wouter";
import { login, register, logout, apiError, type User } from "../lib/api";
import { User as UserIcon, Lock, Mail, ArrowRight, Eye, EyeOff, CheckCircle2, AlertCircle, LogOut } from "lucide-react";

export default function Auth() {
  const [, setLocation] = useLocation();
  const [, resetParams] = useRoute("/reset-password/:uid/:token");

  // Determine mode from query or default
  const queryParams = new URLSearchParams(window.location.search);
  const initialMode = queryParams.get("mode") === "register" ? "register" : "login";
  const [mode, setMode] = useState<"login" | "register">(initialMode);

  // Form fields
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  // State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("house_muse_user");
      if (stored) setCurrentUser(JSON.parse(stored));
    } catch {
      setCurrentUser(null);
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === "login") {
        await login(username, password);
      } else {
        await register({
          username,
          email,
          password,
          first_name: firstName || undefined,
          last_name: lastName || undefined,
        });
      }
      setLocation("/dashboard");
    } catch (err: unknown) {
      setError(apiError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch {
      // ignore
    }
    setCurrentUser(null);
    setLocation("/auth?mode=login");
  };

  return (
    <div className="editorial-auth">
      {/* Left Editorial Art Column */}
      <div className="auth-art-column">
        <img
          src="/images/livingroom.jpg"
          className="auth-art-bg"
          alt="House Muse Sanctuary"
        />
        <div className="auth-art-vignette" />

        <div className="auth-art-header">
          <Link href="/" className="brand-lockup" style={{ color: "#FAF7F2" }}>
            <span
              className="brand-mark"
              style={{ background: "#FAF7F2", color: "var(--espresso)" }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path
                  d="M4 19V5H8.5L12 11.5L15.5 5H20V19H16V10.5L13.2 15.5H10.8L8 10.5V19H4Z"
                  fill="currentColor"
                />
              </svg>
            </span>
            <span className="wordmark" style={{ color: "#FAF7F2" }}>
              HOUSE MUSE
            </span>
          </Link>
        </div>

        <div className="auth-art-copy">
          <span className="scene-kicker">Architectural Intelligence</span>
          <h2
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "clamp(2rem, 3.2vw, 3rem)",
              color: "#FAF7F2",
              margin: 0,
              lineHeight: 1.1,
            }}
          >
            Design with algorithmic clarity & human poise.
          </h2>
          <p
            style={{
              color: "rgba(245, 242, 235, 0.8)",
              fontSize: "14px",
              lineHeight: 1.6,
              margin: 0,
              maxWidth: "460px",
            }}
          >
            Explore AI-assisted dimensional space layouts, bespoke harmonic palettes, and photorealistic architectural generation for discerning spaces.
          </p>

          <div className="auth-art-metrics">
            <div>
              <b>98.4%</b>
              <span>Spatial Harmony</span>
            </div>
            <div className="divider-h" />
            <div>
              <b>9</b>
              <span>Aesthetic Engines</span>
            </div>
            <div className="divider-h" />
            <div>
              <b>&lt; 3.2s</b>
              <span>Diffusion Speed</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Editorial Auth Panel Column */}
      <div className="auth-panel-column">
        <div className="auth-form-card">
          {currentUser ? (
            <div>
              <div className="auth-glyph-ring">
                <CheckCircle2 size={22} />
              </div>
              <h2 className="section-title">Already Signed In</h2>
              <p className="auth-desc">
                You are currently logged in as <strong>{currentUser.username}</strong> ({currentUser.email}).
              </p>
              <div style={{ display: "flex", gap: "12px", marginTop: "20px" }}>
                <Link href="/dashboard" className="button primary" style={{ flex: 1 }}>
                  <span>Go to Dashboard</span>
                  <ArrowRight size={14} />
                </Link>
                <button
                  onClick={handleLogout}
                  className="button ghost"
                  style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}
                >
                  <LogOut size={14} />
                  <span>Sign Out</span>
                </button>
              </div>
            </div>
          ) : (
            <div>
              <div className="auth-glyph-ring">
                <UserIcon size={20} />
              </div>

              <h2 className="section-title">
                {mode === "login" ? "Welcome Back" : "Create Account"}
              </h2>
              <p className="auth-desc">
                {mode === "login"
                  ? "Enter your credentials to access your saved interior projects and studio workspace."
                  : "Join House Muse to design, generate, and orchestrate harmonic architectural spaces."}
              </p>

              {error && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "10px",
                    padding: "12px 14px",
                    background: "rgba(168, 88, 50, 0.1)",
                    border: "1px solid rgba(168, 88, 50, 0.3)",
                    borderRadius: "8px",
                    color: "var(--terracotta)",
                    fontSize: "13px",
                    marginBottom: "16px",
                  }}
                >
                  <AlertCircle size={16} style={{ flexShrink: 0, marginTop: "2px" }} />
                  <span>{error}</span>
                </div>
              )}

              <form onSubmit={handleSubmit} className="auth-form-fields">
                <div>
                  <label
                    style={{
                      display: "block",
                      fontSize: "11px",
                      fontWeight: 600,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      color: "var(--soft-ink)",
                      marginBottom: "6px",
                    }}
                  >
                    Username
                  </label>
                  <input
                    type="text"
                    required
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="e.g. akshata"
                    style={{
                      width: "100%",
                      padding: "11px 14px",
                      borderRadius: "8px",
                      border: "1px solid var(--line)",
                      background: "#FFFFFF",
                      fontSize: "14px",
                      outline: "none",
                      boxSizing: "border-box",
                    }}
                  />
                </div>

                {mode === "register" && (
                  <div>
                    <label
                      style={{
                        display: "block",
                        fontSize: "11px",
                        fontWeight: 600,
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                        color: "var(--soft-ink)",
                        marginBottom: "6px",
                      }}
                    >
                      Email Address
                    </label>
                    <input
                      type="email"
                      required
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="name@example.com"
                      style={{
                        width: "100%",
                        padding: "11px 14px",
                        borderRadius: "8px",
                        border: "1px solid var(--line)",
                        background: "#FFFFFF",
                        fontSize: "14px",
                        outline: "none",
                        boxSizing: "border-box",
                      }}
                    />
                  </div>
                )}

                {mode === "register" && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                    <div>
                      <label
                        style={{
                          display: "block",
                          fontSize: "11px",
                          fontWeight: 600,
                          letterSpacing: "0.08em",
                          textTransform: "uppercase",
                          color: "var(--soft-ink)",
                          marginBottom: "6px",
                        }}
                      >
                        First Name
                      </label>
                      <input
                        type="text"
                        value={firstName}
                        onChange={(e) => setFirstName(e.target.value)}
                        placeholder="Optional"
                        style={{
                          width: "100%",
                          padding: "11px 14px",
                          borderRadius: "8px",
                          border: "1px solid var(--line)",
                          background: "#FFFFFF",
                          fontSize: "14px",
                          outline: "none",
                          boxSizing: "border-box",
                        }}
                      />
                    </div>
                    <div>
                      <label
                        style={{
                          display: "block",
                          fontSize: "11px",
                          fontWeight: 600,
                          letterSpacing: "0.08em",
                          textTransform: "uppercase",
                          color: "var(--soft-ink)",
                          marginBottom: "6px",
                        }}
                      >
                        Last Name
                      </label>
                      <input
                        type="text"
                        value={lastName}
                        onChange={(e) => setLastName(e.target.value)}
                        placeholder="Optional"
                        style={{
                          width: "100%",
                          padding: "11px 14px",
                          borderRadius: "8px",
                          border: "1px solid var(--line)",
                          background: "#FFFFFF",
                          fontSize: "14px",
                          outline: "none",
                          boxSizing: "border-box",
                        }}
                      />
                    </div>
                  </div>
                )}

                <div>
                  <label
                    style={{
                      display: "block",
                      fontSize: "11px",
                      fontWeight: 600,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      color: "var(--soft-ink)",
                      marginBottom: "6px",
                    }}
                  >
                    Password
                  </label>
                  <div className="password-wrap">
                    <input
                      type={showPassword ? "text" : "password"}
                      required
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="Enter password"
                      style={{
                        width: "100%",
                        padding: "11px 40px 11px 14px",
                        borderRadius: "8px",
                        border: "1px solid var(--line)",
                        background: "#FFFFFF",
                        fontSize: "14px",
                        outline: "none",
                        boxSizing: "border-box",
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      aria-label="Toggle password visibility"
                    >
                      {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="button primary full-width"
                >
                  {loading ? (
                    <span>Processing...</span>
                  ) : (
                    <>
                      <span>{mode === "login" ? "Sign In to Studio" : "Create Studio Account"}</span>
                      <ArrowRight size={15} />
                    </>
                  )}
                </button>
              </form>

              <div className="auth-toggle-footer" style={{ marginTop: "24px", textAlign: "center" }}>
                {mode === "login" ? (
                  <p style={{ fontSize: "13px", color: "var(--soft-ink)", margin: 0 }}>
                    New to House Muse?{" "}
                    <button
                      type="button"
                      onClick={() => {
                        setError(null);
                        setMode("register");
                      }}
                      style={{
                        background: "none",
                        border: 0,
                        color: "var(--espresso)",
                        fontWeight: 600,
                        textDecoration: "underline",
                        padding: 0,
                      }}
                    >
                      Create an account
                    </button>
                  </p>
                ) : (
                  <p style={{ fontSize: "13px", color: "var(--soft-ink)", margin: 0 }}>
                    Already have an account?{" "}
                    <button
                      type="button"
                      onClick={() => {
                        setError(null);
                        setMode("login");
                      }}
                      style={{
                        background: "none",
                        border: 0,
                        color: "var(--espresso)",
                        fontWeight: 600,
                        textDecoration: "underline",
                        padding: 0,
                      }}
                    >
                      Sign in here
                    </button>
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
