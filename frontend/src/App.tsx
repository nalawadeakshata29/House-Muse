import { Switch, Route, Link, useLocation } from "wouter";
import { useState, useEffect } from "react";
import { Menu, X, ArrowRight, User as UserIcon } from "lucide-react";
import Home from "./pages/Home";
import Studio from "./pages/Studio";
import Dashboard from "./pages/Dashboard";
import Auth from "./pages/Auth";
import NotFound from "./pages/NotFound";
import "./index.css";

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 19V5H8.5L12 11.5L15.5 5H20V19H16V10.5L13.2 15.5H10.8L8 10.5V19H4Z" fill="currentColor" />
      </svg>
    </span>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [location] = useLocation();
  const [user, setUser] = useState<{ username?: string } | null>(null);
  const isLanding = location === "/";

  useEffect(() => {
    try {
      const stored = localStorage.getItem("house_muse_user");
      if (stored) setUser(JSON.parse(stored));
    } catch {
      setUser(null);
    }
  }, [location]);

  const links = [
    ["Home", "/"],
    ["Dashboard", "/dashboard"],
    ["Studio", "/studio"],
  ];

  return (
    <div className={isLanding ? "app landing-app" : "app product-app"}>
      <header className="site-header">
        <Link href="/" className="brand-lockup" aria-label="House Muse home">
          <BrandMark />
          <span className="wordmark">HOUSE MUSE</span>
        </Link>

        <nav className="desktop-nav" aria-label="Primary navigation">
          {links.map(([label, href]) => (
            <Link
              key={href}
              href={href}
              className={location === href ? "active" : ""}
            >
              {label}
            </Link>
          ))}
        </nav>

        <div className="header-actions">
          {user ? (
            <Link href="/dashboard" className="user-badge" title="Go to Dashboard">
              <UserIcon size={14} />
              <span>{user.username || "Account"}</span>
            </Link>
          ) : (
            <Link href="/auth?mode=login" className="signin-link">
              Sign in
            </Link>
          )}
          <Link href="/studio" className="header-cta">
            <span>New Room</span>
            <ArrowRight size={13} />
          </Link>
        </div>

        <button
          className={open ? "menu-toggle open" : "menu-toggle"}
          aria-label="Toggle menu"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </header>

      {open && (
        <>
          <button
            className="mobile-overlay"
            aria-label="Close menu"
            onClick={() => setOpen(false)}
          />
          <nav className="mobile-sheet">
            {links.map(([label, href]) => (
              <Link
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className={location === href ? "active" : ""}
              >
                {label}
              </Link>
            ))}
            <Link
              href="/auth?mode=login"
              onClick={() => setOpen(false)}
              className="sheet-signin"
            >
              {user ? `Account (${user.username})` : "Sign in"}
            </Link>
          </nav>
        </>
      )}

      {children}
    </div>
  );
}

export default function App() {
  return (
    <Shell>
      <Switch>
        <Route path="/" component={Home} />
        <Route path="/studio" component={Studio} />
        <Route path="/dashboard" component={Dashboard} />
        <Route path="/auth" component={Auth} />
        <Route path="/reset-password/:uid/:token" component={Auth} />
        <Route component={NotFound} />
      </Switch>
    </Shell>
  );
}

